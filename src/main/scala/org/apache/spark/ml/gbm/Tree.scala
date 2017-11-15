package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD

private[gbm] object Tree extends Logging {

  /**
    * Implementation of training a new tree
    *
    * @param instances   instances containing (grad, hess, bins)
    * @param boostConfig boosting configuration
    * @param treeConfig  tree growing configuration
    * @tparam B
    * @return a new tree if any
    */
  def train[B: Integral : ClassTag](instances: RDD[(Double, Double, Array[B])],
                                    boostConfig: BoostConfig,
                                    treeConfig: TreeConfig): Option[TreeModel] = {
    val intB = implicitly[Integral[B]]

    val sc = instances.sparkContext

    instances.persist(boostConfig.storageLevel)

    val root = LearningNode.create(1L)

    var hists = sc.emptyRDD[((Long, Int), Array[Double])]
    val histsCheckpointer = new Checkpointer[((Long, Int), Array[Double])](sc,
      boostConfig.checkpointInterval, boostConfig.storageLevel)

    var minNodeId = 1L
    var numLeaves = 1L
    var finished = false

    if (root.subtreeDepth >= boostConfig.maxDepth) {
      finished = true
    }
    if (numLeaves >= boostConfig.maxLeaves) {
      finished = true
    }

    val logPrefix = s"Iter ${treeConfig.iteration}: Tree ${treeConfig.treeIndex}:"
    logDebug(s"$logPrefix tree building start")

    while (!finished) {
      val depth = root.subtreeDepth

      logDebug(s"$logPrefix Depth $depth: splitting start")

      val instancesWithNodeId = instances.map {
        case (grad, hess, bins) =>
          (root.index(bins.map(intB.toInt)), grad, hess, bins)
      }

      if (minNodeId == 1) {
        hists = computeHists[B](instancesWithNodeId, minNodeId)
      } else {
        val prevHists = hists
        val leftHists = computeHists[B](instancesWithNodeId, minNodeId)
        hists = subtractHists(prevHists, leftHists, boostConfig.minNodeHess)
      }
      histsCheckpointer.update(hists)

      val seed = boostConfig.seed + treeConfig.treeIndex + depth
      val splits = findSplits(hists, boostConfig, treeConfig, seed)

      if (splits.isEmpty) {
        logDebug(s"$logPrefix Depth $depth: no more splits found, tree building finished")
        finished = true

      } else if (numLeaves + splits.size > boostConfig.maxLeaves) {
        logDebug(s"$logPrefix Depth $depth: maxLeaves=${boostConfig.maxLeaves} reached, tree building finished")
        finished = true

      } else {
        val gains = splits.values.map(_.gain).toArray
        logDebug(s"$logPrefix Depth $depth: splitting finished, ${splits.size}/$numLeaves leaves split, " +
          s"avgGain=${gains.sum / gains.length}, minGain=${gains.min}, maxGain=${gains.max}")
        numLeaves += splits.size

        val nodes = root.nodeIterator.filter { node =>
          node.nodeId >= minNodeId && splits.contains(node.nodeId)
        }.toArray

        /** update tree */
        nodes.foreach { node =>
          node.isLeaf = false
          node.split = splits.get(node.nodeId)

          val leftId = node.nodeId << 1
          node.leftNode = Some(LearningNode.create(leftId))
          node.leftNode.get.prediction = node.split.get.leftWeight

          val rightId = leftId + 1
          node.rightNode = Some(LearningNode.create(rightId))
          node.rightNode.get.prediction = node.split.get.rightWeight
        }
      }

      if (root.subtreeDepth >= boostConfig.maxDepth) {
        logDebug(s"$logPrefix maxDepth=${boostConfig.maxDepth} reached, tree building finished")
        finished = true
      }
      if (numLeaves >= boostConfig.maxLeaves) {
        logDebug(s"$logPrefix maxLeaves=${boostConfig.maxLeaves} reached, tree building finished")
        finished = true
      }

      minNodeId <<= 1
    }
    logDebug(s"$logPrefix tree building finished")

    histsCheckpointer.deleteAllCheckpoints()
    histsCheckpointer.unpersistDataSet()
    instances.unpersist(blocking = false)

    if (root.subtreeDepth > 0) {
      Some(TreeModel.createModel(root, boostConfig, treeConfig))
    } else {
      None
    }
  }

  /**
    * Compute the histogram of root node or the left leaves with nodeId greater than minNodeId
    *
    * @param data      instances appended with nodeId, containing (nodeId, grad, hess, bins)
    * @param minNodeId minimum nodeId
    * @tparam B
    * @return histogram data containing (nodeId, columnId, histogram)
    */
  def computeHists[B: Integral : ClassTag](data: RDD[(Long, Double, Double, Array[B])],
                                           minNodeId: Long): RDD[((Long, Int), Array[Double])] = {
    val intB = implicitly[Integral[B]]

    data.filter { case (nodeId, _, _, _) =>
      (nodeId >= minNodeId && nodeId % 2 == 0) || nodeId == 1

    }.flatMap { case (nodeId, grad, hess, bins) =>
      bins.zipWithIndex.map { case (bin, featureId) =>
        ((nodeId, featureId), (bin, grad, hess))
      }

    }.aggregateByKey[Array[Double]](Array.emptyDoubleArray)(
      seqOp = {
        case (hist, (bin, grad, hess)) =>
          val index = intB.toInt(bin) << 1

          if (hist.length < index + 2) {
            val newHist = hist ++ Array.ofDim[Double](index + 2 - hist.length)
            newHist(index) = grad
            newHist(index + 1) = hess
            newHist
          } else {
            hist(index) += grad
            hist(index + 1) += hess
            hist
          }

      }, combOp = {
        case (hist1, hist2) if hist1.length >= hist2.length =>
          var i = 0
          while (i < hist2.length) {
            hist1(i) += hist2(i)
            i += 1
          }
          hist1

        case (hist1, hist2) =>
          var i = 0
          while (i < hist1.length) {
            hist2(i) += hist1(i)
            i += 1
          }
          hist2
      })
  }

  /**
    * Histogram subtraction
    *
    * @param nodeHists   histogram data of parent nodes
    * @param leftHists   histogram data of left leaves
    * @param minNodeHess minimum hess needed for a node
    * @return histogram data of both left and right leaves
    */
  def subtractHists(nodeHists: RDD[((Long, Int), Array[Double])],
                    leftHists: RDD[((Long, Int), Array[Double])],
                    minNodeHess: Double): RDD[((Long, Int), Array[Double])] = {

    leftHists.map { case ((nodeId, featureId), hist) =>
      ((nodeId >> 1, featureId), hist)

    }.join(nodeHists)

      .flatMap { case ((nodeId, featureId), (leftHist, nodeHist)) =>
        require(leftHist.length <= nodeHist.length)

        var i = 0
        while (i < leftHist.length) {
          nodeHist(i) -= leftHist(i)
          i += 1
        }

        val leftNodeId = nodeId << 1
        val rightNodeId = leftNodeId + 1

        ((leftNodeId, featureId), leftHist) ::
          ((rightNodeId, featureId), nodeHist) :: Nil

      }.filter { case ((_, _), hist) =>

      var hessSum = 0.0
      var nnz = 0
      var i = 0
      while (i < hist.length) {
        if (hist(i) != 0 || hist(i + 1) != 0) {
          hessSum += hist(i + 1)
          nnz += 1
        }
        i += 2
      }

      /** leaves with hess no more than minNodeHess * 2 can no grow */
      nnz > 1 && hessSum > minNodeHess * 2
    }
  }

  /**
    * Search the optimal splits on each leaves
    *
    * @param nodeHists   histogram data of leaves nodes
    * @param boostConfig boosting configuration
    * @param treeConfig  tree growing configuration
    * @param seed        random seed for column sampling by level
    * @return optimal splits for each node
    */
  def findSplits(nodeHists: RDD[((Long, Int), Array[Double])],
                 boostConfig: BoostConfig,
                 treeConfig: TreeConfig,
                 seed: Long): Map[Long, Split] = {

    /** column sampling by level */
    val sampledHists = if (boostConfig.colSampleByLevel == 1) {
      nodeHists
    } else {
      nodeHists.sample(false, boostConfig.colSampleByLevel, seed)
    }

    sampledHists.flatMap {
      case ((nodeId, featureId), hist) =>
        val split = Split.split(featureId, hist, boostConfig, treeConfig)
        split.map((nodeId, _))

    }.mapPartitions { it =>
      val splits = mutable.Map[Long, Split]()
      it.foreach { case (nodeId, split) =>
        val s = splits.get(nodeId)
        if (s.isEmpty || split.gain > s.get.gain) {
          splits.update(nodeId, split)
        }
      }
      Iterator.single(splits.toArray)

    }.treeReduce(
      f = {
        case (splits1, splits2) =>
          (splits1 ++ splits2).groupBy(_._1)
            .map { case (nodeId, splits) =>
              (nodeId, splits.map(_._2).maxBy(_.gain))
            }.toArray
      }, depth = boostConfig.aggregationDepth)
      .toMap
  }
}

class TreeModel(val root: Node) extends Serializable {

  lazy val depth: Int = root.subtreeDepth

  lazy val numLeaves: Long = root.numLeaves

  lazy val numNodes: Long = root.numDescendants

  def predict(bins: Array[Int]): Double = root.predict(bins)

  def index(bins: Array[Int]): Long = root.index(bins)

  def computeImportance: Map[Int, Double] = {
    val gains = collection.mutable.Map[Int, Double]()
    root.nodeIterator.foreach {
      case n: InternalNode =>
        val gain = gains.getOrElse(n.featureId, 0.0)
        gains.update(n.featureId, gain + n.gain)

      case _ =>
    }

    val sum = gains.values.sum

    gains.map { case (index, gain) =>
      (index, gain / sum)
    }.toMap
  }
}


private[gbm] object TreeModel {

  def createModel(root: LearningNode,
                  boostMeta: BoostConfig,
                  treeMeta: TreeConfig): TreeModel = {
    val leafIds = root.nodeIterator.filter(_.isLeaf).map(_.nodeId).toArray.sorted
    val node = TreeModel.createNode(root, treeMeta.colsMap, leafIds)
    new TreeModel(node)
  }

  def createNode(node: LearningNode,
                 colsMap: Array[Int],
                 leafIds: Array[Long]): Node = {

    if (node.isLeaf) {
      require(node.leftNode.isEmpty &&
        node.rightNode.isEmpty &&
        node.split.isEmpty)

      val leafId = leafIds.indexOf(node.nodeId)
      new LeafNode(node.prediction, leafId)

    } else {
      require(node.split.nonEmpty &&
        node.leftNode.nonEmpty && node.rightNode.nonEmpty)

      val reindex = colsMap(node.split.get.featureId)

      node.split.get match {
        case s: SeqSplit =>
          new InternalNode(reindex, true, s.missingGoLeft,
            Array(s.threshold), s.gain,
            createNode(node.leftNode.get, colsMap, leafIds),
            createNode(node.rightNode.get, colsMap, leafIds))

        case s: SetSplit =>
          new InternalNode(reindex, false, s.missingGoLeft,
            s.leftSet, s.gain,
            createNode(node.leftNode.get, colsMap, leafIds),
            createNode(node.rightNode.get, colsMap, leafIds))
      }
    }
  }
}
