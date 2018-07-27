package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.{specialized => spec}

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD


private[gbm] object Tree extends Logging {

  /**
    * Implementation of training a new tree
    *
    * @param data        instances containing (grad, hess, bins)
    * @param boostConfig boosting configuration
    * @param treeConfig  tree growing configuration
    * @tparam H type of gradient and hessian
    * @tparam B type of bin
    * @return a new tree if any
    */
  def train[H: Numeric : ClassTag : FromDouble, F: Integral : ClassTag, B: Integral : ClassTag](data: RDD[(H, H, GBMVector[F, B])],
                                                                                                boostConfig: BoostConfig,
                                                                                                treeConfig: TreeConfig): Option[TreeModel] = {
    val sc = data.sparkContext

    var nodeIds = sc.emptyRDD[Int]
    val nodesCheckpointer = new Checkpointer[Int](sc,
      boostConfig.getCheckpointInterval, boostConfig.getStorageLevel)

    var hists = sc.emptyRDD[((Int, F), Array[H])]
    val histsCheckpointer = new Checkpointer[((Int, F), Array[H])](sc,
      boostConfig.getCheckpointInterval, boostConfig.getStorageLevel)

    val logPrefix = s"Iter ${treeConfig.iteration}: Tree ${treeConfig.treeIndex}:"
    logInfo(s"$logPrefix tree building start")

    val root = LearningNode.create(1)

    val prevSplits = mutable.OpenHashMap.empty[Int, Split]

    var minNodeId = 1
    var numLeaves = 1
    var finished = false

    while (!finished) {
      val start = System.nanoTime

      val parallelism = boostConfig.getRealParallelism(sc.defaultParallelism)

      val depth = root.subtreeDepth
      logInfo(s"$logPrefix Depth $depth: splitting start, parallelism $parallelism")

      if (minNodeId == 1) {
        nodeIds = data.map(_ => 1)
      } else {
        nodeIds = updateNodeIds[H, F, B](data, nodeIds, prevSplits.toMap)
      }
      nodesCheckpointer.update(nodeIds)


      if (minNodeId == 1) {
        // direct compute the histogram of root node
        hists = computeHistogram[H, F, B](data.zip(nodeIds), (n: Int) => true,
          treeConfig.numCols, parallelism)

      } else {
        // compute the histogram of right leaves
        val rightHists = computeHistogram[H, F, B](data.zip(nodeIds), (n: Int) => n >= minNodeId && n % 2 == 1,
          treeConfig.numCols, parallelism)

        // compute the histogram of both left leaves and right leaves by subtraction
        hists = subtractHistogram[F, H](hists, rightHists, boostConfig.getMinNodeHess, parallelism)
      }
      histsCheckpointer.update(hists)

      // find best splits
      val splits = findSplits[F, H](hists, boostConfig, treeConfig,
        boostConfig.getSeed + treeConfig.treeIndex + depth)

      if (splits.isEmpty) {
        logInfo(s"$logPrefix Depth $depth: no more splits found, tree building finished")
        finished = true

      } else if (numLeaves + splits.size > boostConfig.getMaxLeaves) {
        // choose splits with highest gain score
        val r = (boostConfig.getMaxLeaves - numLeaves).toInt
        val bestSplits = splits.toArray.sortBy(_._2.gain).takeRight(r).toMap

        logInfo(s"$logPrefix Depth $depth: splitting finished, ${bestSplits.size}/$numLeaves leaves split, " +
          s"gain=${bestSplits.values.map(_.gain).sum}, duration ${(System.nanoTime - start) / 1e9} seconds")

        // update tree only by best splits
        updateTree(root, minNodeId, bestSplits)
        finished = true

      } else {
        logInfo(s"$logPrefix Depth $depth: splitting finished, ${splits.size}/$numLeaves leaves split, " +
          s"gain=${splits.values.map(_.gain).sum}, duration ${(System.nanoTime - start) / 1e9} seconds")

        numLeaves += splits.size

        // update tree
        updateTree(root, minNodeId, splits)

        // update last splits
        prevSplits.clear()
        splits.foreach { case (nodeId, split) =>
          prevSplits.update(nodeId, split)
        }
      }

      if (root.subtreeDepth >= boostConfig.getMaxDepth) {
        logInfo(s"$logPrefix maxDepth=${boostConfig.getMaxDepth} reached, tree building finished")
        finished = true
      }
      if (numLeaves >= boostConfig.getMaxLeaves) {
        logInfo(s"$logPrefix maxLeaves=${boostConfig.getMaxLeaves} reached, tree building finished")
        finished = true
      }

      minNodeId <<= 1
    }
    logInfo(s"$logPrefix tree building finished")

    nodesCheckpointer.deleteAllCheckpoints()
    nodesCheckpointer.unpersistDataSet()
    histsCheckpointer.deleteAllCheckpoints()
    histsCheckpointer.unpersistDataSet()

    if (root.subtreeDepth > 0) {
      Some(TreeModel.createModel(root, boostConfig, treeConfig))
    } else {
      None
    }
  }


  /**
    * update tree
    *
    * @param root      root of tree
    * @param minNodeId minimum nodeId for this level
    * @param splits    splits of leaves
    */
  def updateTree(root: LearningNode,
                 minNodeId: Int,
                 splits: Map[Int, Split]): Unit = {
    val nodes = root.nodeIterator.filter { node =>
      node.nodeId >= minNodeId && splits.contains(node.nodeId)
    }.toArray

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


  /**
    * update nodeIds
    *
    * @param data    instances containing (grad, hess, bins)
    * @param nodeIds previous nodeIds
    * @param splits  splits found in the last round
    * @tparam H
    * @tparam B
    * @return updated nodeIds
    */
  def updateNodeIds[H: Numeric : ClassTag, F: Integral : ClassTag, B: Integral : ClassTag](data: RDD[(H, H, GBMVector[F, B])],
                                                                                           nodeIds: RDD[Int],
                                                                                           splits: Map[Int, Split]): RDD[Int] = {
    data.zip(nodeIds).map {
      case ((_, _, bins), nodeId) =>
        val split = splits.get(nodeId)
        if (split.nonEmpty) {
          val leftNodeId = nodeId << 1
          if (split.get.goLeft[F, B](bins)) {
            leftNodeId
          } else {
            leftNodeId + 1
          }
        } else {
          nodeId
        }
    }
  }


  /**
    * Compute the histogram of root node or the right leaves with nodeId greater than minNodeId
    *
    * @param data        instances appended with nodeId, containing ((grad, hess, bins), nodeId)
    * @param f           function to filter nodeIds
    * @param numCols     number of columns
    * @param parallelism parallelism
    * @tparam H
    * @tparam B
    * @return histogram data containing (nodeId, columnId, histogram)
    */
  def computeHistogram[H: Numeric : ClassTag : FromDouble, F: Integral : ClassTag, B: Integral : ClassTag](data: RDD[((H, H, GBMVector[F, B]), Int)],
                                                                                                           f: Int => Boolean,
                                                                                                           numCols: Int,
                                                                                                           parallelism: Int): RDD[((Int, F), Array[H])] = {
    val numH = implicitly[Numeric[H]]
    import numH._

    val intF = implicitly[Integral[F]]

    val intB = implicitly[Integral[B]]

    data.filter(t => f(t._2))
      .mapPartitions { it =>

        val histSums = mutable.OpenHashMap.empty[Int, (H, H)]

        it.flatMap { case ((grad, hess, bins), nodeId) =>

          val (g, h) = histSums.getOrElse(nodeId, (zero, zero))
          histSums.update(nodeId, (g + grad, h + hess))

          // ignore zero-index bins
          bins.activeIter.map { case (col, bin) =>
            ((nodeId, col), (bin, grad, hess))
          }

        } ++
          histSums.iterator.flatMap { case (nodeId, (gradSum, hessSum)) =>
            // make sure all (nodeId, col) pairs are taken into account
            // by the way, store sum of hist in zero-index bin
            Iterator.range(0, numCols).map { c =>
              ((nodeId, intF.fromInt(c)), (intB.zero, gradSum, hessSum))
            }
          }

      }.aggregateByKey[Array[H]](Array.empty[H], parallelism)(
      seqOp = updateHistArray[H, B],
      combOp = mergeHistArray[H]

    ).map { case ((nodeId, col), hist) =>
      ((nodeId, col), adjustHistArray[H](hist))
    }
  }


  /**
    * Histogram subtraction
    *
    * @param nodeHists   histogram data of parent nodes
    * @param rightHists  histogram data of right leaves
    * @param minNodeHess minimum hess needed for a node
    * @param parallelism parallelism
    * @tparam H
    * @return histogram data of both left and right leaves
    */
  def subtractHistogram[F: Integral : ClassTag, H: Numeric : ClassTag](nodeHists: RDD[((Int, F), Array[H])],
                                                                       rightHists: RDD[((Int, F), Array[H])],
                                                                       minNodeHess: Double,
                                                                       parallelism: Int): RDD[((Int, F), Array[H])] = {
    val numH = implicitly[Numeric[H]]
    import numH._

    rightHists.map { case ((rightNodeId, col), hist) =>
      val parentNodeId = rightNodeId >> 1
      ((parentNodeId, col), hist)

    }.join(nodeHists, parallelism)

      .flatMap { case ((nodeId, col), (rightHist, nodeHist)) =>
        require(rightHist.length <= nodeHist.length)

        var i = 0
        while (i < rightHist.length) {
          nodeHist(i) -= rightHist(i)
          i += 1
        }

        val leftNodeId = nodeId << 1
        val rightNodeId = leftNodeId + 1

        ((leftNodeId, col), nodeHist) ::
          ((rightNodeId, col), rightHist) :: Nil

      }.filter { case ((_, _), hist) =>

      var hessSum = 0.0
      var nnz = 0
      var i = 0
      while (i < hist.length) {
        if (hist(i) != zero || hist(i + 1) != zero) {
          hessSum += hist(i + 1).toDouble
          nnz += 1
        }
        i += 2
      }

      // leaves with hess less than minNodeHess * 2 can not grow furthermore
      nnz >= 2 && hessSum >= minNodeHess * 2
    }
  }


  /**
    * Search the optimal splits on each leaves
    *
    * @param nodeHists   histogram data of leaves nodes
    * @param boostConfig boosting configuration
    * @param treeConfig  tree growing configuration
    * @param seed        random seed for column sampling by level
    * @tparam H
    * @return optimal splits for each node
    */
  def findSplits[F: Integral : ClassTag, H: Numeric : ClassTag](nodeHists: RDD[((Int, F), Array[H])],
                                                                boostConfig: BoostConfig,
                                                                treeConfig: TreeConfig,
                                                                seed: Long): Map[Int, Split] = {
    val intF = implicitly[Integral[F]]

    val sc = nodeHists.sparkContext
    val accTrials = sc.longAccumulator("NumTrials")
    val accSplits = sc.longAccumulator("NumSplits")

    // column sampling by level
    val sampled = if (boostConfig.getColSampleByLevel == 1) {
      nodeHists
    } else {
      nodeHists.sample(false, boostConfig.getColSampleByLevel, seed)
    }

    val splits = sampled.mapPartitions { it =>
      val splits = mutable.OpenHashMap.empty[Int, Split]

      it.foreach { case ((nodeId, col), hist) =>
        accTrials.add(1L)

        val split = Split.split[H](intF.toInt(col), hist, boostConfig, treeConfig)
        if (split.nonEmpty) {
          accSplits.add(1L)

          val s = splits.get(nodeId)
          if (s.isEmpty || s.get.gain < split.get.gain) {
            splits.update(nodeId, split.get)
          }
        }
      }

      Iterator.single(splits.toArray)

    }.treeReduce(f = {
      case (splits1, splits2) =>
        (splits1 ++ splits2).groupBy(_._1)
          .map { case (nodeId, s) =>
            (nodeId, s.map(_._2).maxBy(_.gain))
          }.toArray
    }, depth = boostConfig.getAggregationDepth).toMap

    logInfo(s"${accTrials.value} trials -> ${accSplits.value} splits -> ${splits.size} best splits")
    splits
  }


  def updateHistArray[H: Numeric : ClassTag, B: Integral : ClassTag](hist: Array[H],
                                                                     point: (B, H, H)): Array[H] = {
    val numH = implicitly[Numeric[H]]
    import numH._

    val intB = implicitly[Integral[B]]

    val (bin, grad, hess) = point

    val index = intB.toInt(bin) << 1

    if (hist.length < index + 2) {
      val newHist = hist ++ Array.fill(index + 2 - hist.length)(zero)
      newHist(index) = grad
      newHist(index + 1) = hess
      newHist
    } else {
      hist(index) += grad
      hist(index + 1) += hess
      hist
    }
  }


  def mergeHistArray[H: Numeric : ClassTag](hist1: Array[H],
                                            hist2: Array[H]): Array[H] = {
    val numH = implicitly[Numeric[H]]
    import numH._

    var i = 0
    if (hist1.length >= hist2.length) {
      while (i < hist2.length) {
        hist1(i) += hist2(i)
        i += 1
      }
      hist1
    } else {
      while (i < hist1.length) {
        hist2(i) += hist1(i)
        i += 1
      }
      hist2
    }
  }


  def adjustHistArray[H: Numeric : ClassTag](hist: Array[H]): Array[H] = {
    val numH = implicitly[Numeric[H]]
    import numH._

    var i = 2
    while (i < hist.length) {
      // zero-index bin stores the sum of hist
      hist(0) -= hist(i)
      hist(1) -= hist(i + 1)
      i += 2
    }

    hist
  }
}


class TreeModel(val root: Node) extends Serializable {

  lazy val depth: Int = root.subtreeDepth

  lazy val numLeaves: Int = root.numLeaves

  lazy val numNodes: Int = root.numDescendants

  private[gbm] def predict[@spec(Byte, Short, Int) B: Integral](bins: Array[B]): Double = root.predict[B](bins)

  private[gbm] def predict[@spec(Byte, Short, Int) F: Integral, @spec(Byte, Short, Int) B: Integral](bins: GBMVector[F, B]): Double = root.predict[F, B](bins)

  private[gbm] def index[@spec(Byte, Short, Int) B: Integral](bins: Array[B]): Int = root.index[B](bins)

  private[gbm] def index[@spec(Byte, Short, Int) F: Integral, @spec(Byte, Short, Int) B: Integral](bins: GBMVector[F, B]): Int = root.index[F, B](bins)

  def computeImportance: Map[Int, Double] = {
    val gains = mutable.OpenHashMap.empty[Int, Double]

    root.nodeIterator.foreach {
      case n: InternalNode =>
        val g = gains.getOrElse(n.featureId, 0.0)
        gains.update(n.featureId, g + n.gain)

      case _ =>
    }

    gains.toMap
  }
}


private[gbm] object TreeModel {

  def createModel(root: LearningNode,
                  boostMeta: BoostConfig,
                  treeMeta: TreeConfig): TreeModel = {
    val leafIds = root.nodeIterator.filter(_.isLeaf)
      .map(_.nodeId).toArray.sorted
    val node = TreeModel.createNode(root, treeMeta.columns, leafIds)
    new TreeModel(node)
  }


  def createNode(node: LearningNode,
                 columns: Array[Int],
                 leafIds: Array[Int]): Node = {

    if (node.isLeaf) {
      require(node.leftNode.isEmpty &&
        node.rightNode.isEmpty &&
        node.split.isEmpty)

      val leafId = leafIds.indexOf(node.nodeId)
      new LeafNode(node.prediction, leafId)

    } else {
      require(node.split.nonEmpty &&
        node.leftNode.nonEmpty && node.rightNode.nonEmpty)

      val reindex = columns(node.split.get.featureId)

      node.split.get match {
        case SeqSplit(_, missingGo, threshold, left, gain, _) =>
          new InternalNode(reindex, true, missingGo,
            Array(threshold), left, gain,
            createNode(node.leftNode.get, columns, leafIds),
            createNode(node.rightNode.get, columns, leafIds))


        case SetSplit(_, missingGo, set, left, gain, _) =>
          new InternalNode(reindex, false, missingGo,
            set, left, gain,
            createNode(node.leftNode.get, columns, leafIds),
            createNode(node.rightNode.get, columns, leafIds))
      }
    }
  }
}
