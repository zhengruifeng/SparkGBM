package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.{specialized => spec}

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
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
  def train[H: Numeric : ClassTag : FromDouble, B: Integral : ClassTag](data: RDD[(H, H, BinVector[B])],
                                                                        boostConfig: BoostConfig,
                                                                        treeConfig: TreeConfig): Option[TreeModel] = {
    val sc = data.sparkContext
    Split.registerKryoClasses(sc)

    var nodeIds = sc.emptyRDD[Long]
    val nodesCheckpointer = new Checkpointer[Long](sc,
      boostConfig.getCheckpointInterval, boostConfig.getStorageLevel)

    var hists = sc.emptyRDD[((Long, Int), Array[H])]
    val histsCheckpointer = new Checkpointer[((Long, Int), Array[H])](sc,
      boostConfig.getCheckpointInterval, boostConfig.getStorageLevel)

    val logPrefix = s"Iter ${treeConfig.iteration}: Tree ${treeConfig.treeIndex}:"
    logInfo(s"$logPrefix tree building start")

    val root = LearningNode.create(1L)

    val prevSplits = mutable.OpenHashMap.empty[Long, Split]

    var minNodeId = 1L
    var numLeaves = 1L
    var finished = false

    while (!finished) {
      val start = System.nanoTime

      val parallelism = boostConfig.getRealParallelism(sc.defaultParallelism)

      val depth = root.subtreeDepth
      logInfo(s"$logPrefix Depth $depth: splitting start, parallelism $parallelism")

      if (minNodeId == 1L) {
        nodeIds = data.map(_ => 1L)
      } else {
        nodeIds = updateNodeIds[H, B](data, nodeIds, prevSplits.toMap)
      }
      nodesCheckpointer.update(nodeIds)


      if (minNodeId == 1L) {
        // direct compute the histogram of root node
        hists = computeHistogram[H, B](data.zip(nodeIds), (n: Long) => true,
          treeConfig.numCols, prevSplits.toMap, parallelism)

      } else {
        // compute the histogram of right leaves
        val rightHists = computeHistogram[H, B](data.zip(nodeIds), (n: Long) => n >= minNodeId && n % 2 == 1L,
          treeConfig.numCols, prevSplits.toMap, parallelism)

        // pre-compute an even split of (nodeId, col) pairs
        val ranges = computeRanges(prevSplits.keys.toArray, treeConfig.numCols, parallelism)

        // compute the histogram of both left leaves and right leaves by subtraction
        hists = subtractHistogram[H](hists, rightHists, boostConfig.getMinNodeHess, ranges, parallelism)
      }
      histsCheckpointer.update(hists)

      // find best splits
      val splits = findSplits[H](hists, boostConfig, treeConfig,
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
                 minNodeId: Long,
                 splits: Map[Long, Split]): Unit = {
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
  def updateNodeIds[H: Numeric : ClassTag, B: Integral : ClassTag](data: RDD[(H, H, BinVector[B])],
                                                                   nodeIds: RDD[Long],
                                                                   splits: Map[Long, Split]): RDD[Long] = {
    data.zip(nodeIds).map {
      case ((_, _, bins), nodeId) =>
        val split = splits.get(nodeId)
        if (split.nonEmpty) {
          val leftNodeId = nodeId << 1
          if (split.get.goLeft[B](bins)) {
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
    * @param splits      splits found in the last round, if any
    * @param parallelism parallelism
    * @tparam H
    * @tparam B
    * @return histogram data containing (nodeId, columnId, histogram)
    */
  def computeHistogram[H: Numeric : ClassTag : FromDouble, B: Integral : ClassTag](data: RDD[((H, H, BinVector[B]), Long)],
                                                                                   f: (Long => Boolean),
                                                                                   numCols: Int,
                                                                                   splits: Map[Long, Split],
                                                                                   parallelism: Int): RDD[((Long, Int), Array[H])] = {
    val filtered = data.filter(t => f(t._2))

    if (splits.isEmpty) {

      computeHistogramImpl[H, B](filtered, numCols, parallelism)

    } else {

      val toH = implicitly[FromDouble[H]]

      val histSums = splits.flatMap { case (nodeId, split) =>
        val leftNodeId = nodeId << 1
        val leftGrad = toH.fromDouble(split.leftGrad)
        val leftHess = toH.fromDouble(split.leftHess)

        val rightNodeId = leftNodeId + 1
        val rightGrad = toH.fromDouble(split.rightGrad)
        val rightHess = toH.fromDouble(split.rightHess)

        Iterator((leftNodeId, (leftGrad, leftHess)),
          (rightNodeId, (rightGrad, rightHess)))
          .filter(t => f(t._1))
      }

      computeHistogramImpl[H, B](filtered, numCols, histSums, parallelism)
    }
  }


  /**
    * Compute the histogram of nodes. Directly aggregate by (grad, hess).
    *
    * @param data        instances appended with nodeId, containing ((grad, hess, bins), nodeId)
    * @param numCols     number of columns
    * @param parallelism parallelism
    * @tparam H
    * @tparam B
    * @return histogram data containing (nodeId, columnId, histogram)
    */
  def computeHistogramImpl[H: Numeric : ClassTag, B: Integral : ClassTag](data: RDD[((H, H, BinVector[B]), Long)],
                                                                          numCols: Int,
                                                                          parallelism: Int): RDD[((Long, Int), Array[H])] = {
    val intB = implicitly[Integral[B]]
    val numH = implicitly[Numeric[H]]


    data.mapPartitions { it =>

      val histSums = mutable.OpenHashMap.empty[Long, (H, H)]

      it.flatMap { case ((grad, hess, bins), nodeId) =>

        val (g, h) = histSums.getOrElse(nodeId, (numH.zero, numH.zero))
        histSums.update(nodeId, (numH.plus(g, grad), numH.plus(h, hess)))

        // ignore zero-index bins
        bins.activeIter.map { case (col, bin) =>
          ((nodeId, col), (bin, grad, hess))
        }
      } ++
        histSums.iterator.flatMap { case (nodeId, (gradSum, hessSum)) =>
          // make sure all (nodeId, col) pairs are taken into account
          // by the way, store sum of hist in zero-index bin
          Iterator.range(0, numCols).map { c =>
            ((nodeId, c), (intB.zero, gradSum, hessSum))
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
    * Compute the histogram of nodes, given sum of hist.
    *
    * @param data        instances appended with nodeId, containing ((grad, hess, bins), nodeId)
    * @param numCols     number of columns
    * @param histSums    pre-computed sum of hist of right leaves, obtained from the stats of splits in the last round
    * @param parallelism parallelism
    * @tparam H
    * @tparam B
    * @return histogram data containing (nodeId, columnId, histogram)
    */
  def computeHistogramImpl[H: Numeric : ClassTag, B: Integral : ClassTag](data: RDD[((H, H, BinVector[B]), Long)],
                                                                          numCols: Int,
                                                                          histSums: Map[Long, (H, H)],
                                                                          parallelism: Int): RDD[((Long, Int), Array[H])] = {
    val intB = implicitly[Integral[B]]

    val sc = data.sparkContext

    // make sure all (nodeId, col) pairs are taken into account
    // by the way, store sum of hist in zero-index bin
    val rdd0 = sc.range(0, numCols)
      .flatMap { col =>
        histSums.iterator.map { case (nodeId, (gradSum, hessSum)) =>
          ((nodeId, col.toInt), (intB.zero, gradSum, hessSum))
        }
      }

    val rdd1 = data.flatMap { case ((grad, hess, bins), nodeId) =>
      // ignore zero-index bins
      bins.activeIter.map { case (col, bin) =>
        ((nodeId, col), (bin, grad, hess))
      }
    }

    sc.union(rdd0, rdd1)
      .aggregateByKey[Array[H]](Array.empty[H], parallelism)(
      seqOp = updateHistArray[H, B],
      combOp = mergeHistArray[H]

    ).map { case ((nodeId, col), hist) =>
      ((nodeId, col), adjustHistArray[H](hist))
    }
  }


  /**
    * Compute the histogram of nodes in a step-wise fashion. First aggregate by (grad, hess, bin), then by (grad, hess).
    *
    * @param data        instances appended with nodeId, containing ((grad, hess, bins), nodeId)
    * @param numCols     number of columns
    * @param parallelism parallelism
    * @tparam H
    * @tparam B
    * @return histogram data containing (nodeId, columnId, histogram)
    */
  def computeHistogramStepWiseImpl[H: Numeric : ClassTag, B: Integral : ClassTag](data: RDD[((H, H, BinVector[B]), Long)],
                                                                                  numCols: Int,
                                                                                  parallelism: Int): RDD[((Long, Int), Array[H])] = {
    val intB = implicitly[Integral[B]]
    val numH = implicitly[Numeric[H]]

    val partitioner1 = new GBMHashPartitioner(partitions = parallelism,
      by = {
        case (nodeId: Long, col: Int, _) => (nodeId, col)
      })

    val partitioner2 = new GBMHashPartitioner(partitions = parallelism)

    data.mapPartitions { it =>

      val histSums = mutable.OpenHashMap.empty[Long, (H, H)]

      it.flatMap { case ((grad, hess, bins), nodeId) =>

        val (g, h) = histSums.getOrElse(nodeId, (numH.zero, numH.zero))
        histSums.update(nodeId, (numH.plus(g, grad), numH.plus(h, hess)))

        // ignore zero-index bins
        bins.activeIter.map { case (col, bin) =>
          ((nodeId, col, bin), (grad, hess))
        }
      } ++
        histSums.iterator.map { case (nodeId, (gradSum, hessSum)) =>
          // sum of hist of each node, set col=-1
          ((nodeId, -1, intB.zero), (gradSum, hessSum))
        }

    }.reduceByKey(
      // aggregate by (nodeId, col, bin), and partition by (nodeId, col)
      partitioner = partitioner1,
      func = {
        case ((grad1, hess1), (grad2, hess2)) =>
          (numH.plus(grad1, grad2), numH.plus(hess1, hess2))
      }

    ).flatMap { case ((nodeId, col, bin), (grad, hess)) =>
      if (col != -1) {
        require(intB.gt(bin, intB.zero))
        Iterator.single(((nodeId, col), (bin, grad, hess)))
      } else {
        // store sum of hist in zero-index bin
        require(intB.equiv(bin, intB.zero))
        Iterator.range(0, numCols).map(c => ((nodeId, c), (bin, grad, hess)))
      }

      // group by (nodeId, col), while keeping the partitioning (except items with bin=zero)
    }.groupByKey(partitioner2)

      .map { case ((nodeId, col), it) =>
        val seq = it.toSeq
        val len = (seq.map(t => intB.toInt(t._1)).max + 1) << 1

        val hist = Array.fill(len)(numH.zero)
        var i = 0
        while (i < seq.length) {
          val (bin, grad, hess) = seq(i)

          val index = intB.toInt(bin) << 1

          // zero-index bin stores the sum of hist
          if (index == 0) {
            hist(0) = numH.plus(hist(0), grad)
            hist(1) = numH.plus(hist(1), hess)
          } else {
            hist(index) = grad
            hist(index + 1) = hess
            hist(0) = numH.minus(hist(0), grad)
            hist(1) = numH.minus(hist(1), hess)
          }

          i += 1
        }

        ((nodeId, col), hist)
      }
  }


  /**
    * Compute the histogram of nodes in a step-wise fashion, given sum of hist.
    *
    * @param data        instances appended with nodeId, containing ((grad, hess, bins), nodeId)
    * @param numCols     number of columns
    * @param histSums    pre-computed sum of hist of right leaves, obtained from the stats of splits in the last round
    * @param parallelism parallelism
    * @tparam H
    * @tparam B
    * @return histogram data containing (nodeId, columnId, histogram)
    */
  def computeHistogramStepWiseImpl[H: Numeric : ClassTag, B: Integral : ClassTag](data: RDD[((H, H, BinVector[B]), Long)],
                                                                                  numCols: Int,
                                                                                  histSums: Map[Long, (H, H)],
                                                                                  parallelism: Int): RDD[((Long, Int), Array[H])] = {

    val intB = implicitly[Integral[B]]
    val numH = implicitly[Numeric[H]]

    val partitioner1 = new GBMHashPartitioner(partitions = parallelism,
      by = {
        case (nodeId: Long, col: Int, _) => (nodeId, col)
      })

    val partitioner2 = new GBMHashPartitioner(partitions = parallelism)

    val sc = data.sparkContext

    // make sure all (nodeId, col) pairs are taken into account
    // by the way, store sum of hist in zero-index bin
    val rdd0 = sc.range(0, numCols)
      .flatMap { col =>
        histSums.iterator.map { case (nodeId, (gradSum, hessSum)) =>
          ((nodeId, col.toInt, intB.zero), (gradSum, hessSum))
        }
      }

    val rdd1 = data.flatMap { case ((grad, hess, bins), nodeId) =>
      // ignore zero-index bins
      bins.activeIter.map { case (col, bin) =>
        ((nodeId, col, bin), (grad, hess))
      }
    }

    sc.union(rdd0, rdd1)
      .reduceByKey(
        // aggregate by (nodeId, col, bin), and partition by (nodeId, col)
        partitioner = partitioner1,
        func = {
          case ((grad1, hess1), (grad2, hess2)) =>
            (numH.plus(grad1, grad2), numH.plus(hess1, hess2))
        }

      ).map { case ((nodeId, col, bin), (grad, hess)) =>
      ((nodeId, col), (bin, grad, hess))

      // group by (nodeId, col), while keeping the partitioning
    }.groupByKey(partitioner2)

      .map { case ((nodeId, col), it) =>

        val seq = it.toSeq
        val len = (seq.map(t => intB.toInt(t._1)).max + 1) << 1

        val hist = Array.fill(len)(numH.zero)
        var i = 0
        while (i < seq.length) {
          val (bin, grad, hess) = seq(i)

          val index = intB.toInt(bin) << 1

          // zero-index bin stores the sum of hist
          if (index == 0) {
            hist(0) = numH.plus(hist(0), grad)
            hist(1) = numH.plus(hist(1), hess)
          } else {
            hist(index) = grad
            hist(index + 1) = hess
            hist(0) = numH.minus(hist(0), grad)
            hist(1) = numH.minus(hist(1), hess)
          }

          i += 1
        }

        ((nodeId, col), hist)
      }
  }


  /**
    * Histogram subtraction
    *
    * @param nodeHists   histogram data of parent nodes
    * @param rightHists  histogram data of right leaves
    * @param minNodeHess minimum hess needed for a node
    * @param ranges      pre-computed even splits of (nodeId, col) pairs
    * @param parallelism parallelism
    * @tparam H
    * @return histogram data of both left and right leaves
    */
  def subtractHistogram[H: Numeric : ClassTag](nodeHists: RDD[((Long, Int), Array[H])],
                                               rightHists: RDD[((Long, Int), Array[H])],
                                               minNodeHess: Double,
                                               ranges: Array[(Long, Int)],
                                               parallelism: Int): RDD[((Long, Int), Array[H])] = {
    val numH = implicitly[Numeric[H]]

    val partitioner = if (ranges.isEmpty) {
      new GBMHashPartitioner(parallelism)
    } else {
      new GBMRangePartitioner[(Long, Int)](ranges)
    }

    rightHists.map { case ((rightNodeId, col), hist) =>
      val parentNodeId = rightNodeId >> 1
      ((parentNodeId, col), hist)

    }.join(nodeHists, partitioner)

      .flatMap { case ((nodeId, col), (rightHist, nodeHist)) =>
        require(rightHist.length <= nodeHist.length)

        var i = 0
        while (i < rightHist.length) {
          nodeHist(i) = numH.minus(nodeHist(i), rightHist(i))
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
        if (!numH.equiv(hist(i), numH.zero) ||
          !numH.equiv(hist(i + 1), numH.zero)) {
          hessSum += numH.toDouble(hist(i + 1))
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
  def findSplits[H: Numeric : ClassTag](nodeHists: RDD[((Long, Int), Array[H])],
                                        boostConfig: BoostConfig,
                                        treeConfig: TreeConfig,
                                        seed: Long): Map[Long, Split] = {
    val sc = nodeHists.sparkContext
    val accTrials = sc.longAccumulator("NumTrials")
    val accSplits = sc.longAccumulator("NumSplits")

    // column sampling by level
    val sampled = if (boostConfig.getColSampleByLevel == 1) {
      nodeHists
    } else {
      nodeHists.sample(false, boostConfig.getColSampleByLevel, seed)
    }

    val splits = sampled.flatMap {
      case ((nodeId, col), hist) =>
        accTrials.add(1L)

        val split = Split.split[H](col, hist, boostConfig, treeConfig)
        if (split.nonEmpty) {
          accSplits.add(1L)
          Iterator.single((nodeId, split.get))
        } else {
          Iterator.empty
        }

    }.reduceByKeyLocally(func = {
      case (split1, split2) =>
        Seq(split1, split2).maxBy(_.gain)

    }).toMap

    logInfo(s"${accTrials.value} trials -> ${accSplits.value} splits -> ${splits.size} best splits")
    splits
  }


  /**
    * Since the (nodeId, columnId) candidate pairs to search optimum split are known before computation
    * we can partition them in a partial sorted way to reduce the communication overhead in following aggregation
    *
    * @param leaves      leaves in current level
    * @param numCols     number of columns
    * @param parallelism parallelism
    * @return even splits of pairs
    */
  def computeRanges(leaves: Array[Long],
                    numCols: Int,
                    parallelism: Int): Array[(Long, Int)] = {
    require(parallelism > 0)

    val numLeaves = leaves.length

    if (leaves.isEmpty || parallelism == 1) {
      Array.empty

    } else {

      val step = numLeaves.toDouble / parallelism

      // parallelism - 1 splitting points
      Array.range(1, parallelism).map { i =>
        val p = i * step
        val n = p.toInt
        val b = p - n
        val c = (b * numCols).round.toInt
        (leaves(n), c)
      }.distinct.sorted
    }
  }


  def updateHistArray[H: Numeric : ClassTag, B: Integral : ClassTag](hist: Array[H],
                                                                     point: (B, H, H)): Array[H] = {
    val intB = implicitly[Integral[B]]
    val numH = implicitly[Numeric[H]]

    val (bin, grad, hess) = point

    val index = intB.toInt(bin) << 1

    if (hist.length < index + 2) {
      val newHist = hist ++ Array.fill(index + 2 - hist.length)(numH.zero)
      newHist(index) = grad
      newHist(index + 1) = hess
      newHist
    } else {
      hist(index) = numH.plus(hist(index), grad)
      hist(index + 1) = numH.plus(hist(index + 1), hess)
      hist
    }
  }


  def mergeHistArray[H: Numeric : ClassTag](hist1: Array[H],
                                            hist2: Array[H]): Array[H] = {
    val numH = implicitly[Numeric[H]]

    var i = 0
    if (hist1.length >= hist2.length) {
      while (i < hist2.length) {
        hist1(i) = numH.plus(hist1(i), hist2(i))
        i += 1
      }
      hist1
    } else {
      while (i < hist1.length) {
        hist2(i) = numH.plus(hist1(i), hist2(i))
        i += 1
      }
      hist2
    }
  }


  def adjustHistArray[H: Numeric : ClassTag](hist: Array[H]): Array[H] = {
    val numH = implicitly[Numeric[H]]

    var i = 2
    while (i < hist.length) {
      // zero-index bin stores the sum of hist
      hist(0) = numH.minus(hist(0), hist(i))
      hist(1) = numH.minus(hist(1), hist(i + 1))
      i += 2
    }

    hist
  }
}


class TreeModel(val root: Node) extends Serializable {

  lazy val depth: Int = root.subtreeDepth

  lazy val numLeaves: Long = root.numLeaves

  lazy val numNodes: Long = root.numDescendants

  private[gbm] def predict[@spec(Byte, Short, Int) B: Integral](bins: BinVector[B]): Double = root.predict[B](bins)

  def predict(vec: Vector, discretizer: Discretizer): Double = root.predict(vec, discretizer)

  private[gbm] def index[@spec(Byte, Short, Int) B: Integral](bins: BinVector[B]): Long = root.index[B](bins)

  def index(vec: Vector, discretizer: Discretizer): Long = root.index(vec, discretizer)

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
