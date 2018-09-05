package org.apache.spark.ml.gbm

import java.{util => ju}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD


private[gbm] object Tree extends Serializable with Logging {


  /**
    *
    * @param data      instances containing (bins, treeIds, grad-hess), grad&hess is recurrent for compression. i.e
    *                  treeIds = [t1,t2,t5,t6], grad-hess = [g1,h1,g2,h2] -> {t1:(g1,h1), t2:(g2,h2), t5:(g1,h1), t6:(g2,h2)}
    * @param boostConf boosting configure
    * @param baseConf  trees-building configure
    * @return tree models
    */
  def train[T, N, C, B, H](data: RDD[(KVVector[C, B], Array[T], Array[H])],
                           boostConf: BoostConfig,
                           baseConf: BaseConfig)
                          (implicit ct: ClassTag[T], int: Integral[T],
                           cn: ClassTag[N], inn: Integral[N],
                           cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                           cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                           ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Array[TreeModel] = {
    val sc = data.sparkContext

    var nodeIds = sc.emptyRDD[Array[N]]
    val nodesCheckpointer = new Checkpointer[Array[N]](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel)

    var hists = sc.emptyRDD[((T, N, C), KVVector[B, H])]
    val histsCheckpointer = new Checkpointer[((T, N, C), KVVector[B, H])](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel)

    val logPrefix = s"Iteration ${baseConf.iteration}:"
    logInfo(s"$logPrefix trees building start")

    val splitRNG = new Random(boostConf.getSeed)

    val prevSplits = mutable.OpenHashMap.empty[(T, N), Split]

    val roots = Array.fill(baseConf.numTrees)(LearningNode.create(1))
    val numLeaves = Array.fill(baseConf.numTrees)(1)
    val finished = Array.fill(baseConf.numTrees)(false)

    var minNodeId = inn.one
    var depth = 0

    val parallelism = boostConf.getRealParallelism(boostConf.getReduceParallelism, sc.defaultParallelism)
    var treeIds = Array.range(0, baseConf.numTrees)

    while (finished.contains(false) && depth <= boostConf.getMaxDepth) {
      val start = System.nanoTime

      logInfo(s"$logPrefix Depth $depth: splitting start")

      if (inn.equiv(minNodeId, inn.one)) {
        nodeIds = data.map { case (_, treeIds, _) => Array.fill(treeIds.length)(inn.one) }
        nodeIds.setName(s"NodeIds (Iteration ${baseConf.iteration}, depth $depth)")
      } else {
        nodeIds = updateNodeIds[T, N, C, B, H](data, nodeIds, prevSplits.toMap)
        nodeIds.setName(s"NodeIds (Iteration ${baseConf.iteration}, depth $depth)")
        nodesCheckpointer.update(nodeIds)
      }

      val partitioner = if (treeIds.length * boostConf.getNumCols *
        boostConf.getColSampleByTree * boostConf.getColSampleByLevel < parallelism * 4) {
        // partitioning only by treeId-colId tends to cause imbalance
        new HashPartitioner(parallelism)
      } else {
        new HistogramPratitioner[T, C](parallelism, boostConf.getNumCols, treeIds)
      }
      logInfo(s"$logPrefix Depth $depth: partitioner $partitioner")

      //      val partitioner = new HashPartitioner(parallelism)

      if (inn.equiv(minNodeId, inn.one)) {
        // direct compute the histogram of roots
        hists = computeHistograms[T, N, C, B, H](data.zip(nodeIds), boostConf, baseConf, (n: N) => true, partitioner)
      } else {
        // compute the histogram of right leaves
        val rightHists = computeHistograms[T, N, C, B, H](data.zip(nodeIds), boostConf, baseConf,
          (n: N) => inn.gteq(n, minNodeId) && inn.equiv(inn.rem(n, inn.fromInt(2)), inn.one), partitioner)

        // compute the histogram of both left leaves and right leaves by subtraction
        hists = subtractHistograms[T, N, C, B, H](hists, rightHists, boostConf, partitioner)
      }
      hists.setName(s"Histograms (Iteration ${baseConf.iteration}, depth $depth)")
      histsCheckpointer.update(hists)

      prevSplits.clear()

      // find best splits
      val splits = findSplits[T, N, C, B, H](hists, boostConf, baseConf, depth, splitRNG.nextLong)

      if (splits.isEmpty) {
        logInfo(s"$logPrefix Depth $depth: no more splits found, trees building finished")
        Iterator.range(0, finished.length).foreach(finished(_) = true)

      } else {

        val splitsByTree = splits.toArray
          .map { case ((treeId, nodeId), split) => (treeId, nodeId, split) }
          .groupBy(_._1)
          .mapValues(_.map(t => (t._2, t._3)))

        var numFinished = 0

        Iterator.range(0, baseConf.numTrees)
          .filterNot(finished).foreach { treeId =>
          val array = splitsByTree.getOrElse(int.fromInt(treeId), Array.empty[(N, Split)])

          if (array.isEmpty) {
            finished(treeId) = true
            numFinished += 1

          } else {
            val prevNumLeaves = numLeaves(treeId)
            if (prevNumLeaves + array.length >= boostConf.getMaxLeaves) {
              val r = boostConf.getMaxLeaves - prevNumLeaves
              array.sortBy(_._2.gain).takeRight(r)
                .foreach { case (nodeId, split) => prevSplits.update((int.fromInt(treeId), nodeId), split) }
              finished(treeId) = true
              numFinished += 1

            } else {
              numLeaves(treeId) += array.length
              array.foreach { case (nodeId, split) => prevSplits.update((int.fromInt(treeId), nodeId), split) }
            }
          }
        }

        updateTrees[T, N](roots, minNodeId, prevSplits.toMap)

        treeIds = prevSplits.keys.map(_._1).map(int.toInt).toArray.distinct.sorted

        logInfo(s"$logPrefix Depth $depth: splitting finished, $numFinished trees building finished," +
          s" ${prevSplits.size} leaves split, total gain=${prevSplits.values.map(_.gain).sum}," +
          s" duration ${(System.nanoTime - start) / 1e9} seconds")
      }

      minNodeId = inn.plus(minNodeId, minNodeId)
      depth += 1
    }

    if (depth >= boostConf.getMaxDepth) {
      logInfo(s"$logPrefix maxDepth=${boostConf.getMaxDepth} reached, trees building finished")
    } else {
      logInfo(s"$logPrefix trees building finished")
    }

    nodesCheckpointer.cleanup()
    histsCheckpointer.cleanup()

    roots.map(TreeModel.createModel)
  }


  /**
    * update trees
    *
    * @param roots     roots of trees
    * @param minNodeId minimum nodeId for this level
    * @param splits    splits of leaves
    */
  def updateTrees[T, N](roots: Array[LearningNode],
                        minNodeId: N,
                        splits: Map[(T, N), Split])
                       (implicit ct: ClassTag[T], int: Integral[T],
                        cn: ClassTag[N], inn: Integral[N]): Unit = {
    roots.zipWithIndex.foreach {
      case (root, treeId) =>
        val nodes = root.nodeIterator.filter { node =>
          node.nodeId >= inn.toInt(minNodeId) &&
            splits.contains((int.fromInt(treeId), inn.fromInt(node.nodeId)))
        }.toArray

        nodes.foreach { node =>
          node.isLeaf = false
          node.split = splits.get((int.fromInt(treeId), inn.fromInt(node.nodeId)))

          val leftNodeId = node.nodeId << 1
          node.leftNode = Some(LearningNode.create(leftNodeId))
          node.leftNode.get.prediction = node.split.get.leftWeight

          val rightNodeId = leftNodeId + 1
          node.rightNode = Some(LearningNode.create(rightNodeId))
          node.rightNode.get.prediction = node.split.get.rightWeight
        }
    }
  }


  /**
    * update nodeIds
    *
    * @param data    instances containing (bins, treeIds, grad-hess)
    * @param nodeIds previous nodeIds
    * @param splits  splits found in the last round
    * @return updated nodeIds
    */
  def updateNodeIds[T, N, C, B, H](data: RDD[(KVVector[C, B], Array[T], Array[H])],
                                   nodeIds: RDD[Array[N]],
                                   splits: Map[(T, N), Split])
                                  (implicit ct: ClassTag[T], int: Integral[T],
                                   cn: ClassTag[N], inn: Integral[N],
                                   cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                   cb: ClassTag[B], inb: Integral[B],
                                   ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[Array[N]] = {
    data.zip(nodeIds).map { case ((bins, treeIds, _), nodeIds) =>
      treeIds.zip(nodeIds).map { case (treeId, nodeId) =>
        val split = splits.get((treeId, nodeId))
        if (split.nonEmpty) {
          val leftNodeId = inn.plus(nodeId, nodeId)
          if (split.get.goLeft[B](bins.apply)) {
            leftNodeId
          } else {
            inn.plus(leftNodeId, inn.one)
          }
        } else {
          nodeId
        }
      }
    }
  }


  /**
    * Compute the histogram of root node or the right leaves with nodeId greater than minNodeId
    *
    * @param data instances appended with nodeId, containing ((bins, treeIds, grad-hess), nodeIds)
    * @param f    function to filter nodeIds
    * @return histogram data containing (treeId, nodeId, columnId, histogram)
    */
  def computeHistograms[T, N, C, B, H](data: RDD[((KVVector[C, B], Array[T], Array[H]), Array[N])],
                                       boostConf: BoostConfig,
                                       baseConf: BaseConfig,
                                       f: N => Boolean,
                                       partitioner: Partitioner)
                                      (implicit ct: ClassTag[T], int: Integral[T],
                                       cn: ClassTag[N], inn: Integral[N],
                                       cc: ClassTag[C], inc: Integral[C],
                                       cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                       ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {
    import PairRDDFunctions._

    data.mapPartitions { iter =>
      val histSums = mutable.OpenHashMap.empty[(T, N), (H, H)]

      iter.flatMap { case ((bins, treeIds, gradHess), nodeIds) =>
        val gradSize = gradHess.length >> 1
        Iterator.range(0, treeIds.length)
          .filter(i => f(nodeIds(i)))
          .flatMap { i =>
            val treeId = treeIds(i)
            val nodeId = nodeIds(i)
            val indexGrad = (i % gradSize) << 1
            val grad = gradHess(indexGrad)
            val hess = gradHess(indexGrad + 1)

            val (g, h) = histSums.getOrElse((treeId, nodeId), (nuh.zero, nuh.zero))
            histSums.update((treeId, nodeId), (nuh.plus(g, grad), nuh.plus(h, hess)))

            val selector = baseConf.getSelector(int.toInt(treeId))

            // ignore zero-index bins
            bins.activeIter
              .filter { case (colId, _) => selector.contains(colId) }
              .map { case (colId, bin) => ((treeId, nodeId, colId), (bin, grad, hess)) }
          }

      } ++ histSums.iterator.flatMap { case ((treeId, nodeId), (gradSum, hessSum)) =>
        val selector = baseConf.getSelector(int.toInt(treeId))

        // make sure all available (treeId, nodeId, colId) tuples are taken into account
        // by the way, store sum of hist in zero-index bin
        Iterator.range(0, boostConf.getNumCols).filter(selector.contains)
          .map { colId => ((treeId, nodeId, inc.fromInt(colId)), (inb.zero, gradSum, hessSum)) }
      }

    }.aggregatePartitionsByKey(KVVector.empty[B, H])(
      seqOp = {
        case (hist, (bin, grad, hess)) =>
          val indexGrad = inb.plus(bin, bin)
          val indexHess = inb.plus(indexGrad, inb.one)
          hist.plus(indexHess, hess)
            .plus(indexGrad, grad)
      },
      combOp = _ plus _

    ).mapValues { hist =>
      var nzGradSum = nuh.zero
      var nzHessSum = nuh.zero

      hist.activeIter.foreach { case (bin, v) =>
        if (inb.gt(bin, inb.one)) {
          if (inb.equiv(inb.rem(bin, inb.fromInt(2)), inb.zero)) {
            nzGradSum = nuh.plus(nzGradSum, v)
          } else {
            nzHessSum = nuh.plus(nzHessSum, v)
          }
        }
      }

      hist.minus(inb.zero, nzGradSum)
        .minus(inb.one, nzHessSum)
        .compressed

    }.reduceByKey(partitioner, _ plus _)
  }


  /**
    * Histogram subtraction
    *
    * @param nodeHists  histogram data of parent nodes
    * @param rightHists histogram data of right leaves
    * @return histogram data of both left and right leaves
    */
  def subtractHistograms[T, N, C, B, H](nodeHists: RDD[((T, N, C), KVVector[B, H])],
                                        rightHists: RDD[((T, N, C), KVVector[B, H])],
                                        boostConf: BoostConfig,
                                        partitioner: Partitioner)
                                       (implicit ct: ClassTag[T], int: Integral[T],
                                        cn: ClassTag[N], inn: Integral[N],
                                        cc: ClassTag[C], inc: Integral[C],
                                        cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                        ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {
    val threshold = neh.fromDouble(boostConf.getMinNodeHess * 2)

    // Only if the partitioner is a HistogramPratitioner, we can preserves the
    // partitioning after changing the nodeId in key
    val preservesPartitioning1 =
    rightHists.partitioner.nonEmpty &&
      rightHists.partitioner.get.isInstanceOf[HistogramPratitioner[_, _]]

    val preservesPartitioning2 = partitioner.isInstanceOf[HistogramPratitioner[_, _]]

    rightHists.mapPartitions(f = { iter =>
      iter.map { case ((treeId, rightNodeId, colId), rightHist) =>
        val parentNodeId = inn.quot(rightNodeId, inn.fromInt(2))
        ((treeId, parentNodeId, colId), rightHist)
      }
    }, preservesPartitioning1)

      .join(nodeHists, partitioner)

      .mapPartitions(f = { iter =>

        iter.flatMap { case ((treeId, parentNodeId, colId), (rightHist, parentHist)) =>
          require(rightHist.len <= parentHist.len)
          val leftNodeId = inn.plus(parentNodeId, parentNodeId)
          val rightNodeId = inn.plus(leftNodeId, inn.one)
          val leftHist = parentHist.minus(rightHist)

          ((treeId, leftNodeId, colId), leftHist) ::
            ((treeId, rightNodeId, colId), rightHist) :: Nil

        }.filter { case (_, hist) =>
          // leaves with hess less than minNodeHess * 2 can not grow furthermore
          val hessSum = hist.activeIter.filter { case (b, _) =>
            inb.equiv(inb.rem(b, inb.fromInt(2)), inb.one)
          }.map(_._2).sum

          nuh.gteq(hessSum, threshold) && hist.nnz > 2
        }

      }, preservesPartitioning2)
  }


  /**
    * Search the optimal splits on each leaves
    *
    * @param nodeHists histogram data of leaves nodes
    * @param seed      random seed for column sampling by level
    * @return optimal splits for each node
    */
  def findSplits[T, N, C, B, H](nodeHists: RDD[((T, N, C), KVVector[B, H])],
                                boostConf: BoostConfig,
                                baseConf: BaseConfig,
                                depth: Int,
                                seed: Long)
                               (implicit ct: ClassTag[T], int: Integral[T],
                                cn: ClassTag[N], inn: Integral[N],
                                cc: ClassTag[C], inc: Integral[C],
                                cb: ClassTag[B], inb: Integral[B],
                                ch: ClassTag[H], nuh: Numeric[H], fdh: NumericExt[H]): Map[(T, N), Split] = {
    val sc = nodeHists.sparkContext

    // column sampling by level
    val sampled = if (boostConf.getColSampleByLevel == 1) {
      nodeHists
    } else {
      nodeHists.sample(false, boostConf.getColSampleByLevel, seed)
    }

    val parallelism = boostConf.getRealParallelism(boostConf.getTrialParallelism, sc.defaultParallelism)
    val repartitioned = if (parallelism == sampled.getNumPartitions) {
      sampled
    } else if (parallelism < sampled.getNumPartitions) {
      sampled.coalesce(parallelism, false)
    } else {
      import RDDFunctions._
      sampled.extendPartitions(parallelism)
    }

    val (splits, Array(numTrials, numSplits, numDenses, sum, nnz)) =
      repartitioned.mapPartitions { iter =>
        val splits = mutable.OpenHashMap.empty[(T, N), Split]

        // numTrials, numSplits, numDenseHist, sumHistLen, nnz
        val metrics = Array.ofDim[Double](5)

        iter.foreach { case ((treeId, nodeId, colId), hist) =>
          metrics(0) += 1
          metrics(3) += hist.len
          metrics(4) += hist.nnz
          if (hist.isDense) {
            metrics(2) += 1
          }

          val split = Split.split[H](inc.toInt(colId), hist.toArray, boostConf, baseConf)

          if (split.nonEmpty) {
            metrics(1) += 1
            val prevSplit = splits.get((treeId, nodeId))
            if (prevSplit.isEmpty || prevSplit.get.gain < split.get.gain) {
              splits.update((treeId, nodeId), split.get)
            }
          }
        }

        Iterator.single((splits.toArray, metrics))

      }.treeReduce(f = {
        case ((splits1, metrics1), (splits2, metrics2)) =>
          val splits = (splits1 ++ splits2).groupBy(_._1)
            .mapValues(_.map(_._2).maxBy(_.gain)).toArray
          Iterator.range(0, metrics1.length).foreach(i => metrics1(i) += metrics2(i))
          (splits, metrics1)
      }, boostConf.getAggregationDepth)


    logInfo(s"Depth $depth: $numTrials trials -> $numSplits splits -> ${splits.length} best splits")
    logInfo(s"Depth $depth: Fraction of sparse histograms: ${1 - numDenses / numTrials}, " +
      s"sparsity of histogram: ${1 - nnz / sum}")

    splits.toMap
  }
}


/**
  * Partitioner that ignore nodeId in key (treeId, nodeId, colId), this will avoid unnecessary shuffle
  * in histogram subtraction and reduce communication cost in following split-searching.
  */
class HistogramPratitioner[T, C](val numPartitions: Int,
                                 val numCols: Int,
                                 val treeIds: Array[Int])
                                (implicit ct: ClassTag[T], int: Integral[T],
                                 cc: ClassTag[C], inc: Integral[C]) extends Partitioner {
  require(numPartitions > 0)
  require(numCols > 0)
  require(treeIds.nonEmpty)
  require(treeIds.forall(_ >= 0))
  require(Iterator.range(0, treeIds.length - 1).forall(i => treeIds(i) < treeIds(i + 1)))

  private val hash = numPartitions * (numCols + treeIds.sum + treeIds.min + treeIds.max)

  private val treeInterval = numPartitions.toDouble / treeIds.length

  override def getPartition(key: Any): Int = key match {
    case null => 0

    case (treeId: T, _, colId: C) =>
      val i = ju.Arrays.binarySearch(treeIds, int.toInt(treeId))
      require(i >= 0)
      val p = (i + inc.toDouble(colId) / numCols) * treeInterval
      math.min(numPartitions - 1, p.toInt)
  }

  override def equals(other: Any): Boolean = other match {
    case h: HistogramPratitioner[_, _] =>
      numPartitions == h.numPartitions && numCols == h.numCols &&
        treeIds.length == h.treeIds.length &&
        Iterator.range(0, treeIds.length).forall(i => treeIds(i) == h.treeIds(i))

    case _ =>
      false
  }

  override def hashCode: Int = hash

  override def toString: String = {
    s"HistogramPratitioner(numPartitions=$numPartitions, numCols=$numCols, " +
      s"treeIds=${treeIds.mkString("[", ",", "]")})"
  }
}




