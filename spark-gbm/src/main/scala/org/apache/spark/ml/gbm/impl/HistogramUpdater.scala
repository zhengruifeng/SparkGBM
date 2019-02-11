package org.apache.spark.ml.gbm.impl

import java.{util => ju}

import scala.collection.mutable
import scala.reflect.ClassTag

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.rdd._
import org.apache.spark.ml.gbm.rdd.PairRDDFunctions._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.ml.gbm._
import org.apache.spark.rdd.RDD
import org.apache.spark._


private[gbm] trait HistogramUpdater[T, N, C, B, H] extends Logging {

  /**
    * Compute histograms of given level
    */
  def update(binVecBlocks: RDD[KVMatrix[C, B]],
             treeIdBlocks: RDD[ArrayBlock[T]],
             nodeIdBlocks: RDD[ArrayBlock[N]],
             gradBlocks: RDD[ArrayBlock[H]],
             boostConf: BoostConfig,
             bcBoostConf: Broadcast[BoostConfig],
             treeConf: TreeConfig,
             bcTreeConf: Broadcast[TreeConfig],
             extraColSelector: Option[Selector],
             splits: Map[(T, N), Split],
             depth: Int)
            (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
             cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
             cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
             cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
             ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])]


  /**
    * Clear after each level
    */
  def clear(): Unit = {}


  /**
    * Clear after each iteration
    */
  def destroy(): Unit = {}
}


private[gbm] class BasicHistogramUpdater[T, N, C, B, H] extends HistogramUpdater[T, N, C, B, H] {

  override def update(binVecBlocks: RDD[KVMatrix[C, B]],
                      treeIdBlocks: RDD[ArrayBlock[T]],
                      nodeIdBlocks: RDD[ArrayBlock[N]],
                      gradBlocks: RDD[ArrayBlock[H]],
                      boostConf: BoostConfig,
                      bcBoostConf: Broadcast[BoostConfig],
                      treeConf: TreeConfig,
                      bcTreeConf: Broadcast[TreeConfig],
                      extraColSelector: Option[Selector],
                      splits: Map[(T, N), Split],
                      depth: Int)
                     (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                      cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                      cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                      cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                      ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {

    val treeNodeIds = if (depth == 0) {
      Array.tabulate(boostConf.getNumTrees)(t => (int.fromInt(t), inn.one))

    } else {
      splits.keysIterator
        .flatMap { case (treeId, nodeId) =>
          val leftNodeId = inn.plus(nodeId, nodeId)
          val rightNodeId = inn.plus(leftNodeId, inn.one)
          Iterator.apply((treeId, leftNodeId), (treeId, rightNodeId))
        }.toArray.sorted
    }

    val numCols = treeConf.getNumCols.getOrElse(boostConf.getNumCols)
    val partitioner = new RangePratitioner[T, N, C](boostConf.getRealHistogramParallelism, numCols, treeNodeIds)
    logInfo(s"Iter ${treeConf.iteration}: Depth $depth, partitioner $partitioner")


    val minNodeId = inn.fromInt(1 << depth)
    HistogramUpdater.computeHistograms[T, N, C, B, H](binVecBlocks, treeIdBlocks, nodeIdBlocks, gradBlocks,
      boostConf, bcBoostConf, treeConf, bcTreeConf,
      extraColSelector, (n: N) => inn.gteq(n, minNodeId), partitioner, depth)
  }
}


private[gbm] class SubtractHistogramUpdater[T, N, C, B, H] extends HistogramUpdater[T, N, C, B, H] {

  private var delta = 1.0

  private var prevHistograms: RDD[((T, N, C), KVVector[B, H])] = null

  private var checkpointer: Checkpointer[((T, N, C), KVVector[B, H])] = null

  override def update(binVecBlocks: RDD[KVMatrix[C, B]],
                      treeIdBlocks: RDD[ArrayBlock[T]],
                      nodeIdBlocks: RDD[ArrayBlock[N]],
                      gradBlocks: RDD[ArrayBlock[H]],
                      boostConf: BoostConfig,
                      bcBoostConf: Broadcast[BoostConfig],
                      treeConf: TreeConfig,
                      bcTreeConf: Broadcast[TreeConfig],
                      extraColSelector: Option[Selector],
                      splits: Map[(T, N), Split],
                      depth: Int)
                     (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                      cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                      cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                      cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                      ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {

    val sc = binVecBlocks.sparkContext
    val minNodeId = inn.fromInt(1 << depth)

    val (treeIds, prevPartitioner) = if (depth == 0) {
      checkpointer = new Checkpointer[((T, N, C), KVVector[B, H])](sc,
        boostConf.getCheckpointInterval, boostConf.getStorageLevel1)

      (Array.tabulate(boostConf.getNumTrees)(int.fromInt), None)

    } else {
      (splits.keysIterator.map(_._1).toArray.distinct.sorted, prevHistograms.partitioner)
    }

    val partitioner = HistogramUpdater.updatePartitioner[T, N, C](boostConf, treeConf,
      treeIds, depth, boostConf.getRealHistogramParallelism, prevPartitioner)
    logInfo(s"Iter ${treeConf.iteration}: Depth $depth, minNodeId $minNodeId, partitioner $partitioner")

    val histograms = if (depth == 0) {
      // direct compute the histogram of roots
      HistogramUpdater.computeHistograms[T, N, C, B, H](binVecBlocks, treeIdBlocks, nodeIdBlocks, gradBlocks,
        boostConf, bcBoostConf, treeConf, bcTreeConf,
        extraColSelector, (n: N) => inn.gt(n, inn.zero), partitioner, depth)

    } else {
      // compute the histogram of right leaves
      val rightHistograms = HistogramUpdater.computeHistograms[T, N, C, B, H](binVecBlocks, treeIdBlocks, nodeIdBlocks, gradBlocks,
        boostConf, bcBoostConf, treeConf, bcTreeConf,
        extraColSelector, (n: N) => inn.gteq(n, minNodeId) && inn.equiv(inn.rem(n, inn.fromInt(2)), inn.one), partitioner, depth)
        .setName(s"Iter ${treeConf.iteration}, depth: $depth: Right Leaves Histograms")

      // compute the histogram of both left leaves and right leaves by subtraction
      HistogramUpdater.subtractHistograms[T, N, C, B, H](prevHistograms, rightHistograms, boostConf, partitioner)
    }

    val numCols = treeConf.getNumCols.getOrElse(boostConf.getNumCols)
    val expectedSize = (boostConf.getNumTrees << depth) * numCols * boostConf.getColSampleRateByTree * delta

    // cut off lineage if size is small
    if (expectedSize < (1 << 16)) {
      val numPartitions = histograms.getNumPartitions
      val collected = histograms.collect

      val exactSize = collected.length
      logInfo(s"Iter ${treeConf.iteration}, depth: $depth, " +
        s"expectedSize: ${expectedSize.toInt}, exactSize: $exactSize, delta: $delta")
      delta *= exactSize / expectedSize

      val smallHistograms = sc.parallelize(collected, numPartitions)

      prevHistograms = smallHistograms
      smallHistograms

    } else {

      prevHistograms = histograms
      checkpointer.update(histograms)
      histograms
    }
  }

  override def destroy(): Unit = {
    if (checkpointer != null) {
      checkpointer.clear(false)
      checkpointer = null
    }
    prevHistograms = null
    delta = 1.0
  }
}


private[gbm] class VoteHistogramUpdater[T, N, C, B, H] extends HistogramUpdater[T, N, C, B, H] {

  private var cleaner: ResourceCleaner = null

  private var delta = 1.0

  override def update(binVecBlocks: RDD[KVMatrix[C, B]],
                      treeIdBlocks: RDD[ArrayBlock[T]],
                      nodeIdBlocks: RDD[ArrayBlock[N]],
                      gradBlocks: RDD[ArrayBlock[H]],
                      boostConf: BoostConfig,
                      bcBoostConf: Broadcast[BoostConfig],
                      treeConf: TreeConfig,
                      bcTreeConf: Broadcast[TreeConfig],
                      extraColSelector: Option[Selector],
                      splits: Map[(T, N), Split],
                      depth: Int)
                     (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                      cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                      cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                      cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                      ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {

    val sc = binVecBlocks.sparkContext

    val treeNodeIds = if (depth == 0) {
      cleaner = new ResourceCleaner

      Array.tabulate(boostConf.getNumTrees)(t => (int.fromInt(t), inn.one))

    } else {
      splits.keysIterator
        .flatMap { case (treeId, nodeId) =>
          val leftNodeId = inn.plus(nodeId, nodeId)
          val rightNodeId = inn.plus(leftNodeId, inn.one)
          Iterator.apply((treeId, leftNodeId), (treeId, rightNodeId))
        }.toArray.sorted
    }

    val numCols = treeConf.getNumCols.getOrElse(boostConf.getNumCols)
    val partitioner = new RangePratitioner[T, N, C](boostConf.getRealHistogramParallelism, numCols, treeNodeIds)
    logInfo(s"Iter ${treeConf.iteration}: Depth $depth, partitioner $partitioner")


    val minNodeId = inn.fromInt(1 << depth)
    val topK = boostConf.getTopK
    val top2K = topK << 1
    val expectedSize = (boostConf.getNumTrees << depth) * top2K * delta

    val localHistograms = HistogramUpdater.computeLocalNonMissingOnlyHistograms[T, N, C, B, H](binVecBlocks, treeIdBlocks, nodeIdBlocks, gradBlocks,
      bcBoostConf, bcTreeConf, extraColSelector, (n: N) => inn.gteq(n, minNodeId))
      .setName(s"Iter ${treeConf.iteration}, depth: $depth: Local Histograms")
    localHistograms.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(localHistograms)


    val localVoted = localHistograms.mapPartitions { iter =>
      val boostConf = bcBoostConf.value
      val treeConf = bcTreeConf.value

      val gainIter = Utils.validateKeyOrdering(iter).flatMap {
        case ((treeId, nodeId, colId), hist) =>
          Split.split[H](inc.toInt(colId), hist.toArray, boostConf, treeConf)
            .map(s => ((treeId, nodeId), (colId, s.gain)))
      }

      Utils.aggregateByKey[(T, N), (C, Float), mutable.ArrayBuilder[(C, Float)]](gainIter,
        () => mutable.ArrayBuilder.make[(C, Float)], _ += _)

        .map { case ((treeId, nodeId), gains) =>
          val colIds = gains.result().sortBy(_._2).takeRight(topK).map(_._1).sorted
          val size = inc.toInt(colIds.last) + 1
          val votes = KVVector.sparse[C, Int](size, colIds, Array.fill(colIds.length)(1))
          ((treeId, nodeId), votes.compress)
        }
    }.setName(s"Iter ${treeConf.iteration}, depth: $depth: Local Voted TopK")


    val globalVoted = localVoted
      .reduceByKey(_.plus(_).compress, boostConf.getRealHistogramParallelism)
      .mapValues { votes =>
        votes.activeIterator.toArray.sortBy(_._2)
          .takeRight(top2K).map(_._1).sorted
      }.setName(s"Iter ${treeConf.iteration}, depth: $depth: Global Voted Top2K")



    // RDD 'globalVoted' is usually much smaller than 'localHistograms'.
    // Instead of `join`, here we adopt `broadcast` or `zipPartitions`
    // to avoid join shuffle of 'localHistograms'.

    if (expectedSize < (1 << 16)) {
      val collected = globalVoted.collect.sortBy(_._1)

      val exactSize = collected.iterator.map(_._2.length).sum
      logInfo(s"Iter: ${treeConf.iteration}, depth: $depth, expectedSize: ${expectedSize.toInt}, " +
        s"exactSize: $exactSize, delta: $delta")
      delta *= exactSize.toDouble / expectedSize

      val treeIds = CompactArray.build[T](collected.iterator.map(_._1._1))
      val nodeIds = collected.map(_._1._2)
      val colIds = ArrayBlock.build[C](collected.iterator.map(_._2))

      val bcIds = sc.broadcast((treeIds, nodeIds, colIds))
      cleaner.registerBroadcastedObjects(bcIds)

      localHistograms.mapPartitions { localIter =>
        val (treeIds, nodeIds, colIds) = bcIds.value

        val flattenIter =
          Utils.zip3(treeIds.iterator, nodeIds.iterator, colIds.iterator)
            .flatMap { case (treeId, nodeId, colIds) =>
              colIds.map { colId => ((treeId, nodeId, colId), null) }
            }

        Utils.innerJoinSortedIters(localIter, flattenIter)
          .map { case (ids, hist, _) => (ids, hist) }
      }.reduceByKey(partitioner, _.plus(_).compress)

    } else {

      import RDDFunctions._

      val numParts = localHistograms.getNumPartitions

      val duplicatedGlobalVoted = globalVoted.allgather(numParts)
        .setName(s"Iter ${treeConf.iteration}, depth: $depth: Global Voted Top2K (AllGathered)")

      localHistograms.zipPartitions(duplicatedGlobalVoted) {
        case (localIter, globalIter) =>
          val flattenIter = globalIter.flatMap { case ((treeId, nodeId), colIds) =>
            colIds.iterator.map { colId => ((treeId, nodeId, colId), null) }
          }

          Utils.innerJoinSortedIters(localIter, flattenIter)
            .map { case (ids, hist, _) => (ids, hist) }
      }.reduceByKey(partitioner, _.plus(_).compress)
    }
  }

  override def clear(): Unit = {
    if (cleaner != null) {
      cleaner.clear(false)
    }
  }

  override def destroy(): Unit = {
    if (cleaner != null) {
      cleaner.clear(false)
      cleaner = null
    }
    delta = 1.0
  }
}


private[gbm] object HistogramUpdater extends Logging {


  /**
    * Compute the histogram of nodes
    *
    * @param nodeIdFilter function to filter nodeIds
    * @return histogram data containing (treeId, nodeId, columnId, histogram)
    */
  def computeHistograms[T, N, C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                                       treeIdBlocks: RDD[ArrayBlock[T]],
                                       nodeIdBlocks: RDD[ArrayBlock[N]],
                                       gradBlocks: RDD[ArrayBlock[H]],
                                       boostConf: BoostConfig,
                                       bcBoostConf: Broadcast[BoostConfig],
                                       treeConf: TreeConfig,
                                       bcTreeConf: Broadcast[TreeConfig],
                                       extraColSelector: Option[Selector],
                                       nodeIdFilter: N => Boolean,
                                       partitioner: Partitioner,
                                       depth: Int)
                                      (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                       cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                       cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                       cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                       ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {


    val numCols = treeConf.getNumCols.getOrElse(boostConf.getNumCols)


    if (boostConf.getForestSize * numCols < (1 << (16 - depth))) {

      computeLocalHistograms[T, N, C, B, H](binVecBlocks, treeIdBlocks, nodeIdBlocks, gradBlocks,
        bcBoostConf, bcTreeConf, extraColSelector, nodeIdFilter)
        .reduceByKey(partitioner, _.plus(_).compress)

    } else {

      computeHistogramsInTwoSteps(binVecBlocks, treeIdBlocks, nodeIdBlocks, gradBlocks,
        boostConf, bcTreeConf, extraColSelector, nodeIdFilter, partitioner)
    }
  }


  /**
    * Compute the histogram of nodes
    *
    * @param filterNodeId function to filter nodeIds
    * @return histogram data containing (treeId, nodeId, columnId, histogram)
    */
  def computeHistogramsVertical[T, N, C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                                               treeIdBlocks: RDD[ArrayBlock[T]],
                                               nodeIdBlocks: RDD[ArrayBlock[N]],
                                               gradBlocks: RDD[ArrayBlock[H]],
                                               boostConf: BoostConfig,
                                               bcBoostConf: Broadcast[BoostConfig],
                                               bcTreeConf: Broadcast[TreeConfig],
                                               extraColSelector: Option[Selector],
                                               nodeIdFilter: N => Boolean)
                                              (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                               cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                               cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                               cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                               ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {

    val localHistograms = computeLocalNonMissingOnlyHistograms[T, N, C, B, H](binVecBlocks, treeIdBlocks, nodeIdBlocks, gradBlocks,
      bcBoostConf, bcTreeConf, extraColSelector, nodeIdFilter)

    if (boostConf.getNumVLayers == 1) {
      localHistograms
    } else {
      localHistograms
        .reduceByKey(_.plus(_).compress, boostConf.getRealHistogramParallelism)
    }
  }


  /**
    * Compute the histogram of nodes
    *
    * @param nodeIdFilter function to filter nodeIds
    * @return histogram data containing (treeId, nodeId, columnId, histogram)
    */
  def computeHistogramsInTwoSteps[T, N, C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                                                 treeIdBlocks: RDD[ArrayBlock[T]],
                                                 nodeIdBlocks: RDD[ArrayBlock[N]],
                                                 gradBlocks: RDD[ArrayBlock[H]],
                                                 boostConf: BoostConfig,
                                                 bcTreeConf: Broadcast[TreeConfig],
                                                 extraColSelector: Option[Selector],
                                                 nodeIdFilter: N => Boolean,
                                                 partitioner: Partitioner)
                                                (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                 cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                 cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                 cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                 ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {

    val sum = treeIdBlocks.zipPartitions(nodeIdBlocks, gradBlocks) {
      case (treeIdBlockIter, nodeIdBlockIter, gradBlockIter) =>
        val sum = mutable.OpenHashMap.empty[(T, N), (H, H)]

        val iter = Utils.zip3(treeIdBlockIter, nodeIdBlockIter, gradBlockIter)
          .flatMap { case (treeIdBlock, nodeIdBlock, gradBlock) =>
            Utils.zip3(treeIdBlock.iterator, nodeIdBlock.iterator, gradBlock.iterator)
          }

        while (iter.hasNext) {
          val (treeIds, nodeIds, gradHess) = iter.next
          require(treeIds.length == nodeIds.length)

          val size = gradHess.length >> 1

          var i = 0
          while (i < treeIds.length) {
            val nodeId = nodeIds(i)

            if (nodeIdFilter(nodeId)) {
              val treeId = treeIds(i)
              val indexGrad = (i % size) << 1
              val grad = gradHess(indexGrad)
              val hess = gradHess(indexGrad + 1)
              val (gradSum, hessSum) = sum.getOrElse((treeId, nodeId), (nuh.zero, nuh.zero))
              sum.update((treeId, nodeId), (nuh.plus(gradSum, grad), nuh.plus(hessSum, hess)))
            }
            i += 1
          }
        }

        Iterator.single(sum.toArray)

    }.treeReduce(f = {
      case (sum0, sum1) =>
        (sum0 ++ sum1).groupBy(_._1).mapValues {
          array => (array.map(_._2._1).sum, array.map(_._2._2).sum)
        }.toArray
    }, depth = boostConf.getAggregationDepth)
      .toMap


    computeLocalNonMissingHistograms[T, N, C, B, H](binVecBlocks, treeIdBlocks, nodeIdBlocks, gradBlocks,
      bcTreeConf, extraColSelector, nodeIdFilter)

      .reduceByKey(partitioner, _.plus(_).compress)

      .mapPartitions(f = { iter =>

        iter.map { case ((treeId, nodeId, colId), hist) =>
          val (gradSum, hessSum) = sum.getOrElse((treeId, nodeId), (nuh.zero, nuh.zero))

          ((treeId, nodeId, colId),
            adjustHistVec(hist, gradSum, hessSum))
        }
      }, true)
  }


  /**
    * Compute local histogram of nodes in a sparse fashion
    *
    * @param nodeIdFilter function to filter nodeIds
    * @return histogram data containing (treeId, nodeId, columnId, histogram)
    */
  def computeLocalHistograms[T, N, C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                                            treeIdBlocks: RDD[ArrayBlock[T]],
                                            nodeIdBlocks: RDD[ArrayBlock[N]],
                                            gradBlocks: RDD[ArrayBlock[H]],
                                            bcBoostConf: Broadcast[BoostConfig],
                                            bcTreeConf: Broadcast[TreeConfig],
                                            extraColSelector: Option[Selector],
                                            nodeIdFilter: N => Boolean)
                                           (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                            cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                            cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                            cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                            ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {

    val flattened = binVecBlocks.zipPartitions(treeIdBlocks, nodeIdBlocks, gradBlocks) {
      case (binVecBlockIter, treeIdBlockIter, nodeIdBlockIter, gradBlockIter) =>
        val treeConf = bcTreeConf.value

        val selector = if (extraColSelector.nonEmpty) {
          Selector.union(treeConf.colSelector, extraColSelector.get)
        } else {
          treeConf.colSelector
        }

        val sum = mutable.OpenHashMap.empty[(T, N), (H, H)]

        Utils.zip4(binVecBlockIter, treeIdBlockIter, nodeIdBlockIter, gradBlockIter)
          .flatMap { case (binVecBlock, treeIdBlock, nodeIdBlock, gradBlock) =>
            Utils.zip4(binVecBlock.activeElementIterator, treeIdBlock.iterator,
              nodeIdBlock.iterator, gradBlock.iterator)

          }.flatMap { case (binIter, treeIds, nodeIds, gradHess) =>
          require(treeIds.length == nodeIds.length)

          val indices = Iterator.range(0, nodeIds.length)
            .filter(i => nodeIdFilter(nodeIds(i))).toArray

          if (indices.nonEmpty) {
            val size = gradHess.length >> 1

            // update gradSum & hessSum
            var j = 0
            while (j < indices.length) {
              val i = indices(j)
              val treeId = treeIds(i)
              val nodeId = nodeIds(i)
              val indexGrad = (i % size) << 1
              val grad = gradHess(indexGrad)
              val hess = gradHess(indexGrad + 1)
              val (gradSum, hessSum) = sum.getOrElse((treeId, nodeId), (nuh.zero, nuh.zero))
              sum.update((treeId, nodeId), (nuh.plus(gradSum, grad), nuh.plus(hessSum, hess)))
              j += 1
            }

            binIter.flatMap { case (colId, bin) =>
              indices.iterator.flatMap { i =>
                val treeId = treeIds(i)
                if (selector.containsById[T, C](treeId, colId)) {
                  val nodeId = nodeIds(i)
                  val indexGrad = (i % size) << 1
                  val grad = gradHess(indexGrad)
                  val hess = gradHess(indexGrad + 1)
                  Iterator.single((treeId, nodeId, colId), (bin, grad, hess))
                } else {
                  Iterator.empty
                }
              }
            }
          } else {
            Iterator.empty
          }

        } ++ sum.iterator.map { case ((treeId, nodeId), (gradSum, hessSum)) =>
          ((treeId, nodeId, inc.fromInt(-1)), (inb.zero, gradSum, hessSum))
        }
    }


    val localAgged = flattened
      .aggregateByKeyWithinPartitions(KVVector.empty[B, H], Some(Ordering.Tuple3[T, N, C]))(
        seqOp = {
          case (hist, (bin, grad, hess)) =>
            val indexGrad = inb.plus(bin, bin)
            val indexHess = inb.plus(indexGrad, inb.one)
            hist.plus(indexHess, hess)
              .plus(indexGrad, grad)

        }, combOp = _.plus(_))


    localAgged.mapPartitions { iter =>
      val boostConf = bcBoostConf.value
      val treeConf = bcTreeConf.value

      val numCols = treeConf.getNumCols.getOrElse(boostConf.getNumCols)

      val selector = if (extraColSelector.nonEmpty) {
        Selector.union(treeConf.colSelector, extraColSelector.get)
      } else {
        treeConf.colSelector
      }

      var prevTreeId = int.fromInt(-1)
      var prevNodeId = inn.fromInt(-1)
      var prevGradSum = nuh.zero
      var prevHessSum = nuh.zero
      var prevHist = KVVector.dense[B, H](Array(nuh.zero, nuh.zero))
      var validColIdIter = Array.empty[C].iterator

      Utils.validateKeyOrdering(iter)
        .flatMap { case ((treeId, nodeId, colId), hist) =>

          require((inc.equiv(colId, inc.fromInt(-1)) && hist.size == 2) ||
            (int.equiv(treeId, prevTreeId) && inn.equiv(nodeId, prevNodeId)))

          if (inc.equiv(colId, inc.fromInt(-1))) {

            validColIdIter.map { validColId =>
              ((prevTreeId, prevNodeId, validColId), prevHist)

            } ++ {
              prevTreeId = treeId
              prevNodeId = nodeId

              prevGradSum = hist(0)
              prevHessSum = hist(1)
              prevHist = hist

              validColIdIter = Iterator.range(0, numCols)
                .filter(colId => selector.containsById(treeId, colId))
                .map(inc.fromInt)

              Iterator.empty
            }

          } else {

            require(validColIdIter.hasNext)

            var validColId = validColIdIter.next()
            require(inc.lteq(validColId, colId))

            if (inc.equiv(validColId, colId)) {

              Iterator.single(((treeId, nodeId, colId),
                adjustHistVec(hist, prevGradSum, prevHessSum)))

            } else {

              val builder = mutable.ArrayBuilder.make[C]
              while (inc.lt(validColId, colId)) {
                builder += validColId
                validColId = validColIdIter.next()
              }

              require(inc.equiv(validColId, colId))

              builder.result().map { validColId =>
                ((prevTreeId, prevNodeId, validColId), prevHist)

              } ++ Iterator.single(((treeId, nodeId, colId),
                adjustHistVec(hist, prevGradSum, prevHessSum)))
            }
          }

        } ++ validColIdIter.map { validColId =>
        ((prevTreeId, prevNodeId, validColId), prevHist)
      }
    }
  }


  /**
    * Compute local histogram of nodes on non-zero bins in a sparse fashion.
    * That is, only take gradient on non-zero indices into account.
    *
    * @param nodeIdFilter function to filter nodeIds
    * @return histogram data containing (treeId, nodeId, columnId, histogram)
    */
  def computeLocalNonMissingHistograms[T, N, C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                                                      treeIdBlocks: RDD[ArrayBlock[T]],
                                                      nodeIdBlocks: RDD[ArrayBlock[N]],
                                                      gradBlocks: RDD[ArrayBlock[H]],
                                                      bcTreeConf: Broadcast[TreeConfig],
                                                      extraColSelector: Option[Selector],
                                                      nodeIdFilter: N => Boolean)
                                                     (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                      cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                      cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                      cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                      ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {

    val flattened = binVecBlocks.zipPartitions(treeIdBlocks, nodeIdBlocks, gradBlocks) {
      case (binVecBlockIter, treeIdBlockIter, nodeIdBlockIter, gradBlockIter) =>
        val treeConf = bcTreeConf.value

        val selector = if (extraColSelector.nonEmpty) {
          Selector.union(treeConf.colSelector, extraColSelector.get)
        } else {
          treeConf.colSelector
        }

        Utils.zip4(binVecBlockIter, treeIdBlockIter, nodeIdBlockIter, gradBlockIter)
          .flatMap { case (binVecBlock, treeIdBlock, nodeIdBlock, gradBlock) =>
            Utils.zip4(binVecBlock.activeElementIterator, treeIdBlock.iterator,
              nodeIdBlock.iterator, gradBlock.iterator)

          }.flatMap { case (binIter, treeIds, nodeIds, gradHess) =>
          require(treeIds.length == nodeIds.length)

          val indices = Iterator.range(0, nodeIds.length)
            .filter(i => nodeIdFilter(nodeIds(i))).toArray

          if (indices.nonEmpty) {
            val size = gradHess.length >> 1

            binIter.flatMap { case (colId, bin) =>
              indices.iterator.flatMap { i =>
                val treeId = treeIds(i)
                if (selector.containsById[T, C](treeId, colId)) {
                  val nodeId = nodeIds(i)
                  val indexGrad = (i % size) << 1
                  val grad = gradHess(indexGrad)
                  val hess = gradHess(indexGrad + 1)
                  Iterator.single((treeId, nodeId, colId), (bin, grad, hess))
                } else {
                  Iterator.empty
                }
              }
            }
          } else {
            Iterator.empty
          }
        }
    }

    flattened
      .aggregateByKeyWithinPartitions(KVVector.empty[B, H], Some(Ordering.Tuple3[T, N, C]))(
        seqOp = {
          case (hist, (bin, grad, hess)) =>
            val indexGrad = inb.plus(bin, bin)
            val indexHess = inb.plus(indexGrad, inb.one)
            hist.plus(indexHess, hess)
              .plus(indexGrad, grad)

        }, combOp = _.plus(_))
  }


  /**
    * Compute local histogram of nodes, only take histograms containing gradient
    * on non-zero indices into account. That is, if one histogram vector only
    * contains values on zero index, it will be ignored.
    * Note that this method DO NOT compute the exact local histograms.
    *
    * @param nodeIdFilter function to filter nodeIds
    * @return histogram data containing (treeId, nodeId, columnId, histogram)
    */
  def computeLocalNonMissingOnlyHistograms[T, N, C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                                                          treeIdBlocks: RDD[ArrayBlock[T]],
                                                          nodeIdBlocks: RDD[ArrayBlock[N]],
                                                          gradBlocks: RDD[ArrayBlock[H]],
                                                          bcBoostConf: Broadcast[BoostConfig],
                                                          bcTreeConf: Broadcast[TreeConfig],
                                                          extraColSelector: Option[Selector],
                                                          nodeIdFilter: N => Boolean)
                                                         (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                          cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                          cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                          cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                          ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {

    val flattened = binVecBlocks.zipPartitions(treeIdBlocks, nodeIdBlocks, gradBlocks) {
      case (binVecBlockIter, treeIdBlockIter, nodeIdBlockIter, gradBlockIter) =>
        val treeConf = bcTreeConf.value

        val selector = if (extraColSelector.nonEmpty) {
          Selector.union(treeConf.colSelector, extraColSelector.get)
        } else {
          treeConf.colSelector
        }

        val sum = mutable.OpenHashMap.empty[(T, N), (H, H)]

        Utils.zip4(binVecBlockIter, treeIdBlockIter, nodeIdBlockIter, gradBlockIter)
          .flatMap { case (binVecBlock, treeIdBlock, nodeIdBlock, gradBlock) =>
            Utils.zip4(binVecBlock.activeElementIterator, treeIdBlock.iterator,
              nodeIdBlock.iterator, gradBlock.iterator)

          }.flatMap { case (binIter, treeIds, nodeIds, gradHess) =>
          require(treeIds.length == nodeIds.length)

          val indices = Iterator.range(0, nodeIds.length)
            .filter(i => nodeIdFilter(nodeIds(i))).toArray

          if (indices.nonEmpty) {
            val size = gradHess.length >> 1

            // update gradSum & hessSum
            var j = 0
            while (j < indices.length) {
              val i = indices(j)
              val treeId = treeIds(i)
              val nodeId = nodeIds(i)
              val indexGrad = (i % size) << 1
              val grad = gradHess(indexGrad)
              val hess = gradHess(indexGrad + 1)
              val (gradSum, hessSum) = sum.getOrElse((treeId, nodeId), (nuh.zero, nuh.zero))
              sum.update((treeId, nodeId), (nuh.plus(gradSum, grad), nuh.plus(hessSum, hess)))
              j += 1
            }

            binIter.flatMap { case (colId, bin) =>
              indices.iterator.flatMap { i =>
                val treeId = treeIds(i)
                if (selector.containsById[T, C](treeId, colId)) {
                  val nodeId = nodeIds(i)
                  val indexGrad = (i % size) << 1
                  val grad = gradHess(indexGrad)
                  val hess = gradHess(indexGrad + 1)
                  Iterator.single((treeId, nodeId, colId), (bin, grad, hess))
                } else {
                  Iterator.empty
                }
              }
            }
          } else {
            Iterator.empty
          }

        } ++ sum.iterator.map { case ((treeId, nodeId), (gradSum, hessSum)) =>
          ((treeId, nodeId, inc.fromInt(-1)), (inb.zero, gradSum, hessSum))
        }

    }


    val localAgged = flattened
      .aggregateByKeyWithinPartitions(KVVector.empty[B, H], Some(Ordering.Tuple3[T, N, C]))(
        seqOp = {
          case (hist, (bin, grad, hess)) =>
            val indexGrad = inb.plus(bin, bin)
            val indexHess = inb.plus(indexGrad, inb.one)
            hist.plus(indexHess, hess)
              .plus(indexGrad, grad)

        }, combOp = _.plus(_))


    localAgged.mapPartitions { iter =>

      var prevTreeId = int.fromInt(-1)
      var prevNodeId = inn.fromInt(-1)
      var prevGradSum = nuh.zero
      var prevHessSum = nuh.zero

      Utils.validateKeyOrdering(iter)
        .flatMap { case ((treeId, nodeId, colId), hist) =>

          require((inc.equiv(colId, inc.fromInt(-1)) && hist.size == 2) ||
            (int.equiv(treeId, prevTreeId) && inn.equiv(nodeId, prevNodeId)))

          if (inc.equiv(colId, inc.fromInt(-1))) {

            prevTreeId = treeId
            prevNodeId = nodeId

            prevGradSum = hist(0)
            prevHessSum = hist(1)

            Iterator.empty

          } else {

            Iterator.single(((treeId, nodeId, colId),
              adjustHistVec(hist, prevGradSum, prevHessSum)))
          }
        }
    }
  }


  def adjustHistVec[B, H](hist: KVVector[B, H],
                          gradSum: H,
                          hessSum: H)
                         (implicit cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                          ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): KVVector[B, H] = {
    val two = inb.fromInt(2)

    var nzGradSum = nuh.zero
    var nzHessSum = nuh.zero

    val iter = hist.activeIterator
    while (iter.hasNext) {
      val (bin, v) = iter.next()
      if (inb.equiv(inb.rem(bin, two), inb.zero)) {
        nzGradSum = nuh.plus(nzGradSum, v)
      } else {
        nzHessSum = nuh.plus(nzHessSum, v)
      }
    }

    val g0 = nuh.minus(gradSum, nzGradSum)
    val h0 = nuh.minus(hessSum, nzHessSum)

    hist.plus(inb.zero, g0)
      .plus(inb.one, h0)
      .compress
  }


  /**
    * Histogram subtraction
    *
    * @param nodeHistograms  histogram data of parent nodes
    * @param rightHistograms histogram data of right leaves
    * @return histogram data of both left and right leaves
    */
  def subtractHistograms[T, N, C, B, H](nodeHistograms: RDD[((T, N, C), KVVector[B, H])],
                                        rightHistograms: RDD[((T, N, C), KVVector[B, H])],
                                        boostConf: BoostConfig,
                                        partitioner: Partitioner)
                                       (implicit ct: ClassTag[T], int: Integral[T],
                                        cn: ClassTag[N], inn: Integral[N],
                                        cc: ClassTag[C], inc: Integral[C],
                                        cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                        ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {

    val threshold = neh.fromDouble(boostConf.getMinNodeHess * 2)

    // Only if the partitioner is a SkipNodePratitioner or DepthPratitioner, we can preserves
    // the partitioning after changing the nodeId in key
    val preserve1 = nodeHistograms.partitioner match {
      case Some(_: SkipNodePratitioner[_, _, _]) => true
      case Some(_: DepthPratitioner[_, _, _]) => true
      case _ => false
    }

    val preserve2 = partitioner match {
      case _: SkipNodePratitioner[_, _, _] => true
      case _: DepthPratitioner[_, _, _] => true
      case _ => false
    }

    nodeHistograms.mapPartitions(f = { iter =>
      iter.map { case ((treeId, parentNodeId, colId), parentHist) =>
        val leftNodeId = inn.plus(parentNodeId, parentNodeId)
        val rightNodeId = inn.plus(leftNodeId, inn.one)
        ((treeId, rightNodeId, colId), parentHist)
      }
    }, preserve1)

      .join(rightHistograms, partitioner)

      .mapPartitions(f = { iter =>

        iter.flatMap { case ((treeId, rightNodeId, colId), (parentHist, rightHist)) =>
          require(rightHist.size <= parentHist.size)
          val leftNodeId = inn.minus(rightNodeId, inn.one)
          val leftHist = parentHist.minus(rightHist)

          ((treeId, leftNodeId, colId), leftHist.compress) ::
            ((treeId, rightNodeId, colId), rightHist.compress) :: Nil

        }.filter { case (_, hist) =>
          // leaves with hess less than minNodeHess * 2 can not grow furthermore
          val hessSum = hist.activeIterator.filter { case (b, _) =>
            inb.equiv(inb.rem(b, inb.fromInt(2)), inb.one)
          }.map(_._2).sum

          nuh.gteq(hessSum, threshold) && hist.nnz > 2
        }

      }, preserve2)
  }


  /**
    * In histogram subtraction, update partitioner for the current depth to avoid shuffle if possible
    *
    * @param treeIds         current treeIds
    * @param depth           current depth
    * @param prevPartitioner previous partitioner
    */
  def updatePartitioner[T, N, C](boostConf: BoostConfig,
                                 treeConf: TreeConfig,
                                 treeIds: Array[T],
                                 depth: Int,
                                 parallelism: Int,
                                 prevPartitioner: Option[Partitioner])
                                (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                 cn: ClassTag[N], inn: Integral[N],
                                 cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C]): Partitioner = {
    prevPartitioner match {
      case Some(p: SkipNodePratitioner[_, _, _])
        if p.numPartitions == parallelism && p.treeIds.length == treeIds.length => p

      case Some(p: DepthPratitioner[_, _, _])
        if p.numPartitions == parallelism && p.treeIds.length == treeIds.length => p

      case _ =>

        val numCols = treeConf.getNumCols.getOrElse(boostConf.getNumCols)

        // ignore nodeId here
        val expectedNumKeys = treeIds.length * numCols *
          boostConf.getColSampleRateByTree * boostConf.getColSampleRateByNode

        if (expectedNumKeys >= (parallelism << 3)) {
          new SkipNodePratitioner[T, N, C](parallelism, numCols, treeIds)

        } else if (depth > 2 && expectedNumKeys * (1 << (depth - 1)) >= (parallelism << 3)) {
          // check the parent level (not current level)
          new DepthPratitioner[T, N, C](parallelism, numCols, depth - 1, treeIds)

        } else {
          new HashPartitioner(parallelism)
        }
    }
  }
}


/**
  * Partitioner that ignore nodeId in key (treeId, nodeId, colId), this will avoid unnecessary shuffle
  * in histogram subtraction and reduce communication cost in following split-searching.
  */

private[gbm] class SkipNodePratitioner[T, N, C](val numPartitions: Int,
                                                val numCols: Int,
                                                val treeIds: Array[T])
                                               (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                cn: ClassTag[N], inn: Integral[N],
                                                cc: ClassTag[C], inc: Integral[C]) extends Partitioner {
  require(numPartitions > 0)
  require(numCols > 0)
  require(treeIds.nonEmpty)
  require(Utils.validateOrdering[T](treeIds.iterator).forall(t => int.gteq(t, int.zero)))

  private val hash = numPartitions * (numCols + int.toInt(treeIds.sum) + int.toInt(treeIds.min) + int.toInt(treeIds.max))

  private val treeInterval = numPartitions.toDouble / treeIds.length

  private val colInterval = treeInterval / numCols

  override def getPartition(key: Any): Int = key match {

    case (treeId: T, _: N, colId: C) =>
      val i = net.search(treeIds, treeId)
      require(i >= 0, s"Can not index key $treeId in ${treeIds.mkString("[", ",", "]")}")

      val p = i * treeInterval + inc.toInt(colId) * colInterval
      math.min(numPartitions - 1, p.toInt)
  }

  override def equals(other: Any): Boolean = other match {
    case p: SkipNodePratitioner[T, N, C] =>
      numPartitions == p.numPartitions && numCols == p.numCols &&
        treeIds.length == p.treeIds.length &&
        Iterator.range(0, treeIds.length).forall(i => treeIds(i) == p.treeIds(i))

    case _ =>
      false
  }

  override def hashCode: Int = hash

  override def toString: String = {
    s"SkipNodePratitioner[${ct.runtimeClass.toString.capitalize}, ${cn.runtimeClass.toString.capitalize}, " +
      s"${cc.runtimeClass.toString.capitalize}](numPartitions=$numPartitions, numCols=$numCols, " +
      s"treeIds=${treeIds.mkString("[", ",", "]")})"
  }
}


/**
  * Partitioner that will map nodeId into certain depth before partitioning:
  * if nodeId is of depth #depth, just keep it;
  * if nodeId is a descendant of depth level, map it to its ancestor in depth #depth;
  * otherwise, throw an exception
  * this will avoid unnecessary shuffle in histogram subtraction and reduce communication cost in following split-searching.
  */
private[gbm] class DepthPratitioner[T, N, C](val numPartitions: Int,
                                             val numCols: Int,
                                             val depth: Int,
                                             val treeIds: Array[T])
                                            (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                             cn: ClassTag[N], inn: Integral[N],
                                             cc: ClassTag[C], inc: Integral[C]) extends Partitioner {
  require(numPartitions > 0)
  require(numCols > 0)
  require(depth > 1)
  require(treeIds.nonEmpty)
  require(Utils.validateOrdering[T](treeIds.iterator).forall(t => int.gteq(t, int.zero)))

  private val lowerBound: Int = 1 << depth

  private val upperBound: Int = lowerBound << 1

  private val hash = numPartitions * depth *
    (numCols + int.toInt(treeIds.sum) + int.toInt(treeIds.min) + int.toInt(treeIds.max))

  private val treeInterval = numPartitions.toDouble / treeIds.length

  private val nodeInterval = treeInterval / lowerBound

  private val colInterval = nodeInterval / numCols

  override def getPartition(key: Any): Int = key match {

    case (treeId: T, nodeId: N, colId: C) =>
      val i = net.search(treeIds, treeId)
      require(i >= 0, s"Can not index key $treeId in ${treeIds.mkString("[", ",", "]")}")

      val nodeId2 = adjust(inn.toInt(nodeId))

      val p = i * treeInterval + (nodeId2 - lowerBound) * nodeInterval + inc.toDouble(colId) * colInterval
      math.min(numPartitions - 1, p.toInt)
  }

  private def adjust(nodeId: Int): Int = {
    require(nodeId >= lowerBound, s"nodeId $nodeId < lowerBound $lowerBound")
    var n = nodeId
    while (n >= upperBound) {
      n >>= 1
    }
    n
  }

  override def equals(other: Any): Boolean = other match {
    case p: DepthPratitioner[T, N, C] =>
      numPartitions == p.numPartitions &&
        numCols == p.numCols && depth == p.depth &&
        treeIds.length == p.treeIds.length &&
        Iterator.range(0, treeIds.length).forall(i => treeIds(i) == p.treeIds(i))

    case _ =>
      false
  }

  override def hashCode: Int = hash

  override def toString: String = {
    s"DepthPratitioner[${ct.runtimeClass.toString.capitalize}, ${cn.runtimeClass.toString.capitalize}, " +
      s"${cc.runtimeClass.toString.capitalize}](numPartitions=$numPartitions, numCols=$numCols, " +
      s"depth=$depth, treeIds=${treeIds.mkString("[", ",", "]")})"
  }
}


/**
  * Partitioner that partition the keys (treeId, nodeId, colId) by order, this will
  * reduce communication cost in following split-searching.
  */
private[gbm] class RangePratitioner[T, N, C](val numPartitions: Int,
                                             val numCols: Int,
                                             val treeNodeIds: Array[(T, N)])
                                            (implicit ct: ClassTag[T], int: Integral[T],
                                             cn: ClassTag[N], inn: Integral[N],
                                             cc: ClassTag[C], inc: Integral[C],
                                             order: Ordering[(T, N)]) extends Partitioner {
  require(numPartitions > 0)
  require(numCols > 0)
  require(treeNodeIds.nonEmpty)
  require(Utils.validateOrdering[(T, N)](treeNodeIds.iterator)
    .forall { case (treeId, nodeId) => int.gteq(treeId, int.zero) && inn.gteq(nodeId, inn.zero) })

  private val hash = {
    val treeIds = treeNodeIds.map(_._1)
    val nodeIds = treeNodeIds.map(_._2)
    numPartitions * (numCols + int.toInt(treeIds.sum) + int.toInt(treeIds.min) + int.toInt(treeIds.max)
      + inn.toInt(nodeIds.sum) + inn.toInt(nodeIds.max) + inn.toInt(nodeIds.min))
  }

  private val nodeInterval = numPartitions.toDouble / treeNodeIds.length

  private val colInterval = nodeInterval / numCols

  override def getPartition(key: Any): Int = key match {

    case (treeId: T, nodeId: N, colId: C) =>
      val i = ju.Arrays.binarySearch(treeNodeIds, (treeId, nodeId),
        order.asInstanceOf[ju.Comparator[(T, N)]])
      require(i >= 0, s"Can not index key ${(treeId, nodeId)} in ${treeNodeIds.mkString("[", ",", "]")}")

      val p = i * nodeInterval + inc.toDouble(colId) * colInterval
      math.min(numPartitions - 1, p.toInt)
  }

  override def equals(other: Any): Boolean = other match {
    case p: RangePratitioner[T, N, C] =>
      numPartitions == p.numPartitions &&
        numCols == p.numCols && treeNodeIds.length == p.treeNodeIds.length &&
        Iterator.range(0, treeNodeIds.length).forall(i => order.equiv(treeNodeIds(i), p.treeNodeIds(i)))

    case _ =>
      false
  }

  override def hashCode: Int = hash

  override def toString: String = {
    s"RangePratitioner[${ct.runtimeClass.toString.capitalize}, ${cn.runtimeClass.toString.capitalize}, " +
      s"${cc.runtimeClass.toString.capitalize}](numPartitions=$numPartitions, numCols=$numCols, " +
      s"treeNodeIds=${treeNodeIds.mkString("[", ",", "]")})"
  }
}


