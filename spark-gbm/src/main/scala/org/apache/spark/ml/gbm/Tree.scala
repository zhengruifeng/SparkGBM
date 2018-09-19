package org.apache.spark.ml.gbm

import java.{util => ju}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random
import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.util.BoundedPriorityQueue


private[gbm] object Tree extends Serializable with Logging {


  /**
    *
    * @param data      instances containing (bins, treeIds, grad-hess), grad&hess is recurrent for compression. i.e
    *                  treeIds = [t1,t2,t5,t6], grad-hess = [g1,h1,g2,h2] -> {t1:(g1,h1), t2:(g2,h2), t5:(g1,h1), t6:(g2,h2)}
    * @param boostConf boosting configure
    * @param baseConf  trees-growth configure
    * @return tree models
    */
  def trainWithDataParallelism[T, N, C, B, H](data: RDD[(KVVector[C, B], Array[T], Array[H])],
                                              boostConf: BoostConfig,
                                              baseConf: BaseConfig)
                                             (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                              cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
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

    logInfo(s"Iteration ${baseConf.iteration}: trees growth start")

    val splitRNG = new Random(boostConf.getSeed)

    val roots = Array.fill(baseConf.numTrees)(LearningNode.create(1))
    val numLeaves = Array.fill(baseConf.numTrees)(1)
    val finished = Array.fill(baseConf.numTrees)(false)

    var minNodeId = inn.one
    var depth = 0

    val parallelism = boostConf.getRealParallelism(boostConf.getReduceParallelism, sc.defaultParallelism)

    var prevTreeIds = Array.tabulate(baseConf.numTrees)(int.fromInt)
    var prevPartitioner = Option.empty[Partitioner]
    val prevSplits = mutable.OpenHashMap.empty[(T, N), Split]

    while (finished.contains(false) && depth <= boostConf.getMaxDepth) {
      val start = System.nanoTime

      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth: splitting start")

      if (inn.equiv(minNodeId, inn.one)) {
        nodeIds = data.map { case (_, treeIds, _) => Array.fill(treeIds.length)(inn.one) }
        nodeIds.setName(s"NodeIds (Iteration ${baseConf.iteration}, depth $depth)")
      } else {
        nodeIds = updateNodeIds[T, N, C, B, H](data, nodeIds, prevSplits.toMap)
        nodeIds.setName(s"NodeIds (Iteration ${baseConf.iteration}, depth $depth)")
        nodesCheckpointer.update(nodeIds)
      }


      val partitioner = updatePartitioner[T, N, C](boostConf, prevTreeIds, depth, parallelism, prevPartitioner)
      prevPartitioner = Some(partitioner)
      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth, minNodeId $minNodeId, partitioner $partitioner")


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

      // find best splits
      val splits = findSplits[T, N, C, B, H](hists, boostConf, baseConf, depth, splitRNG.nextLong)

      // update trees
      updateTrees[T, N](splits, boostConf, baseConf, roots, numLeaves, finished, depth, minNodeId, prevSplits)
      logInfo(s"Iteration ${baseConf.iteration}: $depth: growth finished, duration ${(System.nanoTime - start) / 1e9} seconds")

      prevTreeIds = prevSplits.keysIterator.map(_._1).toArray.distinct.sorted

      minNodeId = inn.plus(minNodeId, minNodeId)
      depth += 1
    }

    if (depth >= boostConf.getMaxDepth) {
      logInfo(s"Iteration ${baseConf.iteration}: maxDepth=${boostConf.getMaxDepth} reached, trees growth finished")
    } else {
      logInfo(s"Iteration ${baseConf.iteration}: trees growth finished")
    }

    nodesCheckpointer.cleanup()
    histsCheckpointer.cleanup()

    roots.map(TreeModel.createModel)
  }


  /**
    *
    * @param data      instances containing (bins, treeIds, grad-hess), grad&hess is recurrent for compression. i.e
    *                  treeIds = [t1,t2,t5,t6], grad-hess = [g1,h1,g2,h2] -> {t1:(g1,h1), t2:(g2,h2), t5:(g1,h1), t6:(g2,h2)}
    * @param boostConf boosting configure
    * @param baseConf  trees-growth configure
    * @return tree models
    */
  def trainWithVotingParallelism[T, N, C, B, H](data: RDD[(KVVector[C, B], Array[T], Array[H])],
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

    logInfo(s"Iteration ${baseConf.iteration}: trees growth start")

    val splitRNG = new Random(boostConf.getSeed)

    val roots = Array.fill(baseConf.numTrees)(LearningNode.create(1))
    val numLeaves = Array.fill(baseConf.numTrees)(1)
    val finished = Array.fill(baseConf.numTrees)(false)

    var minNodeId = inn.one
    var depth = 0

    val parallelism = boostConf.getRealParallelism(boostConf.getReduceParallelism, sc.defaultParallelism)

    var prevTreeNodeIds = Array.tabulate(baseConf.numTrees)(t => (int.fromInt(t), inn.one))
    val prevSplits = mutable.OpenHashMap.empty[(T, N), Split]

    val recoder = new ResourceRecoder

    while (finished.contains(false) && depth <= boostConf.getMaxDepth) {
      val start = System.nanoTime

      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth: splitting start")

      if (inn.equiv(minNodeId, inn.one)) {
        nodeIds = data.map { case (_, treeIds, _) => Array.fill(treeIds.length)(inn.one) }
        nodeIds.setName(s"NodeIds (Iteration ${baseConf.iteration}, depth $depth)")
      } else {
        nodeIds = updateNodeIds[T, N, C, B, H](data, nodeIds, prevSplits.toMap)
        nodeIds.setName(s"NodeIds (Iteration ${baseConf.iteration}, depth $depth)")
        nodesCheckpointer.update(nodeIds)
      }

      val partitioner = new IDRangePratitioner[T, N, C](parallelism, boostConf.getNumCols, depth, prevTreeNodeIds)

      // compute histograms
      val hists = computeHistogramsWithVoting[T, N, C, B, H](data.zip(nodeIds), boostConf, baseConf, depth, partitioner, recoder)

      // find best splits
      val splits = findSplits[T, N, C, B, H](hists, boostConf, baseConf, depth, splitRNG.nextLong)

      recoder.cleanup()

      // update trees
      updateTrees[T, N](splits, boostConf, baseConf, roots, numLeaves, finished, depth, minNodeId, prevSplits)
      logInfo(s"Iteration ${baseConf.iteration}: $depth: growth finished, duration ${(System.nanoTime - start) / 1e9} seconds")

      prevTreeNodeIds = prevSplits.keys.toArray.sorted

      minNodeId = inn.plus(minNodeId, minNodeId)
      depth += 1
    }

    if (depth >= boostConf.getMaxDepth) {
      logInfo(s"Iteration ${baseConf.iteration}: maxDepth=${boostConf.getMaxDepth} reached, trees growth finished")
    } else {
      logInfo(s"Iteration ${baseConf.iteration}: trees growth finished")
    }

    nodesCheckpointer.cleanup()

    roots.map(TreeModel.createModel)
  }


  /**
    * create partitioner for the current depth to avoid shuffle if possible
    *
    * @param treeIds         current treeIds
    * @param depth           current depth
    * @param prevPartitioner previous partitioner
    */
  def updatePartitioner[T, N, C](boostConf: BoostConfig,
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

        // ignore nodeId here
        val numKeys = treeIds.length * boostConf.getNumCols *
          boostConf.getColSampleByTree * boostConf.getColSampleByLevel

        if (numKeys >= (parallelism << 3)) {
          new SkipNodePratitioner[T, N, C](parallelism, boostConf.getNumCols, treeIds)

        } else if (depth > 2 && numKeys * (1 << (depth - 1)) >= (parallelism << 3)) {
          // check the parent level (not current level)
          new DepthPratitioner[T, N, C](parallelism, boostConf.getNumCols, depth - 1, treeIds)

        } else {
          new HashPartitioner(parallelism)
        }
    }
  }


  /**
    * update trees and other metrics
    *
    * @param splits     best splits
    * @param boostConf  boosting config
    * @param baseConf   base model config
    * @param roots      root nodes of trees, will be updated
    * @param numLeaves  number of leaves of trees, will be updated
    * @param finished   indicates that where a tree has stopped growth, will be updated
    * @param depth      current depth
    * @param minNodeId  current minimum index of node of this depth
    * @param prevSplits previous selected splits, will be cleaned and set to selected splits of this depth
    */
  def updateTrees[T, N](splits: Map[(T, N), Split],
                        boostConf: BoostConfig,
                        baseConf: BaseConfig,
                        roots: Array[LearningNode],
                        numLeaves: Array[Int],
                        finished: Array[Boolean],
                        depth: Int,
                        minNodeId: N,
                        prevSplits: mutable.OpenHashMap[(T, N), Split])
                       (implicit ct: ClassTag[T], int: Integral[T],
                        cn: ClassTag[N], inn: Integral[N]): Unit = {
    prevSplits.clear()

    if (splits.nonEmpty) {
      val splits2 = splits.toArray
        .map { case ((treeId, nodeId), split) => (treeId, nodeId, split) }
        .groupBy(_._1)
        .mapValues(_.map(t => (t._2, t._3)))

      var numFinished = 0

      Iterator.range(0, baseConf.numTrees).filterNot(finished).foreach { treeId =>
        val maxRemains = boostConf.getMaxLeaves - numLeaves(treeId)

        splits2.get(int.fromInt(treeId)) match {

          case Some(array) if array.length == maxRemains =>
            array.foreach { case (nodeId, split) => prevSplits.update((int.fromInt(treeId), nodeId), split) }
            numLeaves(treeId) += array.length
            finished(treeId) = true
            numFinished += 1

          case Some(array) if array.length > maxRemains =>
            // if number of splits in a tree exceeds `maxLeaves`, only a part will be selected
            array.sortBy(_._2.gain).takeRight(maxRemains)
              .foreach { case (nodeId, split) => prevSplits.update((int.fromInt(treeId), nodeId), split) }
            numLeaves(treeId) += maxRemains
            finished(treeId) = true
            numFinished += 1

          case Some(array) if array.length < maxRemains =>
            array.foreach { case (nodeId, split) => prevSplits.update((int.fromInt(treeId), nodeId), split) }
            numLeaves(treeId) += array.length

          case None =>
            finished(treeId) = true
            numFinished += 1
        }
      }

      updateRoots[T, N](roots, minNodeId, prevSplits.toMap)

      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth: splitting finished, $numFinished trees growth finished," +
        s" ${prevSplits.size} leaves split, total gain=${prevSplits.values.map(_.gain).sum}.")

    } else {
      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth: no more splits found, trees growth finished.")
      Iterator.range(0, finished.length).foreach(finished(_) = true)
    }
  }


  /**
    * update roots of trees
    *
    * @param roots     roots of trees
    * @param minNodeId minimum nodeId for this level
    * @param splits    splits of leaves
    */
  def updateRoots[T, N](roots: Array[LearningNode],
                        minNodeId: N,
                        splits: Map[(T, N), Split])
                       (implicit ct: ClassTag[T], int: Integral[T],
                        cn: ClassTag[N], inn: Integral[N]): Unit = {
    if (splits.nonEmpty) {
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
      }, combOp = _ plus _

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

    // Only if the partitioner is a SkipNodePratitioner or DepthPratitioner, we can preserves
    // the partitioning after changing the nodeId in key
    val preserve1 = nodeHists.partitioner match {
      case Some(_: SkipNodePratitioner[_, _, _]) => true
      case Some(_: DepthPratitioner[_, _, _]) => true
      case _ => false
    }

    val preserve2 = partitioner match {
      case _: SkipNodePratitioner[_, _, _] => true
      case _: DepthPratitioner[_, _, _] => true
      case _ => false
    }

    nodeHists.mapPartitions(f = { iter =>
      iter.map { case ((treeId, parentNodeId, colId), parentHist) =>
        val leftNodeId = inn.plus(parentNodeId, parentNodeId)
        val rightNodeId = inn.plus(leftNodeId, inn.one)
        ((treeId, rightNodeId, colId), parentHist)
      }
    }, preserve1)

      .join(rightHists, partitioner)

      .mapPartitions(f = { iter =>

        iter.flatMap { case ((treeId, rightNodeId, colId), (parentHist, rightHist)) =>
          require(rightHist.len <= parentHist.len)
          val leftNodeId = inn.minus(rightNodeId, inn.one)
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

      }, preserve2)
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

        if (metrics.head > 0) {
          Iterator.single((splits.toArray, metrics))
        } else {
          Iterator.empty
        }

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


  /**
    * Compute the histogram of root node or the right leaves with nodeId greater than minNodeId
    *
    * @param data instances appended with nodeId, containing ((bins, treeIds, grad-hess), nodeIds)
    * @return histogram data containing (treeId, nodeId, columnId, histogram)
    */
  def computeHistogramsWithVoting[T, N, C, B, H](data: RDD[((KVVector[C, B], Array[T], Array[H]), Array[N])],
                                                 boostConf: BoostConfig,
                                                 baseConf: BaseConfig,
                                                 depth: Int,
                                                 partitioner: Partitioner,
                                                 recoder: ResourceRecoder)
                                                (implicit ct: ClassTag[T], int: Integral[T],
                                                 cn: ClassTag[N], inn: Integral[N],
                                                 cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                 cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                 ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[((T, N, C), KVVector[B, H])] = {
    import PairRDDFunctions._

    val sc = data.sparkContext

    // config of level-based column sampling
    val baseConf2 = if (boostConf.getColSampleByLevel == 1) {
      new BaseConfig(-1, -1, Array.empty)

    } else if (boostConf.getNumCols * boostConf.getColSampleByLevel > 32) {
      val rng = new Random(boostConf.getSeed * baseConf.iteration + depth)
      val numBaseModels = baseConf.numTrees / boostConf.getRawSize
      val maximum = (Int.MaxValue * boostConf.getColSampleByTree).ceil.toInt
      val selectors: Array[ColumSelector] = Array.range(0, numBaseModels).flatMap { i =>
        val seed = rng.nextInt
        Iterator.fill(boostConf.getRawSize)(HashSelector(maximum, seed))
      }
      new BaseConfig(-1, -1, selectors)

    } else {
      val rng = new Random(boostConf.getSeed * baseConf.iteration + depth)
      val numBaseModels = baseConf.numTrees / boostConf.getRawSize
      val numSelected = (boostConf.getNumCols * boostConf.getColSampleByTree).ceil.toInt
      val selectors: Array[ColumSelector] = Array.range(0, numBaseModels).flatMap { i =>
        val selected = rng.shuffle(Seq.range(0, boostConf.getNumCols))
          .take(numSelected).toArray.sorted
        Iterator.fill(boostConf.getRawSize)(SetSelector(selected))
      }
      new BaseConfig(-1, -1, selectors)
    }
    require(baseConf.colSelectors.length == baseConf2.colSelectors.length)


    val localHistograms = data.mapPartitions { iter =>
      val histSums = mutable.OpenHashMap.empty[(T, N), (H, H)]

      iter.flatMap { case ((bins, treeIds, gradHess), nodeIds) =>
        val gradSize = gradHess.length >> 1
        Iterator.range(0, treeIds.length)
          .flatMap { i =>
            val treeId = treeIds(i)
            val nodeId = nodeIds(i)
            val indexGrad = (i % gradSize) << 1
            val grad = gradHess(indexGrad)
            val hess = gradHess(indexGrad + 1)

            val (g, h) = histSums.getOrElse((treeId, nodeId), (nuh.zero, nuh.zero))
            histSums.update((treeId, nodeId), (nuh.plus(g, grad), nuh.plus(h, hess)))

            val selector = baseConf.getSelector(int.toInt(treeId))
            val selector2 = baseConf2.getSelector(int.toInt(treeId))

            // ignore zero-index bins
            bins.activeIter
              .filter { case (colId, _) => selector.contains(colId) && selector2.contains(colId) }
              .map { case (colId, bin) => ((treeId, nodeId, colId), (bin, grad, hess)) }
          }

      } ++ histSums.iterator.flatMap { case ((treeId, nodeId), (gradSum, hessSum)) =>
        val selector = baseConf.getSelector(int.toInt(treeId))
        val selector2 = baseConf2.getSelector(int.toInt(treeId))

        // make sure all available (treeId, nodeId, colId) tuples are taken into account
        // by the way, store sum of hist in zero-index bin
        Iterator.range(0, boostConf.getNumCols)
          .filter(colId => selector.contains(colId) && selector2.contains(colId))
          .map { colId => ((treeId, nodeId, inc.fromInt(colId)), (inb.zero, gradSum, hessSum)) }
      }

    }.aggregatePartitionsByKey(KVVector.empty[B, H])(
      seqOp = {
        case (hist, (bin, grad, hess)) =>
          val indexGrad = inb.plus(bin, bin)
          val indexHess = inb.plus(indexGrad, inb.one)
          hist.plus(indexHess, hess)
            .plus(indexGrad, grad)
      }, combOp = _ plus _

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
    }.setName("Local Histograms")

    localHistograms.persist(boostConf.getStorageLevel)
    recoder.append(localHistograms)

    val parallelism = partitioner.numPartitions
    val topK = boostConf.getTopK
    val top2K = topK << 1

    val voted = localHistograms.flatMap { case ((treeId, nodeId, colId), hist) =>
      val split = Split.split[H](inc.toInt(colId), hist.toArray, boostConf, baseConf)
      split.map(s => ((treeId, nodeId), (-s.gain, colId)))

    }.aggregatePartitionsByKey(new BoundedPriorityQueue[(Float, C)](topK))(
      seqOp = _ += _,
      combOp = _ ++= _
    ).setName("Local TopK")

      .aggregateByKey(KVVector.empty[C, Int], parallelism)(
        seqOp = {
          case (votes, localTopK) =>
            var v = votes
            localTopK.iterator.foreach { case (_, colId) => v = v.plus(colId, 1) }
            v
        }, combOp = _ plus _

      ).setName("Global Votes")

      .flatMap { case ((treeId, nodeId), votes) =>
        votes.activeIter.toArray
          .sortBy(_._2).takeRight(top2K)
          .iterator.map { case (colId, _) =>
          ((treeId, nodeId, colId), true)
        }
      }.setName("Global Top2K")

    voted.persist(boostConf.getStorageLevel)
    recoder.append(voted)

    if (voted.count < (1 << 16)) {
      val set = voted.map(_._1).collect().toSet
      val bcSet = sc.broadcast(set)
      recoder.append(bcSet)

      localHistograms.mapPartitions { iter =>
        val set = bcSet.value
        iter.filter { case (id, _) =>
          set.contains(id)
        }
      }.reduceByKey(partitioner, _ plus _)

    } else {
      localHistograms.join(voted, partitioner)
        .mapValues(_._1)
        .reduceByKey(partitioner, _ plus _)
    }
  }
}


/**
  * Partitioner that ignore nodeId in key (treeId, nodeId, colId), this will avoid unnecessary shuffle
  * in histogram subtraction and reduce communication cost in following split-searching.
  */
class SkipNodePratitioner[T, N, C](val numPartitions: Int,
                                   val numCols: Int,
                                   val treeIds: Array[T])
                                  (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                   cn: ClassTag[N], inn: Integral[N],
                                   cc: ClassTag[C], inc: Integral[C]) extends Partitioner {
  require(numPartitions > 0)
  require(numCols > 0)
  require(treeIds.nonEmpty)
  require(treeIds.forall(t => int.gteq(t, int.zero)))
  require(Iterator.range(0, treeIds.length - 1).forall(i => int.lt(treeIds(i), treeIds(i + 1))))

  private val hash = numPartitions * (numCols + int.toInt(treeIds.sum) + int.toInt(treeIds.min) + int.toInt(treeIds.max))

  private val treeInterval = numPartitions.toDouble / treeIds.length

  private val colInterval = treeInterval / numCols

  override def getPartition(key: Any): Int = key match {
    case null => 0

    case (treeId: T, _: N, colId: C) =>
      val i = net.search(treeIds, treeId)
      require(i >= 0)

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
    s"SkipNodePratitioner[${ct.runtimeClass.toString.capitalize}, ${cn.runtimeClass.toString.capitalize}, ${cc.runtimeClass.toString.capitalize}]" +
      s"(numPartitions=$numPartitions, numCols=$numCols, treeIds=${treeIds.mkString("[", ",", "]")})"
  }
}


/**
  * Partitioner that will map nodeId into certain depth before partitioning:
  * if nodeId is of depth #depth, just keep it;
  * if nodeId is a descendant of depth level, map it to its ancestor in depth #depth;
  * otherwise, throw an exception
  * this will avoid unnecessary shuffle in histogram subtraction and reduce communication cost in following split-searching.
  */
class DepthPratitioner[T, N, C](val numPartitions: Int,
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
  require(treeIds.forall(t => int.gteq(t, int.zero)))
  require(Iterator.range(0, treeIds.length - 1).forall(i => int.lt(treeIds(i), treeIds(i + 1))))

  private val lowerBound: Int = 1 << depth

  private val upperBound: Int = lowerBound << 1

  private val hash = numPartitions * depth * (numCols + int.toInt(treeIds.sum) + int.toInt(treeIds.min) + int.toInt(treeIds.max))

  private val treeInterval = numPartitions.toDouble / treeIds.length

  private val nodeInterval = treeInterval / lowerBound

  private val colInterval = nodeInterval / numCols

  override def getPartition(key: Any): Int = key match {
    case null => 0

    case (treeId: T, nodeId: N, colId: C) =>
      val i = net.search(treeIds, treeId)
      require(i >= 0)

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
    s"DepthPratitioner[${ct.runtimeClass.toString.capitalize}, ${cn.runtimeClass.toString.capitalize}, ${cc.runtimeClass.toString.capitalize}]" +
      s"(numPartitions=$numPartitions, numCols=$numCols, depth=$depth, treeIds=${treeIds.mkString("[", ",", "]")})"
  }
}


/**
  * Partitioner that partition the keys (treeId, nodeId, colId) by order, this will
  * reduce communication cost in following split-searching.
  */
class IDRangePratitioner[T, N, C](val numPartitions: Int,
                                  val numCols: Int,
                                  val depth: Int,
                                  val treeNodeIds: Array[(T, N)])
                                 (implicit ct: ClassTag[T], int: Integral[T],
                                  cn: ClassTag[N], inn: Integral[N],
                                  cc: ClassTag[C], inc: Integral[C],
                                  order: Ordering[(T, N)]) extends Partitioner {
  require(numPartitions > 0)
  require(numCols > 0)
  require(depth > 1)
  require(treeNodeIds.nonEmpty)
  require(Iterator.range(0, treeNodeIds.length - 1).forall(i => order.lt(treeNodeIds(i), treeNodeIds(i + 1))))

  private val hash = {
    val treeIds = treeNodeIds.map(_._1)
    val nodeIds = treeNodeIds.map(_._2)
    numPartitions * depth * (numCols + int.toInt(treeIds.sum) + int.toInt(treeIds.min) + int.toInt(treeIds.max)
      + inn.toInt(nodeIds.sum) + inn.toInt(nodeIds.max) + inn.toInt(nodeIds.min))
  }

  private val nodeInterval = numPartitions.toDouble / treeNodeIds.length

  private val colInterval = nodeInterval / numCols

  override def getPartition(key: Any): Int = key match {
    case null => 0

    case (treeId: T, nodeId: N, colId: C) =>
      val i = ju.Arrays.binarySearch(treeNodeIds, (treeId, nodeId), order.asInstanceOf[ju.Comparator[(T, N)]])
      require(i >= 0)

      val p = i * nodeInterval + inc.toDouble(colId) * colInterval
      math.min(numPartitions - 1, p.toInt)
  }

  override def equals(other: Any): Boolean = other match {
    case p: IDRangePratitioner[T, N, C] =>
      numPartitions == p.numPartitions &&
        numCols == p.numCols && depth == p.depth &&
        treeNodeIds.length == p.treeNodeIds.length &&
        Iterator.range(0, treeNodeIds.length).forall(i => order.equiv(treeNodeIds(i), p.treeNodeIds(i)))

    case _ =>
      false
  }

  override def hashCode: Int = hash

  override def toString: String = {
    s"IDRangePratitioner[${ct.runtimeClass.toString.capitalize}, ${cn.runtimeClass.toString.capitalize}, ${cc.runtimeClass.toString.capitalize}]" +
      s"(numPartitions=$numPartitions, numCols=$numCols, depth=$depth, treeNodeIds=${treeNodeIds.mkString("[", ",", "]")})"
  }
}





