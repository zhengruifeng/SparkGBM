package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD

private[gbm] object Tree extends Serializable with Logging {

  val Basic = "basic"
  val Subtract = "subtract"
  val Vote = "vote"

  /**
    *
    * @param data      instances containing (bins, treeIds, grad-hess), grad&hess is recurrent for compression. i.e
    *                  treeIds = [t1,t2,t5,t6], grad-hess = [g1,h1,g2,h2] -> {t1:(g1,h1), t2:(g2,h2), t5:(g1,h1), t6:(g2,h2)}
    * @param boostConf boosting configure
    * @param baseConf  trees-growth configure
    * @return tree models
    */
  def train[T, N, C, B, H](data: RDD[(KVVector[C, B], Array[T], Array[H])],
                           boostConf: BoostConfig,
                           baseConf: BaseConfig)
                          (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                           cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                           cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                           cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                           ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Array[TreeModel] = {
    logInfo(s"Iteration ${baseConf.iteration}: trees growth start")

    val sc = data.sparkContext

    var nodeIdBlocks = sc.emptyRDD[ArrayBlock[N]]
    val nodeIdBlocksCheckpointer = new Checkpointer[ArrayBlock[N]](sc, boostConf.getCheckpointInterval, boostConf.getStorageLevel)

    val histogramComputer = boostConf.getHistogramComputationType match {
      case Basic => new BasicHistogramComputer[T, N, C, B, H]
      case Subtract => new SubtractHistogramComputer[T, N, C, B, H]
      case Vote => new VoteHistogramComputer[T, N, C, B, H]
    }

    val roots = Array.fill(baseConf.numTrees)(LearningNode.create(1))
    val remainingLeaves = Array.fill(baseConf.numTrees)(boostConf.getMaxLeaves - 1)
    val finished = Array.fill(baseConf.numTrees)(false)

    var depth = 0
    var splits = Map.empty[(T, N), Split]

    while (finished.contains(false) && depth <= boostConf.getMaxDepth) {
      val start = System.nanoTime

      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth: splitting start")

      if (depth == 0) {
        nodeIdBlocks = computeNodeIdBlocks[T, N, C, B, H](data, boostConf)
      } else {
        nodeIdBlocks = updateNodeIdBlocks[T, N, C, B, H](data, nodeIdBlocks, boostConf, splits)
      }
      nodeIdBlocks.setName(s"NodeIdBlocks (Iteration ${baseConf.iteration}, depth $depth)")
      nodeIdBlocksCheckpointer.update(nodeIdBlocks)


      // compute histograms
      val histograms = histogramComputer.compute(data.zip(nodeIdBlocks.flatMap(_.iterator)), boostConf, baseConf, splits, depth)

      // find best splits
      splits = findSplits[T, N, C, B, H](histograms, boostConf, baseConf, remainingLeaves, depth)

      histogramComputer.clear()

      // update trees
      updateTrees[T, N](splits, boostConf, baseConf, roots, remainingLeaves, finished, depth)
      logInfo(s"Iteration ${baseConf.iteration}: $depth: growth finished, duration ${(System.nanoTime - start) / 1e9} seconds")

      depth += 1
    }

    if (depth >= boostConf.getMaxDepth) {
      logInfo(s"Iteration ${baseConf.iteration}: maxDepth=${boostConf.getMaxDepth} reached, trees growth finished")
    } else {
      logInfo(s"Iteration ${baseConf.iteration}: trees growth finished")
    }

    nodeIdBlocksCheckpointer.clear()
    histogramComputer.destroy()

    roots.map(TreeModel.createModel)
  }


  /**
    * update trees and other metrics
    *
    * @param splits          best splits
    * @param boostConf       boosting config
    * @param baseConf        base model config
    * @param roots           root nodes of trees, will be updated
    * @param remainingLeaves number of remained leaves of trees, will be updated
    * @param finished        indicates that where a tree has stopped growth, will be updated
    * @param depth           current depth
    */
  def updateTrees[T, N](splits: Map[(T, N), Split],
                        boostConf: BoostConfig,
                        baseConf: BaseConfig,
                        roots: Array[LearningNode],
                        remainingLeaves: Array[Int],
                        finished: Array[Boolean],
                        depth: Int)
                       (implicit ct: ClassTag[T], int: Integral[T],
                        cn: ClassTag[N], inn: Integral[N]): Unit = {

    if (splits.nonEmpty) {
      val cnts = splits.keysIterator.toSeq.groupBy(_._1).mapValues(_.length)
      var numFinished = 0

      Iterator.range(0, baseConf.numTrees).filterNot(finished).foreach { treeId =>
        val rem = remainingLeaves(treeId)
        val cnt = cnts.getOrElse(int.fromInt(treeId), 0)
        require(cnt <= rem)

        if (cnt == 0 || cnt == rem) {
          finished(treeId) = true
          numFinished += 1
        }

        remainingLeaves(treeId) -= cnt
      }

      updateRoots[T, N](roots, depth, splits)

      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth: splitting finished, $numFinished trees growth finished," +
        s" ${splits.size} leaves split, total gain=${splits.valuesIterator.map(_.gain).sum}.")

    } else {
      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth: no more splits found, trees growth finished.")
      Iterator.range(0, finished.length).foreach(finished(_) = true)
    }
  }


  /**
    * update roots of trees
    *
    * @param roots  roots of trees
    * @param splits splits of leaves
    */
  def updateRoots[T, N](roots: Array[LearningNode],
                        depth: Int,
                        splits: Map[(T, N), Split])
                       (implicit ct: ClassTag[T], int: Integral[T],
                        cn: ClassTag[N], inn: Integral[N]): Unit = {
    if (splits.nonEmpty) {
      val minNodeId = inn.fromInt(1 << depth)

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


  def computeNodeIdBlocks[T, N, C, B, H](data: RDD[(KVVector[C, B], Array[T], Array[H])],
                                         boostConf: BoostConfig)
                                        (implicit ct: ClassTag[T], int: Integral[T],
                                         cn: ClassTag[N], inn: Integral[N],
                                         cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                         cb: ClassTag[B], inb: Integral[B],
                                         ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[ArrayBlock[N]] = {
    val blockSize = boostConf.getBlockSize

    data.mapPartitions { iter =>
      iter.map { case (_, treeIds, _) => treeIds.length }
        .grouped(blockSize)
        .map { lens =>
          val it = lens.iterator.map(len => Array.fill(len)(inn.one))
          ArrayBlock.build[N](it)
        }
    }
  }


  def updateNodeIdBlocks[T, N, C, B, H](data: RDD[(KVVector[C, B], Array[T], Array[H])],
                                        nodeIdBlocks: RDD[ArrayBlock[N]],
                                        boostConf: BoostConfig,
                                        splits: Map[(T, N), Split])
                                       (implicit ct: ClassTag[T], int: Integral[T],
                                        cn: ClassTag[N], inn: Integral[N],
                                        cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                        cb: ClassTag[B], inb: Integral[B],
                                        ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[ArrayBlock[N]] = {
    val blockSize = boostConf.getBlockSize

    data.zip(nodeIdBlocks.flatMap(_.iterator))
      .mapPartitions { iter =>
        val it = iter.map { case ((bins, treeIds, _), nodeIds) =>
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

        it.grouped(blockSize)
          .map { seq => ArrayBlock.build(seq.iterator) }
      }
  }


  /**
    * Search the optimal splits on all leaves
    *
    * @param nodeHistograms histogram data of leaves nodes
    * @return optimal splits for each node
    */
  def findSplits[T, N, C, B, H](nodeHistograms: RDD[((T, N, C), KVVector[B, H])],
                                boostConf: BoostConfig,
                                baseConf: BaseConfig,
                                remainingLeaves: Array[Int],
                                depth: Int)
                               (implicit ct: ClassTag[T], int: Integral[T],
                                cn: ClassTag[N], inn: Integral[N],
                                cc: ClassTag[C], inc: Integral[C],
                                cb: ClassTag[B], inb: Integral[B],
                                ch: ClassTag[H], nuh: Numeric[H], fdh: NumericExt[H]): Map[(T, N), Split] = {
    val sc = nodeHistograms.sparkContext

    val bcRemainingLeaves = sc.broadcast(remainingLeaves)

    // column sampling by level
    val sampled = if (boostConf.getHistogramComputationType == Subtract && boostConf.getColSampleRateByLevel < 1) {
      // In `SubtractHistogramComputer`, level-sampling is not applied
      nodeHistograms.sample(false, boostConf.getColSampleRateByLevel, baseConf.iteration + depth)
    } else {
      nodeHistograms
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
      repartitioned.mapPartitionsWithIndex { case (partId, iter) =>
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

        if (partId == 0 || metrics.head > 0) {
          val splits2 = splits.toArray
            .groupBy(_._1._1).iterator
            .flatMap { case (treeId, array) =>
              val rem = bcRemainingLeaves.value(int.toInt(treeId))
              array.sortBy(_._2.gain).takeRight(rem)
            }.toArray

          Iterator.single((splits2, metrics))

        } else {
          Iterator.empty
        }

      }.treeReduce(f = {
        case ((splits1, metrics1), (splits2, metrics2)) =>
          val splits = (splits1 ++ splits2)
            .groupBy(_._1).mapValues(_.map(_._2).maxBy(_.gain)).toArray
            .groupBy(_._1._1).iterator.flatMap { case (treeId, array) =>
            val rem = bcRemainingLeaves.value(int.toInt(treeId))
            array.sortBy(_._2.gain).takeRight(rem)
          }.toArray

          Iterator.range(0, metrics1.length).foreach(i => metrics1(i) += metrics2(i))
          (splits, metrics1)
      }, boostConf.getAggregationDepth)


    logInfo(s"Depth $depth: $numTrials trials -> $numSplits splits -> ${splits.length} best splits")
    logInfo(s"Depth $depth: Fraction of sparse histograms: ${1 - numDenses / numTrials}, " +
      s"sparsity of histogram: ${1 - nnz / sum}")

    bcRemainingLeaves.destroy(false)

    splits.toMap
  }
}
