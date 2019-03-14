package org.apache.spark.ml.gbm.impl

import scala.collection.mutable
import scala.reflect.ClassTag

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.rdd.RDDFunctions._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.unsafe.hash.Murmur3_x86_32
import org.apache.spark.util.random.XORShiftRandom


private[gbm] object Tree extends Logging {

  val Basic = "basic"
  val Subtract = "subtract"
  val Vote = "vote"


  /**
    * update trees and other metrics
    *
    * @param splits          best splits
    * @param boostConf       boosting config
    * @param treeConf        base model config
    * @param roots           root nodes of trees, will be updated
    * @param remainingLeaves number of remained leaves of trees, will be updated
    * @param finished        indicates that where a tree has stopped growth, will be updated
    * @param depth           current depth
    */
  def updateTrees[T, N](splits: Map[(T, N), Split],
                        boostConf: BoostConfig,
                        treeConf: TreeConfig,
                        roots: Array[LearningNode],
                        remainingLeaves: Array[Int],
                        finished: Array[Boolean],
                        depth: Int,
                        additive: Boolean)
                       (implicit ct: ClassTag[T], int: Integral[T],
                        cn: ClassTag[N], inn: Integral[N]): Unit = {

    if (splits.nonEmpty) {
      val counts = splits.keysIterator.toSeq.groupBy(_._1).mapValues(_.length)
      var numFinished = 0

      Iterator.range(0, boostConf.getNumTrees)
        .filterNot(finished).foreach { treeId =>
        val rem = remainingLeaves(treeId)
        val count = counts.getOrElse(int.fromInt(treeId), 0)
        require(count <= rem)

        if (count == 0 || count == rem) {
          finished(treeId) = true
          numFinished += 1
        }

        remainingLeaves(treeId) -= count
      }

      updateRoots[T, N](roots, depth, splits, additive)

      logInfo(s"Iter ${treeConf.iteration}: Depth $depth: splitting finished, $numFinished trees" +
        s" growth finished, ${splits.size} leaves split, total gain=${splits.valuesIterator.map(_.gain).sum}.")

    } else {
      Iterator.range(0, finished.length)
        .foreach(finished(_) = true)
      logInfo(s"Iter ${treeConf.iteration}: Depth $depth: no more splits found, trees growth finished.")
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
                        splits: Map[(T, N), Split],
                        additive: Boolean)
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

            val rightNodeId = leftNodeId + 1
            node.rightNode = Some(LearningNode.create(rightNodeId))

            if (additive) {
              node.leftNode.get.prediction = node.prediction + node.split.get.leftWeight
              node.rightNode.get.prediction = node.prediction + node.split.get.rightWeight

            } else {
              node.leftNode.get.prediction = node.split.get.leftWeight
              node.rightNode.get.prediction = node.split.get.rightWeight
            }
          }
      }
    }
  }


  def initializeNodeIdBlocks[T, N](treeIdBlocks: RDD[ArrayBlock[T]])
                                  (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                   cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N]): RDD[ArrayBlock[N]] = {
    treeIdBlocks.map { treeIdBlock =>
      val iter = treeIdBlock.iterator
        .map { treeIds => Array.fill(treeIds.length)(inn.one) }
      ArrayBlock.build(iter)
    }
  }


  def updateNodeIdBlocks[T, N, C, B](binVecBlocks: RDD[KVMatrix[C, B]],
                                     treeIdBlocks: RDD[ArrayBlock[T]],
                                     nodeIdBlocks: RDD[ArrayBlock[N]],
                                     splits: Map[(T, N), Split])
                                    (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                     cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                     cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                     cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B]): RDD[ArrayBlock[N]] = {
    binVecBlocks.zip2(treeIdBlocks, nodeIdBlocks)
      .map { case (binVecBlock, treeIdBlock, nodeIdBlock) =>
        require(binVecBlock.size == treeIdBlock.size)
        require(binVecBlock.size == nodeIdBlock.size)

        val iter = Utils.zip3(binVecBlock.iterator, treeIdBlock.iterator, nodeIdBlock.iterator)
          .map { case (binVec, treeIds, nodeIds) =>
            require(treeIds.length == nodeIds.length)

            var i = 0
            while (i < treeIds.length) {
              val treeId = treeIds(i)
              val nodeId = nodeIds(i)

              splits.get((treeId, nodeId))
                .foreach { split =>
                  val leftNodeId = inn.plus(nodeId, nodeId)
                  if (split.goLeft[B](binVec.apply)) {
                    nodeIds(i) = leftNodeId
                  } else {
                    nodeIds(i) = inn.plus(leftNodeId, inn.one)
                  }
                }

              i += 1
            }

            nodeIds
          }

        ArrayBlock.build(iter)
      }
  }


  def findSplits[T, N, C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                                treeIdBlocks: RDD[ArrayBlock[T]],
                                nodeIdBlocks: RDD[ArrayBlock[N]],
                                gradBlocks: RDD[ArrayBlock[H]],
                                updater: HistogramUpdater[T, N, C, B, H],
                                boostConf: BoostConfig,
                                bcBoostConf: Broadcast[BoostConfig],
                                treeConf: TreeConfig,
                                bcTreeConf: Broadcast[TreeConfig],
                                splits: Map[(T, N), Split],
                                remainingLeaves: Array[Int],
                                depth: Int)
                               (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Map[(T, N), Split] = {

    updater match {
      case _: SubtractHistogramUpdater[_, _, _, _, _] =>
        // Note: param subSampleRateByNode is always IGNORED!
        // If we update histograms by subtraction, we should compute histogram on all
        // columns selected by `colSamplingByTree`, since the updated histograms will be
        // used in next level, and `colSamplingByNode` in each level perform differently.
        val histograms = updater.update(binVecBlocks, treeIdBlocks, nodeIdBlocks, gradBlocks,
          boostConf, bcBoostConf, treeConf, bcTreeConf, None, splits, depth)
          .setName(s"Iter ${treeConf.iteration}, depth $depth: Histograms")


        val sampledHistograms = if (boostConf.getColSampleRateByNode < 1) {

          // perform node-wise col-sampling

          val threshold = (boostConf.getColSampleRateByNode * Long.MaxValue).toLong
          val seed = boostConf.getSeed.toInt * treeConf.iteration + depth

          histograms.filter { case ((treeId, nodeId, colId), _) =>
            Murmur3_x86_32.hashLong(inc.toLong(colId), seed + int.toInt(treeId) + inn.toInt(nodeId)).abs < threshold
          }

        } else {
          histograms
        }

        findSplitsImpl[T, N, C, B, H](sampledHistograms, boostConf, bcBoostConf,
          treeConf, bcTreeConf, remainingLeaves, depth)


      case _ =>
        // For other types, we do not need to cache intermediate histograms,
        // so we create extra column selector to perform node-wise col-sampling at one time,
        // and randomly mask nodeIds in `nodeIdBlocks` to perform node-wise rowsample.

        val seed = boostConf.getSeed.toInt * treeConf.iteration + depth

        val extraSelector = if (boostConf.getColSampleRateByNode < 1) {
          val numCols = treeConf.getNumCols.getOrElse(boostConf.getNumCols)

          // Note that different tree in one base model DO NOT share same col-sampler.
          Some(Selector.create(boostConf.getColSampleRateByNode, numCols,
            boostConf.getNumTrees, 1, seed))

        } else {
          None
        }


        val maskedNodeIdBlocks = if (boostConf.getSubSampleRateByNode < 1) {

          // perform node-wise row-sampling
          // randomly mask some nodes in histogram computation by setting zero values
          nodeIdBlocks.mapPartitionsWithIndex { case (partId, iter) =>
            val rate = bcBoostConf.value.getSubSampleRateByNode
            val rng = new XORShiftRandom(seed * partId + 1)

            iter.map { nodeIdBlock =>
              val iter2 = nodeIdBlock.iterator
                .map { nodeIds =>
                  var i = 0
                  while (i < nodeIds.length) {
                    if (rng.nextDouble() < rate) {
                      // nodeIds of zero value will be ignored in histogram computation
                      nodeIds(i) = inn.zero
                    }
                    i += 1
                  }

                  nodeIds
                }

              ArrayBlock.build[N](iter2)
            }
          }

        } else {
          nodeIdBlocks
        }

        val histograms = updater.update(binVecBlocks, treeIdBlocks, maskedNodeIdBlocks, gradBlocks,
          boostConf, bcBoostConf, treeConf, bcTreeConf, extraSelector, splits, depth)
          .setName(s"Iter ${treeConf.iteration}, depth $depth: Histograms")

        findSplitsImpl[T, N, C, B, H](histograms, boostConf, bcBoostConf,
          treeConf, bcTreeConf, remainingLeaves, depth)
    }
  }


  /**
    * Search the optimal splits on all leaves
    *
    * @param histograms histogram data of leaves nodes
    * @return optimal splits for each node
    */
  def findSplitsImpl[T, N, C, B, H](histograms: RDD[((T, N, C), KVVector[B, H])],
                                    boostConf: BoostConfig,
                                    bcBoostConf: Broadcast[BoostConfig],
                                    treeConf: TreeConfig,
                                    bcTreeConf: Broadcast[TreeConfig],
                                    remainingLeaves: Array[Int],
                                    depth: Int)
                                   (implicit ct: ClassTag[T], int: Integral[T],
                                    cn: ClassTag[N], inn: Integral[N],
                                    cc: ClassTag[C], inc: Integral[C],
                                    cb: ClassTag[B], inb: Integral[B],
                                    ch: ClassTag[H], nuh: Numeric[H]): Map[(T, N), Split] = {
    val sc = histograms.sparkContext
    val bcRemainingLeaves = sc.broadcast(remainingLeaves)


    val (splits, Array(numTrials, numSplits, numDenses, sumSize, nnz)) =
      histograms.mapPartitionsWithIndex { case (partId, iter) =>
        val boostConf = bcBoostConf.value
        val treeConf = bcTreeConf.value

        val splits = mutable.OpenHashMap.empty[(T, N), Split]

        var numTrials = 0L
        var numSplits = 0L
        var numDenses = 0L
        var sumSize = 0L
        var nnz = 0L

        while (iter.hasNext) {
          val ((treeId, nodeId, colId), hist) = iter.next()
          numTrials += 1
          sumSize += hist.size
          nnz += hist.nnz
          if (hist.isDense) {
            numDenses += 1
          }

          Split.split[H](inc.toInt(colId), hist.toArray, boostConf, treeConf)
            .foreach { split =>
              numSplits += 1
              val prevSplit = splits.get((treeId, nodeId))
              if (prevSplit.isEmpty || prevSplit.get.gain < split.gain) {
                splits.update((treeId, nodeId), split)
              }
            }
        }

        if (numTrials > 0) {
          val filtered = splits.toArray.groupBy(_._1._1).iterator
            .flatMap { case (treeId, array) =>
              val rem = bcRemainingLeaves.value(int.toInt(treeId))
              array.sortBy(_._2.gain).takeRight(rem)
            }.toArray

          Iterator.single((filtered, Array(numTrials, numSplits, numDenses, sumSize, nnz)))

        } else if (partId == 0) {
          // avoid `treeReduce` on empty RDD
          Iterator.single(Array.empty[((T, N), Split)], Array.ofDim[Long](5))

        } else {
          Iterator.empty
        }

      }.treeReduce(f = {
        case ((splits1, metrics1), (splits2, metrics2)) =>
          require(metrics1.length == metrics2.length)

          val splits = (splits1 ++ splits2)
            .groupBy(_._1).mapValues(_.map(_._2).maxBy(_.gain)).toArray
            .groupBy(_._1._1).iterator
            .flatMap { case (treeId, array) =>
              val rem = bcRemainingLeaves.value(int.toInt(treeId))
              array.sortBy(_._2.gain).takeRight(rem)
            }.toArray

          var i = 0
          while (i < metrics1.length) {
            metrics1(i) += metrics2(i)
            i += 1
          }

          (splits, metrics1)
      }, boostConf.getAggregationDepth)


    logInfo(s"Iter: ${treeConf.iteration}, Depth $depth: " +
      s"$numTrials trials -> $numSplits splits -> ${splits.length} best splits, " +
      s"fraction of sparse vectors: ${1 - numDenses.toDouble / numTrials}, " +
      s"sparsity of histogram elements: ${1 - nnz.toDouble / sumSize}")

    bcRemainingLeaves.destroy(true)

    splits.toMap
  }
}


