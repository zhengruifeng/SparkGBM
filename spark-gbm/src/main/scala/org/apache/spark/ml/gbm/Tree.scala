package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.rdd._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD


private[gbm] object Tree extends Serializable with Logging {

  val Basic = "basic"
  val Subtract = "subtract"
  val Vote = "vote"

  /**
    *
    * @param gradBlocks grad&hess is recurrent for compression. i.e
    *                   treeIds = [t1,t2,t5,t6], grad-hess = [g1,h1,g2,h2] =>
    *                   {t1:(g1,h1), t2:(g2,h2), t5:(g1,h1), t6:(g2,h2)}
    * @param boostConf  boosting configure
    * @param baseConf   trees-growth configure
    * @return tree models
    */
  def trainHorizontal[T, N, C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                                     treeIdBlocks: RDD[ArrayBlock[T]],
                                     gradBlocks: RDD[ArrayBlock[H]],
                                     boostConf: BoostConfig,
                                     bcBoostConf: Broadcast[BoostConfig],
                                     baseConf: BaseConfig)
                                    (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                     cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                     cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                     cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                     ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Array[TreeModel] = {
    val sc = binVecBlocks.sparkContext

    logInfo(s"Iteration ${baseConf.iteration}: trees growth start")

    var nodeIdBlocks: RDD[ArrayBlock[N]] = null
    val nodeIdBlocksCheckpointer = new Checkpointer[ArrayBlock[N]](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel1)

    val updater = boostConf.getHistogramComputationType match {
      case Basic => new BasicHistogramUpdater[T, N, C, B, H]
      case Subtract => new SubtractHistogramUpdater[T, N, C, B, H]
      case Vote => new VoteHistogramUpdater[T, N, C, B, H]
    }

    val roots = Array.fill(baseConf.numTrees)(LearningNode.create(1))
    val remainingLeaves = Array.fill(baseConf.numTrees)(boostConf.getMaxLeaves - 1)
    val finished = Array.fill(baseConf.numTrees)(false)

    var depth = 0
    var splits = Map.empty[(T, N), Split]

    while (finished.contains(false) && depth <= boostConf.getMaxDepth) {
      val start = System.nanoTime()

      logInfo(s"Iteration ${baseConf.iteration}, Depth $depth: splitting start")

      if (depth == 0) {
        nodeIdBlocks = initializeNodeIdBlocks[T, N](treeIdBlocks)
      } else {
        nodeIdBlocks = updateNodeIdBlocks[T, N, C, B](binVecBlocks, treeIdBlocks, nodeIdBlocks, splits)
      }
      nodeIdBlocks.setName(s"NodeIdBlocks (Iteration ${baseConf.iteration}, depth $depth)")
      nodeIdBlocksCheckpointer.update(nodeIdBlocks)


      splits = findSplits[T, N, C, B, H](binVecBlocks, treeIdBlocks, gradBlocks, nodeIdBlocks, updater,
        boostConf, bcBoostConf, baseConf, splits, remainingLeaves, depth)


      updater.clear()

      // update trees
      updateTrees[T, N](splits, boostConf, baseConf, roots, remainingLeaves, finished, depth)
      logInfo(s"Iteration ${baseConf.iteration}, Depth $depth: growth finished," +
        s" duration ${(System.nanoTime - start) / 1e9} seconds")

      depth += 1
    }

    if (depth >= boostConf.getMaxDepth) {
      logInfo(s"Iteration ${baseConf.iteration}: maxDepth=${boostConf.getMaxDepth} reached, trees growth finished")
    } else {
      logInfo(s"Iteration ${baseConf.iteration}: trees growth finished")
    }

    nodeIdBlocksCheckpointer.clear(true)
    updater.destroy()

    roots.map(TreeModel.createModel)
  }


  /**
    *
    * @param boostConf boosting configure
    * @param baseConf  trees-growth configure
    * @return tree models
    */
  def trainVertical[T, N, C, B, H, G](binVecBlocks: RDD[KVMatrix[C, B]],
                                      treeIdBlocks: RDD[ArrayBlock[T]],
                                      subBinVecBlocks: RDD[KVVector[G, B]],
                                      agTreeIdBlocks: RDD[ArrayBlock[T]],
                                      agGradBlocks: RDD[ArrayBlock[H]],
                                      boostConf: BoostConfig,
                                      bcBoostConf: Broadcast[BoostConfig],
                                      baseConf: BaseConfig)
                                     (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                      cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                      cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                      cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                      ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H],
                                      cg: ClassTag[G], ing: Integral[G], neg: NumericExt[G]): Array[TreeModel] = {
    import RDDFunctions._

    val sc = binVecBlocks.sparkContext

    logInfo(s"Iteration ${baseConf.iteration}: trees growth start")

    var nodeIdBlocks: RDD[ArrayBlock[N]] = null
    val nodeIdBlocksCheckpointer = new Checkpointer[ArrayBlock[N]](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel1)

    val roots = Array.fill(baseConf.numTrees)(LearningNode.create(1))
    val remainingLeaves = Array.fill(baseConf.numTrees)(boostConf.getMaxLeaves - 1)
    val finished = Array.fill(baseConf.numTrees)(false)

    var depth = 0
    var splits = Map.empty[(T, N), Split]

    val numVParts = subBinVecBlocks.getNumPartitions

    while (finished.contains(false) && depth <= boostConf.getMaxDepth) {
      val start = System.nanoTime

      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth: splitting start")

      if (depth == 0) {
        nodeIdBlocks = initializeNodeIdBlocks[T, N](treeIdBlocks)
      } else {
        nodeIdBlocks = updateNodeIdBlocks[T, N, C, B](binVecBlocks, treeIdBlocks, nodeIdBlocks, splits)
      }
      nodeIdBlocks.setName(s"NodeIdBlocks (Iteration ${baseConf.iteration}, depth $depth)")
      nodeIdBlocksCheckpointer.update(nodeIdBlocks)


      // merge `colSamplingByLevel` into the BaseConf.
      val newBaseConfig = BaseConfig.mergeColSamplingByLevel(boostConf, baseConf, depth)


      val vdata = if (depth == 0) {

        subBinVecBlocks.zip2(agTreeIdBlocks, agGradBlocks)
          .mapPartitionsWithIndex { case (vPartId, iter) =>
            val numCols = bcBoostConf.value.getNumCols
            val localColIds = bcBoostConf.value.getVCols[C](vPartId)
            val numLocalCols = localColIds.length

            iter.flatMap { case (subBinVecBlock, treeIdBlock, gradBlock) =>
              require(treeIdBlock.size == gradBlock.size)
              require(treeIdBlock.size * numLocalCols == subBinVecBlock.size)

              val subBinVecIter = subBinVecBlock.iterator
                .map(_._2).grouped(numLocalCols)
                .map { values => KVVector.sparse[C, B](numCols, localColIds, values.toArray) }

              Utils.zip3(subBinVecIter, treeIdBlock.iterator, gradBlock.iterator)
                .map { case (subBinVec, treeIds, grad) =>
                  (subBinVec, treeIds, Array.fill(treeIds.length)(inn.one), grad)
                }
            }
          }

      } else {

        val vPartIds = boostConf.getVCols[C]().iterator
          .zipWithIndex.filter { case (colIds, _) =>
          colIds.exists { colId =>
            Iterator.range(0, newBaseConfig.numTrees)
              .exists(treeId => newBaseConfig.colSelector.contains(treeId, colId))
          }
        }.map(_._2).toArray

        val agNodeIdBlocks = nodeIdBlocks.allgather(numVParts, vPartIds)

        subBinVecBlocks.zip3(agTreeIdBlocks, agGradBlocks, agNodeIdBlocks, false)
          .mapPartitionsWithIndex { case (vPartId, iter) =>
            val numCols = bcBoostConf.value.getNumCols
            val localColIds = bcBoostConf.value.getVCols[C](vPartId)
            val numLocalCols = localColIds.length

            iter.flatMap { case (subBinVecBlock, treeIdBlock, gradBlock, nodeIdBlock) =>
              require(treeIdBlock.size == gradBlock.size)
              require(treeIdBlock.size == nodeIdBlock.size)
              require(treeIdBlock.size * numLocalCols == subBinVecBlock.size)

              val subBinVecIter = subBinVecBlock.iterator
                .map(_._2).grouped(numLocalCols)
                .map { values => KVVector.sparse[C, B](numCols, localColIds, values.toArray) }

              Utils.zip4(subBinVecIter, treeIdBlock.iterator, nodeIdBlock.iterator, gradBlock.iterator())
            }
          }
      }

      val hists = HistogramUpdater.computeLocalHistograms[T, N, C, B, H](vdata,
        boostConf, newBaseConfig, (n: N) => true)

      splits = findSplitsImpl[T, N, C, B, H](hists, boostConf, bcBoostConf, baseConf, remainingLeaves, depth)

      // update trees
      updateTrees[T, N](splits, boostConf, baseConf, roots, remainingLeaves, finished, depth)
      logInfo(s"Iteration ${baseConf.iteration}: $depth: growth finished," +
        s" duration ${(System.nanoTime - start) / 1e9} seconds")

      depth += 1
    }

    if (depth >= boostConf.getMaxDepth) {
      logInfo(s"Iteration ${baseConf.iteration}: maxDepth=${boostConf.getMaxDepth} reached, trees growth finished")
    } else {
      logInfo(s"Iteration ${baseConf.iteration}: trees growth finished")
    }

    nodeIdBlocksCheckpointer.clear(true)

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

      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth: splitting finished, $numFinished trees" +
        s" growth finished, ${splits.size} leaves split, total gain=${splits.valuesIterator.map(_.gain).sum}.")

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
    import RDDFunctions._

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
                                gradBlocks: RDD[ArrayBlock[H]],
                                nodeIdBlocks: RDD[ArrayBlock[N]],
                                updater: HistogramUpdater[T, N, C, B, H],
                                boostConf: BoostConfig,
                                bcBoostConf: Broadcast[BoostConfig],
                                baseConf: BaseConfig,
                                splits: Map[(T, N), Split],
                                remainingLeaves: Array[Int],
                                depth: Int)
                               (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Map[(T, N), Split] = {

    import RDDFunctions._

    val data = binVecBlocks.zip3(treeIdBlocks, nodeIdBlocks, gradBlocks)
      .flatMap { case (binVecBlock, treeIdBlock, nodeIdBlock, gradBlock) =>
        require(binVecBlock.size == treeIdBlock.size)
        require(binVecBlock.size == nodeIdBlock.size)
        require(binVecBlock.size == gradBlock.size)

        Utils.zip4(binVecBlock.iterator, treeIdBlock.iterator, nodeIdBlock.iterator, gradBlock.iterator)
      }


    updater match {
      case _: SubtractHistogramUpdater[_, _, _, _, _] =>
        // If we update histograms by subtraction, we should compute histogram on all
        // columns selected by `colSamplingByTree`, since the updated histograms will be
        // used in next level, and `colSamplingByLevel` in each level perform differently.
        val histograms = updater.update(data, boostConf, baseConf, splits, depth)
          .setName(s"Histograms (Iteration: ${baseConf.iteration}, depth: $depth)")

        // perform column sampling by level
        val sampled = if (boostConf.getColSampleRateByLevel < 1) {
          histograms.sample(false, boostConf.getColSampleRateByLevel, baseConf.iteration + depth)
        } else {
          histograms
        }

        findSplitsImpl[T, N, C, B, H](sampled, boostConf, bcBoostConf, baseConf, remainingLeaves, depth)


      case _ =>
        // For other types, we do not need to cache intermediate histograms,
        // so we can merge `colSamplingByLevel` into the BaseConf.
        val newBaseConfig = BaseConfig.mergeColSamplingByLevel(boostConf, baseConf, depth)

        val histograms = updater.update(data, boostConf, newBaseConfig, splits, depth)
          .setName(s"Histograms (Iteration: ${baseConf.iteration}, depth: $depth)")

        findSplitsImpl[T, N, C, B, H](histograms, boostConf, bcBoostConf, baseConf, remainingLeaves, depth)
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
                                    baseConf: BaseConfig,
                                    remainingLeaves: Array[Int],
                                    depth: Int)
                                   (implicit ct: ClassTag[T], int: Integral[T],
                                    cn: ClassTag[N], inn: Integral[N],
                                    cc: ClassTag[C], inc: Integral[C],
                                    cb: ClassTag[B], inb: Integral[B],
                                    ch: ClassTag[H], nuh: Numeric[H]): Map[(T, N), Split] = {
    val sc = histograms.sparkContext

    val bcRemainingLeaves = sc.broadcast(remainingLeaves)

    val parallelism = boostConf.getRealParallelism(boostConf.getTrialParallelism, sc.defaultParallelism)
    val repartitioned = if (parallelism == histograms.getNumPartitions) {
      histograms
    } else if (parallelism < histograms.getNumPartitions) {
      histograms.coalesce(parallelism, false)
    } else {
      import RDDFunctions._
      histograms.extendPartitions(parallelism)
    }

    val (splits, Array(numTrials, numSplits, numDenses, sumSize, nnz)) =
      repartitioned.mapPartitionsWithIndex { case (partId, iter) =>
        val boostConfig_ = bcBoostConf.value

        val splits = mutable.OpenHashMap.empty[(T, N), Split]

        var numTrials = 0L
        var numSplits = 0L
        var numDenses = 0L
        var sumSize = 0L
        var nnz = 0L

        iter.foreach { case ((treeId, nodeId, colId), hist) =>
          numTrials += 1
          sumSize += hist.size
          nnz += hist.nnz
          if (hist.isDense) {
            numDenses += 1
          }

          Split.split[H](inc.toInt(colId), hist.toArray, boostConfig_, baseConf)
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
          val splits = (splits1 ++ splits2)
            .groupBy(_._1).mapValues(_.map(_._2).maxBy(_.gain)).toArray
            .groupBy(_._1._1).iterator.flatMap { case (treeId, array) =>
            val rem = bcRemainingLeaves.value(int.toInt(treeId))
            array.sortBy(_._2.gain).takeRight(rem)
          }.toArray

          Iterator.range(0, metrics1.length).foreach(i => metrics1(i) += metrics2(i))
          (splits, metrics1)
      }, boostConf.getAggregationDepth)


    logInfo(s"Depth $depth: $numTrials trials -> $numSplits splits -> ${splits.length} best splits, " +
      s"fraction of sparse histograms: ${1 - numDenses.toDouble / numTrials}, " +
      s"sparsity of histogram: ${1 - nnz.toDouble / sumSize}")

    bcRemainingLeaves.destroy(true)

    splits.toMap
  }
}



