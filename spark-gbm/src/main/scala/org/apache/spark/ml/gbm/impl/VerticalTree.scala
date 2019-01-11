package org.apache.spark.ml.gbm.impl

import scala.reflect.ClassTag

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.rdd.RDDFunctions._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD


object VerticalTree extends Logging {


  /**
    *
    * @param boostConf boosting configure
    * @param treeConf  trees-growth configure
    * @return tree models
    */
  def train[T, N, C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                           treeIdBlocks: RDD[ArrayBlock[T]],
                           subBinVecBlocks: RDD[KVMatrix[C, B]],
                           blockIds: RDD[Long],
                           agTreeIdBlocks: RDD[ArrayBlock[T]],
                           agGradBlocks: RDD[ArrayBlock[H]],
                           boostConf: BoostConfig,
                           bcBoostConf: Broadcast[BoostConfig],
                           treeConf: TreeConfig,
                           bcTreeConf: Broadcast[TreeConfig])
                          (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                           cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                           cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                           cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                           ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Array[TreeModel] = {
    val tic0 = System.nanoTime()
    val sc = binVecBlocks.sparkContext
    logInfo(s"Iter ${treeConf.iteration}: trees growth start")


    var nodeIdBlocks: RDD[ArrayBlock[N]] = null
    val nodeIdBlocksCheckpointer = new Checkpointer[ArrayBlock[N]](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel1)

    val roots = Array.fill(boostConf.getNumTrees)(LearningNode.create(1))
    val remainingLeaves = Array.fill(boostConf.getNumTrees)(boostConf.getMaxLeaves - 1)
    val finished = Array.fill(boostConf.getNumTrees)(false)

    var depth = 0
    var splits = Map.empty[(T, N), Split]

    val cleaner = new ResourceCleaner

    while (finished.contains(false) && depth < boostConf.getMaxDepth) {
      val tic1 = System.nanoTime()

      logInfo(s"Iter ${treeConf.iteration}: Depth $depth: splitting start")

      if (depth == 0) {
        nodeIdBlocks = Tree.initializeNodeIdBlocks[T, N](treeIdBlocks)
      } else {
        nodeIdBlocks = Tree.updateNodeIdBlocks[T, N, C, B](binVecBlocks, treeIdBlocks, nodeIdBlocks, splits)
      }
      nodeIdBlocks.setName(s"Iter ${treeConf.iteration}, depth $depth: NodeIds")
      nodeIdBlocksCheckpointer.update(nodeIdBlocks)


      // col-sampling by node
      val extraSelector = if (boostConf.getColSampleRateByNode < 1) {
        val seed = boostConf.getSeed.toInt * treeConf.iteration + depth

        // Note that different tree in one base model DO NOT share same col-sampler.
        Some(Selector.create(boostConf.getColSampleRateByNode, boostConf.getNumCols,
          boostConf.getNumTrees, 1, seed))

      } else {
        None
      }


      val vdata = if (depth == 0) {
        subBinVecBlocks.zipPartitions(agTreeIdBlocks, agGradBlocks) {
          case (subBinVecBlockIter, agTreeIdBlockIter, agGradBlockIter) =>

            Utils.zip3(subBinVecBlockIter, agTreeIdBlockIter, agGradBlockIter)
              .map { case (subBinVecBlock, treeIdBlock, gradBlock) =>
                require(subBinVecBlock.size == treeIdBlock.size)
                require(subBinVecBlock.size == gradBlock.size)

                val iter2 = treeIdBlock.iterator
                  .map { treeIds => Array.fill(treeIds.length)(inn.one) }
                val nodeIdBlock = ArrayBlock.build[N](iter2)
                (subBinVecBlock, treeIdBlock, nodeIdBlock, gradBlock)
              }
        }

      } else {

        val unionSelector = if (extraSelector.nonEmpty) {
          Selector.union(treeConf.colSelector, extraSelector.get)
        } else {
          treeConf.colSelector
        }

        val baseVPartIds = boostConf.getBaseVCols[C]()
          .iterator.zipWithIndex
          .filter { case (colIds, _) =>
            colIds.exists(unionSelector.contains[C])
          }.map(_._2).toArray

        val agNodeIdBlocks = VerticalGBM.gatherByLayer(nodeIdBlocks, blockIds,
          baseVPartIds, boostConf, bcBoostConf, cleaner)

        subBinVecBlocks.zip3(agTreeIdBlocks, agNodeIdBlocks, agGradBlocks, false)
      }


      // `computeLocalHistograms` will treat an unavailable column on one partititon as
      // all a zero-value column, so we should filter it in `computeHistogramsVertical`.
      val minNodeId = inn.fromInt(1 << depth)
      val histograms = HistogramUpdater.computeHistogramsVertical[T, N, C, B, H](vdata,
        boostConf, bcBoostConf, bcTreeConf, extraSelector, (n: N) => inn.gteq(n, minNodeId))
        .setName(s"Iter ${treeConf.iteration}, depth $depth: Histograms")


      splits = Tree.findSplitsImpl[T, N, C, B, H](histograms, boostConf, bcBoostConf,
        treeConf, bcTreeConf, remainingLeaves, depth)


      // update trees
      Tree.updateTrees[T, N](splits, boostConf, treeConf, roots, remainingLeaves, finished, depth, false)
      logInfo(s"Iter ${treeConf.iteration}: $depth: growth finished," +
        s" duration ${(System.nanoTime() - tic1) / 1e9} seconds")


      cleaner.clear()
      depth += 1
    }

    if (depth == boostConf.getMaxDepth) {
      logInfo(s"Iter ${treeConf.iteration}: maxDepth=${boostConf.getMaxDepth} reached, " +
        s"trees growth finished, duration ${(System.nanoTime() - tic0) / 1e9} seconds")
    } else {
      logInfo(s"Iter ${treeConf.iteration}: trees growth finished, " +
        s"duration ${(System.nanoTime() - tic0) / 1e9} seconds")
    }

    nodeIdBlocksCheckpointer.clear(false)

    roots.map(TreeModel.createModel)
  }
}

