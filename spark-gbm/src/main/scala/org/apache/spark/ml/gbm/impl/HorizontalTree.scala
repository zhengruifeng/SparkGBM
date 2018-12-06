package org.apache.spark.ml.gbm.impl

import scala.reflect.ClassTag

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.rdd.RDDFunctions._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD


object HorizontalTree extends Logging {

  /**
    *
    * @param gradBlocks grad&hess is recurrent for compression. i.e
    *                   treeIds = [t1,t2,t5,t6], grad-hess = [g1,h1,g2,h2] =>
    *                   {t1:(g1,h1), t2:(g2,h2), t5:(g1,h1), t6:(g2,h2)}
    * @param boostConf  boosting configure
    * @param baseConf   trees-growth configure
    * @return tree models
    */
  def train[T, N, C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
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
    val tic0 = System.nanoTime()
    val sc = binVecBlocks.sparkContext
    logInfo(s"Iteration ${baseConf.iteration}: trees growth start")


    var nodeIdBlocks: RDD[ArrayBlock[N]] = null
    val nodeIdBlocksCheckpointer = new Checkpointer[ArrayBlock[N]](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel1)


    val updater = boostConf.getHistogramComputationType match {
      case Tree.Basic =>
        new BasicHistogramUpdater[T, N, C, B, H]

      case Tree.Vote =>
        new VoteHistogramUpdater[T, N, C, B, H]

      case Tree.Subtract =>
        require(boostConf.getSubSampleRateByLevel == 1)
        new SubtractHistogramUpdater[T, N, C, B, H]
    }


    val roots = Array.fill(boostConf.getNumTrees)(LearningNode.create(1))
    val remainingLeaves = Array.fill(boostConf.getNumTrees)(boostConf.getMaxLeaves - 1)
    val finished = Array.fill(boostConf.getNumTrees)(false)

    var depth = 0
    var splits = Map.empty[(T, N), Split]

    while (finished.contains(false) && depth <= boostConf.getMaxDepth) {
      val tic1 = System.nanoTime()

      logInfo(s"Iter ${baseConf.iteration}, Depth $depth: splitting start")

      if (depth == 0) {
        nodeIdBlocks = Tree.initializeNodeIdBlocks[T, N](treeIdBlocks)
      } else {
        nodeIdBlocks = Tree.updateNodeIdBlocks[T, N, C, B](binVecBlocks, treeIdBlocks, nodeIdBlocks, splits)
      }
      nodeIdBlocks.setName(s"Iter ${baseConf.iteration}, depth $depth: NodeIds")
      nodeIdBlocksCheckpointer.update(nodeIdBlocks)


      splits = Tree.findSplits[T, N, C, B, H](binVecBlocks.zip3(treeIdBlocks, nodeIdBlocks, gradBlocks), updater,
        boostConf, bcBoostConf, baseConf, splits, remainingLeaves, depth)


      // update trees
      Tree.updateTrees[T, N](splits, boostConf, baseConf, roots, remainingLeaves, finished, depth, false)
      logInfo(s"Iter ${baseConf.iteration}, Depth $depth: growth finished," +
        s" duration ${(System.nanoTime() - tic1) / 1e9} seconds")

      updater.clear()
      depth += 1
    }


    if (depth >= boostConf.getMaxDepth) {
      logInfo(s"Iter ${baseConf.iteration}: maxDepth=${boostConf.getMaxDepth} reached, " +
        s"trees growth finished, duration ${(System.nanoTime() - tic0) / 1e9} seconds")
    } else {
      logInfo(s"Iter ${baseConf.iteration}: trees growth finished, " +
        s"duration ${(System.nanoTime() - tic0) / 1e9} seconds")
    }


    nodeIdBlocksCheckpointer.clear(false)
    updater.destroy()


    roots.map(TreeModel.createModel)
  }
}
