package org.apache.spark.ml.gbm.impl

import scala.reflect.ClassTag

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.rdd.RDDFunctions._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD


object GreedierTree extends Logging {


  /**
    *
    * @param boostConf boosting configure
    * @param baseConf  trees-growth configure
    * @return tree models
    */
  def train[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                           labelBlocks: RDD[ArrayBlock[H]],
                           binVecBlocks: RDD[KVMatrix[C, B]],
                           rawPredBlocks: RDD[ArrayBlock[H]],
                           treeIdBlocks: RDD[ArrayBlock[T]],
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


    var auxilaryBlocks: RDD[(ArrayBlock[N], ArrayBlock[H])] = null
    val auxilaryBlocksCheckpointer = new Checkpointer[(ArrayBlock[N], ArrayBlock[H])](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel1)


    val updater = boostConf.getHistogramComputationType match {
      case Tree.Vote =>
        new VoteHistogramUpdater[T, N, C, B, H]

      case Tree.Basic =>
        new BasicHistogramUpdater[T, N, C, B, H]
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
        auxilaryBlocks = initializeAuxilaryBlocks[T, N, H](rawPredBlocks, treeIdBlocks)
      } else {
        auxilaryBlocks = updateAuxilaryBlocks[T, N, C, B, H](auxilaryBlocks, binVecBlocks, treeIdBlocks, bcBoostConf, splits)
      }
      auxilaryBlocks.setName(s"Iter ${baseConf.iteration}, depth $depth: NodeIds & LevelPreds")
      auxilaryBlocksCheckpointer.update(auxilaryBlocks)


      val nodeIdBlocks = auxilaryBlocks.map(_._1)
        .setName(s"Iter ${baseConf.iteration}, depth $depth: NodeIds")

      val gradBlocks = computeGradBlocks[T, N, C, B, H](auxilaryBlocks, weightBlocks, labelBlocks, bcBoostConf)
        .setName(s"Iter ${baseConf.iteration}, depth $depth: Grads")


      splits = Tree.findSplits[T, N, C, B, H](binVecBlocks.zip3(treeIdBlocks, nodeIdBlocks, gradBlocks), updater,
        boostConf, bcBoostConf, baseConf, splits, remainingLeaves, depth)


      // update trees
      Tree.updateTrees[T, N](splits, boostConf, baseConf, roots, remainingLeaves, finished, depth, true)
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


    auxilaryBlocksCheckpointer.clear(false)
    updater.destroy()


    roots.map(TreeModel.createModel)
  }


  def initializeAuxilaryBlocks[T, N, H](rawBlocks: RDD[ArrayBlock[H]],
                                        treeIdBlocks: RDD[ArrayBlock[T]])
                                       (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                        cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                        ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[(ArrayBlock[N], ArrayBlock[H])] = {

    rawBlocks.zipPartitions(treeIdBlocks) {
      case (rawBlockIter, treeIdBlockIter) =>

        Utils.zip2(rawBlockIter, treeIdBlockIter)
          .map { case (rawBlock, treeIdBlock) =>
            require(rawBlock.size == treeIdBlock.size)

            val nodeIdIter = treeIdBlock.iterator
              .map { treeIds => Array.fill(treeIds.length)(inn.one) }

            val nodeIdBlock = ArrayBlock.build[N](nodeIdIter)

            val levelPredIter = Utils.zip2(rawBlock.iterator, treeIdBlock.iterator)
              .map { case (rawSeq, treeIds) =>
                if (treeIds.isEmpty) {
                  neh.emptyArray
                } else if (rawSeq.length == treeIds.length) {
                  rawSeq
                } else {
                  require(treeIds.length % rawSeq.length == 0)
                  val levelPred = Array.ofDim[H](treeIds.length)
                  var offset = 0
                  while (offset < levelPred.length) {
                    System.arraycopy(rawSeq, 0, levelPred, offset, rawSeq.length)
                    offset += rawSeq.length
                  }
                  levelPred
                }
              }

            val levelPredBlock = ArrayBlock.build[H](levelPredIter)

            (nodeIdBlock, levelPredBlock)
          }
    }
  }


  def updateAuxilaryBlocks[T, N, C, B, H](auxilaryBlocks: RDD[(ArrayBlock[N], ArrayBlock[H])],
                                          binVecBlocks: RDD[KVMatrix[C, B]],
                                          treeIdBlocks: RDD[ArrayBlock[T]],
                                          bcBoostConf: Broadcast[BoostConfig],
                                          splits: Map[(T, N), Split])
                                         (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                          cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                          cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                          cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                          ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[(ArrayBlock[N], ArrayBlock[H])] = {
    import nuh._

    auxilaryBlocks.zipPartitions(binVecBlocks, treeIdBlocks) {
      case (auxilaryBlockIter, binVecBlockIter, treeIdBlockIter) =>

        val boostConf = bcBoostConf.value
        val stepSize = neh.fromDouble(boostConf.getStepSize)

        Utils.zip3(auxilaryBlockIter, binVecBlockIter, treeIdBlockIter)
          .map { case ((nodeIdBlock, levelPredBlock), binVecBlock, treeIdBlock) =>
            require(nodeIdBlock.size == levelPredBlock.size)
            require(nodeIdBlock.size == binVecBlock.size)
            require(nodeIdBlock.size == treeIdBlock.size)

            val iter = Utils.zip4(binVecBlock.iterator, treeIdBlock.iterator, nodeIdBlock.iterator, levelPredBlock.iterator)
              .map { case (binVec, treeIds, nodeIds, levelPred) =>
                require(treeIds.length == nodeIds.length)
                require(treeIds.length == levelPred.length)

                var i = 0
                while (i < treeIds.length) {
                  val treeId = treeIds(i)
                  val nodeId = nodeIds(i)

                  splits.get((treeId, nodeId))
                    .foreach { split =>
                      val leftNodeId = inn.plus(nodeId, nodeId)
                      if (split.goLeft[B](binVec.apply)) {
                        nodeIds(i) = leftNodeId
                        levelPred(i) += neh.fromFloat(split.leftWeight) * stepSize
                      } else {
                        nodeIds(i) = inn.plus(leftNodeId, inn.one)
                        levelPred(i) += neh.fromFloat(split.rightWeight) * stepSize
                      }
                    }

                  i += 1
                }

                (nodeIds, levelPred)
              }

            val nodeIdBlockBuilder = new ArrayBlockBuilder[N]
            val levelPredBlockBuilder = new ArrayBlockBuilder[H]

            while (iter.hasNext) {
              val (nodeIds, levelPred) = iter.next()
              nodeIdBlockBuilder += nodeIds
              levelPredBlockBuilder += levelPred
            }

            (nodeIdBlockBuilder.result(), levelPredBlockBuilder.result())
          }
    }
  }


  def computeGradBlocks[T, N, C, B, H](auxilaryBlocks: RDD[(ArrayBlock[N], ArrayBlock[H])],
                                       weightBlocks: RDD[CompactArray[H]],
                                       labelBlocks: RDD[ArrayBlock[H]],
                                       bcBoostConf: Broadcast[BoostConfig])
                                      (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                       cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                       cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                       cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                       ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[ArrayBlock[H]] = {
    import nuh._

    auxilaryBlocks.zipPartitions(weightBlocks, labelBlocks) {
      case (auxilaryBlockIter, weightBlockIter, labelBlockIter) =>

        val boostConf = bcBoostConf.value
        val rawSize = boostConf.getRawSize
        val objFunc = boostConf.getObjFunc

        Utils.zip3(auxilaryBlockIter, weightBlockIter, labelBlockIter)
          .map { case ((nodeIdBlock, levelPredBlock), weightBlock, labelBlock) =>
            require(nodeIdBlock.size == levelPredBlock.size)
            require(nodeIdBlock.size == weightBlock.size)
            require(nodeIdBlock.size == labelBlock.size)

            val iter = Utils.zip4(nodeIdBlock.iterator, levelPredBlock.iterator, weightBlock.iterator, labelBlock.iterator)
              .map { case (nodeIds, levelPred, weight, label) =>
                require(nodeIds.length % rawSize == 0)
                require(nodeIds.length == levelPred.length)

                val gradArr = Array.ofDim[H](nodeIds.length << 1)

                if (levelPred.length == rawSize) {
                  val score = objFunc.transform(neh.toDouble(levelPred))
                  val (grad, hess) = objFunc.compute(neh.toDouble(label), score)
                  require(grad.length == rawSize && hess.length == rawSize)

                  var i = 0
                  while (i < rawSize) {
                    val j = i << 1
                    gradArr(j) = neh.fromDouble(grad(i)) * weight
                    gradArr(j + 1) = neh.fromDouble(hess(i)) * weight
                    i += 1
                  }

                } else {

                  var j = 0
                  levelPred.grouped(rawSize)
                    .foreach { pred =>
                      val score = objFunc.transform(neh.toDouble(pred))
                      val (grad, hess) = objFunc.compute(neh.toDouble(label), score)
                      require(grad.length == rawSize && hess.length == rawSize)

                      var i = 0
                      while (i < rawSize) {
                        gradArr(j) = neh.fromDouble(grad(i)) * weight
                        j += 1
                        gradArr(j) = neh.fromDouble(hess(i)) * weight
                        j += 1
                        i += 1
                      }
                    }
                }

                gradArr
              }

            ArrayBlock.build[H](iter)
          }
    }
  }
}
