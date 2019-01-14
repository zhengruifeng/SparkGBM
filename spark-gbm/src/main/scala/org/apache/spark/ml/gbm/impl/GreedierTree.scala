package org.apache.spark.ml.gbm.impl

import scala.collection.mutable
import scala.reflect.ClassTag

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom


object GreedierTree extends Logging {


  /**
    *
    * @param boostConf boosting configure
    * @param treeConf  trees-growth configure
    * @return tree models
    */
  def train[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                           labelBlocks: RDD[ArrayBlock[H]],
                           binVecBlocks: RDD[KVMatrix[C, B]],
                           rawPredBlocks: RDD[ArrayBlock[H]],
                           treeIdBlocks: RDD[ArrayBlock[T]],
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

    var auxilaryBlocks = initializeAuxilaryBlocks[T, N, H](rawPredBlocks, treeIdBlocks)

    val auxilaryBlocksCheckpointer = new Checkpointer[(ArrayBlock[N], ArrayBlock[H])](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel1)

    auxilaryBlocksCheckpointer.update(auxilaryBlocks)

    val gradBlocks = computeGradBlocks[T, N, C, B, H](auxilaryBlocks, weightBlocks, labelBlocks, bcBoostConf)

    val rootWeights = computeRootWeights[T, H](gradBlocks, treeIdBlocks, boostConf, treeConf)
    val roots = Array.tabulate(boostConf.getNumTrees)(i => LearningNode.create(1, rootWeights.getOrElse(i, 0.0F)))


    val updater = boostConf.getHistogramComputationType match {
      case Tree.Vote =>
        new VoteHistogramUpdater[T, N, C, B, H]

      case Tree.Basic =>
        new BasicHistogramUpdater[T, N, C, B, H]
    }


    val remainingLeaves = Array.fill(boostConf.getNumTrees)(boostConf.getMaxLeaves - 1)
    val finished = Array.fill(boostConf.getNumTrees)(false)

    var depth = 0
    var splits = Map.empty[(T, N), Split]

    while (finished.contains(false) && depth < boostConf.getMaxDepth) {
      val tic1 = System.nanoTime()

      logInfo(s"Iter ${treeConf.iteration}, Depth $depth: splitting start")


      if (depth == 0) {
        auxilaryBlocks = updateAuxilaryBlocks[T, N, H](auxilaryBlocks, treeIdBlocks, boostConf, rootWeights)
      } else {
        auxilaryBlocks = updateAuxilaryBlocks[T, N, C, B, H](auxilaryBlocks, binVecBlocks, treeIdBlocks, bcBoostConf, splits)
      }
      auxilaryBlocks.setName(s"Iter ${treeConf.iteration}, depth $depth: NodeIds & NodePreds")
      auxilaryBlocksCheckpointer.update(auxilaryBlocks)


      val nodeIdBlocks = auxilaryBlocks.map(_._1)
        .setName(s"Iter ${treeConf.iteration}, depth $depth: NodeIds")

      val gradBlocks = computeGradBlocks[T, N, C, B, H](auxilaryBlocks, weightBlocks, labelBlocks, bcBoostConf)
        .setName(s"Iter ${treeConf.iteration}, depth $depth: Grads")


      splits = Tree.findSplits[T, N, C, B, H](binVecBlocks, treeIdBlocks, nodeIdBlocks, gradBlocks, updater,
        boostConf, bcBoostConf, treeConf, bcTreeConf, splits, remainingLeaves, depth)


      // update trees
      Tree.updateTrees[T, N](splits, boostConf, treeConf, roots, remainingLeaves, finished, depth, true)
      logInfo(s"Iter ${treeConf.iteration}, Depth $depth: growth finished," +
        s" duration ${(System.nanoTime() - tic1) / 1e9} seconds")

      updater.clear()
      depth += 1
    }


    if (depth == boostConf.getMaxDepth) {
      logInfo(s"Iter ${treeConf.iteration}: maxDepth=${boostConf.getMaxDepth} reached, " +
        s"trees growth finished, duration ${(System.nanoTime() - tic0) / 1e9} seconds")
    } else {
      logInfo(s"Iter ${treeConf.iteration}: trees growth finished, " +
        s"duration ${(System.nanoTime() - tic0) / 1e9} seconds")
    }


    auxilaryBlocksCheckpointer.clear(false)
    updater.destroy()


    if (treeConf.colSampledAhead) {
      roots.map(root => TreeModel.createModel(root, treeConf.sortedIndices))
    } else {
      roots.map(TreeModel.createModel)
    }
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

            val nodePredIter = Utils.zip2(rawBlock.iterator, treeIdBlock.iterator)
              .map { case (rawSeq, treeIds) =>
                if (treeIds.isEmpty) {
                  neh.emptyArray
                } else if (rawSeq.length == treeIds.length) {
                  rawSeq
                } else {
                  require(treeIds.length % rawSeq.length == 0)
                  val nodePred = Array.ofDim[H](treeIds.length)
                  var offset = 0
                  while (offset < nodePred.length) {
                    System.arraycopy(rawSeq, 0, nodePred, offset, rawSeq.length)
                    offset += rawSeq.length
                  }
                  nodePred
                }
              }

            val nodePredBlock = ArrayBlock.build[H](nodePredIter)

            (nodeIdBlock, nodePredBlock)
          }
    }
  }


  def updateAuxilaryBlocks[T, N, H](auxilaryBlocks: RDD[(ArrayBlock[N], ArrayBlock[H])],
                                    treeIdBlocks: RDD[ArrayBlock[T]],
                                    boostConf: BoostConfig,
                                    rootWeights: Map[Int, Float])
                                   (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                    cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                    ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[(ArrayBlock[N], ArrayBlock[H])] = {
    import nuh._

    val scaledWeights = rootWeights.map { case (treeId, weight) =>
      (int.fromInt(treeId), neh.fromDouble(weight * boostConf.getStepSize))
    }


    auxilaryBlocks.zipPartitions(treeIdBlocks) {
      case (auxilaryBlockIter, treeIdBlockIter) =>
        Utils.zip2(auxilaryBlockIter, treeIdBlockIter)
          .map { case ((nodeIdBlock, nodePredBlock), treeIdBlock) =>
            require(nodeIdBlock.size == nodePredBlock.size)
            require(nodeIdBlock.size == treeIdBlock.size)

            val newNodePredIter = Utils.zip2(nodePredBlock.iterator, treeIdBlock.iterator)
              .map { case (nodePred, treeIds) =>
                require(nodePred.length == treeIds.length)

                var i = 0
                while (i < treeIds.length) {
                  val treeId = treeIds(i)
                  nodePred(i) += scaledWeights.getOrElse(treeId, nuh.zero)
                  i += 1
                }

                nodePred
              }

            val newNodePredBlock = ArrayBlock.build[H](newNodePredIter)

            (nodeIdBlock, newNodePredBlock)
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
          .map { case ((nodeIdBlock, nodePredBlock), binVecBlock, treeIdBlock) =>
            require(nodeIdBlock.size == nodePredBlock.size)
            require(nodeIdBlock.size == binVecBlock.size)
            require(nodeIdBlock.size == treeIdBlock.size)

            val iter = Utils.zip4(binVecBlock.iterator, treeIdBlock.iterator, nodeIdBlock.iterator, nodePredBlock.iterator)
              .map { case (binVec, treeIds, nodeIds, nodePred) =>
                require(treeIds.length == nodeIds.length)
                require(treeIds.length == nodePred.length)

                var i = 0
                while (i < treeIds.length) {
                  val treeId = treeIds(i)
                  val nodeId = nodeIds(i)

                  splits.get((treeId, nodeId))
                    .foreach { split =>
                      val leftNodeId = inn.plus(nodeId, nodeId)
                      if (split.goLeft[B](binVec.apply)) {
                        nodeIds(i) = leftNodeId
                        nodePred(i) += neh.fromFloat(split.leftWeight) * stepSize
                      } else {
                        nodeIds(i) = inn.plus(leftNodeId, inn.one)
                        nodePred(i) += neh.fromFloat(split.rightWeight) * stepSize
                      }
                    }

                  i += 1
                }

                (nodeIds, nodePred)
              }

            val nodeIdBlockBuilder = new ArrayBlockBuilder[N]
            val nodePredBlockBuilder = new ArrayBlockBuilder[H]

            while (iter.hasNext) {
              val (nodeIds, nodePred) = iter.next()
              nodeIdBlockBuilder += nodeIds
              nodePredBlockBuilder += nodePred
            }

            (nodeIdBlockBuilder.result(), nodePredBlockBuilder.result())
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
          .map { case ((nodeIdBlock, nodePredBlock), weightBlock, labelBlock) =>
            require(nodeIdBlock.size == nodePredBlock.size)
            require(nodeIdBlock.size == weightBlock.size)
            require(nodeIdBlock.size == labelBlock.size)

            val iter = Utils.zip4(nodeIdBlock.iterator, nodePredBlock.iterator, weightBlock.iterator, labelBlock.iterator)
              .map { case (nodeIds, nodePred, weight, label) =>
                require(nodeIds.length % rawSize == 0)
                require(nodeIds.length == nodePred.length)

                val gradArr = Array.ofDim[H](nodeIds.length << 1)

                if (nodePred.length == rawSize) {
                  val score = objFunc.transform(neh.toDouble(nodePred))
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
                  nodePred.grouped(rawSize)
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


  def computeRootWeights[T, H](gradBlocks: RDD[ArrayBlock[H]],
                               treeIdBlocks: RDD[ArrayBlock[T]],
                               boostConf: BoostConfig,
                               treeConf: TreeConfig)
                              (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                               ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Map[Int, Float] = {

    val seed = boostConf.getSeed.toInt * treeConf.iteration + 1
    val rate = boostConf.getColSampleRateByNode

    gradBlocks.zipPartitions(treeIdBlocks) {
      case (gradBlockIter, treeIdBlockIter) =>
        val iter = Utils.zip2(gradBlockIter, treeIdBlockIter)
        val sum = mutable.OpenHashMap.empty[T, (H, H)]

        val rng = new XORShiftRandom(seed)

        while (iter.hasNext) {
          val (gradBlock, treeIdBlock) = iter.next()
          require(gradBlock.size == treeIdBlock.size)

          val iter2 = Utils.zip2(treeIdBlock.iterator, gradBlock.iterator)

          while (iter2.hasNext) {
            val (treeIdArr, gradArr) = iter2.next()

            if (gradArr.nonEmpty) {
              val size = gradArr.length >> 1

              var j = 0
              while (j < treeIdArr.length) {
                if (rng.nextDouble() < rate) {
                  val treeId = treeIdArr(j)
                  val indexGrad = (j % size) << 1
                  val grad = gradArr(indexGrad)
                  val hess = gradArr(indexGrad + 1)
                  val (g0, h0) = sum.getOrElse(treeId, (nuh.zero, nuh.zero))
                  sum.update(treeId, (nuh.plus(g0, grad), nuh.plus(h0, hess)))
                }

                j += 1
              }
            }
          }
        }

        Iterator.single(sum.toArray)

    }.treeReduce(f = {
      case (sum1, sum2) =>
        (sum1 ++ sum2).groupBy(_._1)
          .mapValues { arr => (arr.map(_._2._1).sum, arr.map(_._2._2).sum) }
          .toArray

    }, depth = boostConf.getAggregationDepth)

      .map { case (treeId, (gradSum, hessSum)) =>
        val (weight, _) = Split.computeScore(nuh.toFloat(gradSum), nuh.toFloat(hessSum), boostConf)
        (int.toInt(treeId), weight)
      }.toMap
  }

}


