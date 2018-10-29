package org.apache.spark.ml.gbm

import scala.reflect.ClassTag

import org.apache.spark.Partitioner
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.rdd._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD

object VerticalTree extends Serializable with Logging {


  /**
    *
    * @param subBinVecBlocks column-splitted sub-vectors
    * @param boostConf       boosting configure
    * @param baseConf        trees-growth configure
    * @return tree models
    */
  def train[T, N, C, B, H, G](subBinVecBlocks: RDD[KVVector[G, B]],
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

    val sc = subBinVecBlocks.sparkContext

    logInfo(s"Iteration ${baseConf.iteration}: trees growth start")

    var nodeIdBlocks: RDD[(Long, ArrayBlock[N])] = null
    val nodeIdBlocksCheckpointer = new Checkpointer[(Long, ArrayBlock[N])](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel1)

    var agNodeIdBlocks: RDD[ArrayBlock[N]] = null

    val roots = Array.fill(baseConf.numTrees)(LearningNode.create(1))
    val remainingLeaves = Array.fill(baseConf.numTrees)(boostConf.getMaxLeaves - 1)
    val finished = Array.fill(baseConf.numTrees)(false)

    val numVParts = agTreeIdBlocks.getNumPartitions

    var depth = 0
    var splits = Map.empty[(T, N), Split]
    var vPartIds = Array.range(0, baseConf.numTrees)

    while (finished.contains(false) && depth <= boostConf.getMaxDepth) {
      val start = System.nanoTime

      logInfo(s"Iteration ${baseConf.iteration}: Depth $depth: splitting start")

      if (depth == 0) {
        nodeIdBlocks = initializeNodeIdBlocks[T, N](agTreeIdBlocks)
      } else {
        nodeIdBlocks = updateNodeIdBlocks[T, N, C, B, G](subBinVecBlocks, agTreeIdBlocks, agNodeIdBlocks,
          nodeIdBlocks, boostConf, bcBoostConf, splits)
      }
      nodeIdBlocks.setName(s"NodeIdBlocks (Iteration ${baseConf.iteration}, depth $depth)")
      nodeIdBlocksCheckpointer.update(nodeIdBlocks)


      // merge `colSamplingByLevel` into the BaseConf.
      val newBaseConfig = BaseConfig.mergeColSamplingByLevel(boostConf, baseConf, depth)

      vPartIds = boostConf.getVCols[C]().iterator
        .zipWithIndex.filter { case (colIds, _) =>
        colIds.exists { colId =>
          Iterator.range(0, newBaseConfig.numTrees)
            .exists(treeId => newBaseConfig.colSelector.contains(treeId, colId))
        }
      }.map(_._2).toArray


      agNodeIdBlocks = allgatherNodeIdBlocks[N](nodeIdBlocks, numVParts)
        .setName(s"AllGathered NodeIdBlocks (Iteration ${baseConf.iteration}, depth $depth)")

      val data = subBinVecBlocks.zip3(agTreeIdBlocks, agNodeIdBlocks, agGradBlocks)
        .mapPartitionsWithIndex { case (vPartId, iter) =>
          val numCols = bcBoostConf.value.getNumCols
          val localColIds = bcBoostConf.value.getVCols[C](vPartId)
          val numLocalCols = localColIds.length

          iter.flatMap { case (subBinVecBlock, treeIdBlock, nodeIdBlock, gradBlock) =>
            require(treeIdBlock.size == nodeIdBlock.size)
            require(treeIdBlock.size == gradBlock.size)
            require(treeIdBlock.size * numLocalCols == subBinVecBlock.size)

            val valuesIter = subBinVecBlock.iterator
              .map(_._2).grouped(numLocalCols)

            Utils.zip4(valuesIter, treeIdBlock.iterator, nodeIdBlock.iterator, gradBlock.iterator)
              .map { case (values, treeIds, nodeIds, grad) =>
                require(treeIds.length == nodeIds.length)
                val subBinVec = KVVector.sparse[C, B](numCols, localColIds, values.toArray)
                (subBinVec, treeIds, nodeIds, grad)
              }
          }
        }


      val hists = HistogramUpdater.computeLocalHistograms[T, N, C, B, H](data,
        boostConf, newBaseConfig, (n: N) => true)

//      splits = Tree.findSplitsImpl[T, N, C, B, H](hists, boostConf, baseConf, remainingLeaves, depth)

      // update trees
      Tree.updateTrees[T, N](splits, boostConf, baseConf, roots, remainingLeaves, finished, depth)
      logInfo(s"Iteration ${baseConf.iteration}: $depth: growth finished," +
        s" duration ${(System.nanoTime - start) / 1e9} seconds")

      depth += 1
    }

    if (depth >= boostConf.getMaxDepth) {
      logInfo(s"Iteration ${baseConf.iteration}: maxDepth=${boostConf.getMaxDepth} reached, trees growth finished")
    } else {
      logInfo(s"Iteration ${baseConf.iteration}: trees growth finished")
    }

    nodeIdBlocksCheckpointer.clear()

    roots.map(TreeModel.createModel)
  }


  def initializeNodeIdBlocks[T, N](agTreeIdBlocks: RDD[ArrayBlock[T]])
                                  (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                   cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N]): RDD[(Long, ArrayBlock[N])] = {

    import RDDFunctions._

    agTreeIdBlocks.reorgPartitions(Array(0))
      .mapPartitions { iter =>
        var blockId = -1L
        iter.map { treeIdBlock =>
          blockId += 1

          val iter2 = treeIdBlock.iterator
            .map { treeIds => Array.fill(treeIds.length)(inn.one) }

          (blockId, ArrayBlock.build[N](iter2))
        }
      }
  }


  def updateNodeIdBlocks[T, N, C, B, G](subBinVecBlocks: RDD[KVVector[G, B]],
                                        agTreeIdBlocks: RDD[ArrayBlock[T]],
                                        agNodeIdBlocks: RDD[ArrayBlock[N]],
                                        nodeIdBlocks: RDD[(Long, ArrayBlock[N])],
                                        boostConf: BoostConfig,
                                        bcBoostConf: Broadcast[BoostConfig],
                                        splits: Map[(T, N), Split])
                                       (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                        cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                        cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                        cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                        cg: ClassTag[G], ing: Integral[G], neg: NumericExt[G]): RDD[(Long, ArrayBlock[N])] = {
    import RDDFunctions._

    val sc = subBinVecBlocks.sparkContext
    val parallelism = boostConf.getRealParallelism(boostConf.getReduceParallelism, sc.defaultParallelism)







    val localUpdated =
      subBinVecBlocks.zip2(agTreeIdBlocks, agNodeIdBlocks)
        .mapPartitionsWithIndex { case (vPartId, iter) =>
          val numCols = bcBoostConf.value.getNumCols
          val localColIds = bcBoostConf.value.getVCols[C](vPartId)
          val numLocalCols = localColIds.length

          val localSplits = splits.filter { case (_, split) => localColIds.contains(split.colId) }

          if (localSplits.nonEmpty) {
            var blockId = -1L

            iter.map { case (subBinVecBlock, treeIdBlock, nodeIdBlock) =>
              require(treeIdBlock.size == nodeIdBlock.size)
              require(treeIdBlock.size * numLocalCols == subBinVecBlock.size)
              blockId += 1

              val valuesIter = subBinVecBlock.iterator
                .map(_._2).grouped(numLocalCols)

              val iter2 = Utils.zip3(valuesIter, treeIdBlock.iterator, nodeIdBlock.iterator)
                .map { case (values, treeIds, nodeIds) =>
                  require(treeIds.length == nodeIds.length)
                  val subBinVec = KVVector.sparse[C, B](numCols, localColIds, values.toArray)
                  var updated = false

                  val newNodeIds = treeIds.zip(nodeIds)
                    .map { case (treeId, nodeId) =>
                      val split = localSplits.get((treeId, nodeId))
                      if (split.nonEmpty) {
                        updated = true
                        val leftNodeId = inn.plus(nodeId, nodeId)
                        if (split.get.goLeft[B](subBinVec.apply)) {
                          leftNodeId
                        } else {
                          inn.plus(leftNodeId, inn.one)
                        }
                      } else {
                        nodeId
                      }
                    }

                  if (updated) {
                    newNodeIds
                  } else {
                    nen.emptyArray
                  }
                }

              (blockId, ArrayBlock.build[N](iter2))
            }

          } else {
            Iterator.empty
          }
        }


    sc.union(nodeIdBlocks, localUpdated)
      .reduceByKey(func = {
        case (block1, block2) =>
          require(block1.size == block2.size)
          val iter = block1.iterator.zip(block2.iterator)
            .map { case (array1, array2) =>
              if (array1.isEmpty) {
                array2
              } else if (array2.isEmpty) {
                array1
              } else {
                require(array1.length == array2.length)
                var i = 0
                while (i < array1.length) {
                  array1(i) = inn.max(array1(i), array2(i))
                  i += 1
                }
                array1
              }
            }

          ArrayBlock.build[N](iter)
      }, parallelism)
  }


  def allgatherNodeIdBlocks[N](nodeIdBlocks: RDD[(Long, ArrayBlock[N])],
                               numParts: Int)
                              (implicit cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N]): RDD[ArrayBlock[N]] = {

    nodeIdBlocks.flatMap { case (blockId, nodeIdBlock) =>
      Iterator.range(0, numParts).map { partId =>
        ((blockId, partId), nodeIdBlock)
      }
    }.repartitionAndSortWithinPartitions(new Partitioner {

      override def numPartitions: Int = numParts

      override def getPartition(key: Any): Int = key match {
        case (_, destPartId: Int) => destPartId
      }

    }).map(_._2)
  }




  def updateNodeIdBlocks2[T, N, C, B, G](subBinVecBlocks: RDD[KVVector[G, B]],
                                        agTreeIdBlocks: RDD[ArrayBlock[T]],
                                        agNodeIdBlocks: RDD[ArrayBlock[N]],
                                        nodeIdBlocks: RDD[(Long, ArrayBlock[N])],
                                        boostConf: BoostConfig,
                                        bcBoostConf: Broadcast[BoostConfig],
                                        splits: Map[(T, N), Split])
                                       (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                        cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                        cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                        cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                        cg: ClassTag[G], ing: Integral[G], neg: NumericExt[G]): RDD[(Long, ArrayBlock[N])] = {
    import RDDFunctions._

    val sc = subBinVecBlocks.sparkContext
    val parallelism = boostConf.getRealParallelism(boostConf.getReduceParallelism, sc.defaultParallelism)


//    subBinVecBlocks.safeZip2(agTreeIdBlocks, agNodeIdBlocks)







    val localUpdated =
      subBinVecBlocks.zip2(agTreeIdBlocks, agNodeIdBlocks)
        .mapPartitionsWithIndex { case (vPartId, iter) =>
          val numCols = bcBoostConf.value.getNumCols
          val localColIds = bcBoostConf.value.getVCols[C](vPartId)
          val numLocalCols = localColIds.length

          val localSplits = splits.filter { case (_, split) => localColIds.contains(split.colId) }

          if (localSplits.nonEmpty) {
            var blockId = -1L

            iter.map { case (subBinVecBlock, treeIdBlock, nodeIdBlock) =>
              require(treeIdBlock.size == nodeIdBlock.size)
              require(treeIdBlock.size * numLocalCols == subBinVecBlock.size)
              blockId += 1

              val valuesIter = subBinVecBlock.iterator
                .map(_._2).grouped(numLocalCols)

              val iter2 = Utils.zip3(valuesIter, treeIdBlock.iterator, nodeIdBlock.iterator)
                .map { case (values, treeIds, nodeIds) =>
                  require(treeIds.length == nodeIds.length)
                  val subBinVec = KVVector.sparse[C, B](numCols, localColIds, values.toArray)
                  var updated = false

                  val newNodeIds = treeIds.zip(nodeIds)
                    .map { case (treeId, nodeId) =>
                      val split = localSplits.get((treeId, nodeId))
                      if (split.nonEmpty) {
                        updated = true
                        val leftNodeId = inn.plus(nodeId, nodeId)
                        if (split.get.goLeft[B](subBinVec.apply)) {
                          leftNodeId
                        } else {
                          inn.plus(leftNodeId, inn.one)
                        }
                      } else {
                        nodeId
                      }
                    }

                  if (updated) {
                    newNodeIds
                  } else {
                    nen.emptyArray
                  }
                }

              (blockId, ArrayBlock.build[N](iter2))
            }

          } else {
            Iterator.empty
          }
        }


    sc.union(nodeIdBlocks, localUpdated)
      .reduceByKey(func = {
        case (block1, block2) =>
          require(block1.size == block2.size)
          val iter = block1.iterator.zip(block2.iterator)
            .map { case (array1, array2) =>
              if (array1.isEmpty) {
                array2
              } else if (array2.isEmpty) {
                array1
              } else {
                require(array1.length == array2.length)
                var i = 0
                while (i < array1.length) {
                  array1(i) = inn.max(array1(i), array2(i))
                  i += 1
                }
                array1
              }
            }

          ArrayBlock.build[N](iter)
      }, parallelism)
  }

}

