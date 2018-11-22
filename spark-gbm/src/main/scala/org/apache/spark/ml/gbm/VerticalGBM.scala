package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.Partitioner
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.rdd._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD


object VerticalGBM extends Logging {

  /**
    * implementation of GBM, train a GBMModel, with given types
    *
    * @param trainBlocks  training blocks containing (weight, label, binVec)
    * @param testBlocks   validation blocks containing (weight, label, binVec)
    * @param boostConf    boosting configuration
    * @param discretizer  discretizer to convert raw features into bins
    * @param initialModel inital model
    * @return the model
    */
  def boost[C, B, H](trainBlocks: (RDD[CompactArray[H]], RDD[ArrayBlock[H]], RDD[KVMatrix[C, B]]),
                     testBlocks: Option[(RDD[CompactArray[H]], RDD[ArrayBlock[H]], RDD[KVMatrix[C, B]])],
                     boostConf: BoostConfig,
                     discretizer: Discretizer,
                     initialModel: Option[GBMModel])
                    (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                     cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                     ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): GBMModel = {

    val sc = trainBlocks._1.sparkContext
    boostConf.updateVPartInfo()

    // train blocks
    val (trainWeightBlocks, trainLabelBlocks, trainBinVecBlocks) = trainBlocks
    GBM.touchBlocksAndUpdatePartInfo[H](trainWeightBlocks, boostConf, true)

    // test blocks
    testBlocks.foreach { case (testWeightBlocks, _, _) =>
      GBM.touchBlocksAndUpdatePartInfo(testWeightBlocks, boostConf, false)
    }


    // vertical train sub-binvec blocks
    val subBinVecBlocks = divideBinVecBlocks[C, B](trainBinVecBlocks, boostConf)
      .setName("Train SubBinVecs (Vertical)")
    subBinVecBlocks.persist(boostConf.getStorageLevel2)


    // init tree buffer
    val weightsBuff = mutable.ArrayBuffer.empty[H]
    val treesBuff = mutable.ArrayBuffer.empty[TreeModel]
    initialModel.foreach { model =>
      weightsBuff.appendAll(neh.fromDouble(model.weights))
      treesBuff.appendAll(model.trees)
    }


    // raw scores and checkpointer
    var trainRawBlocks = GBM.initializeRawBlocks[C, B, H](trainWeightBlocks, trainBinVecBlocks,
      treesBuff.toArray, weightsBuff.toArray, boostConf)
      .setName("Initial: Train RawPreds")
    val trainRawBlocksCheckpointer = new Checkpointer[ArrayBlock[H]](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel2)
    if (treesBuff.nonEmpty) {
      trainRawBlocksCheckpointer.update(trainRawBlocks)
    }


    // raw scores and checkpointer for test data
    var testRawBlocksCheckpointer: Checkpointer[ArrayBlock[H]] = null
    var testRawBlocks = testBlocks.map { case (testWeightBlocks, _, testBinVecBlocks) =>
      testRawBlocksCheckpointer = new Checkpointer[ArrayBlock[H]](sc,
        boostConf.getCheckpointInterval, boostConf.getStorageLevel3)

      val newTestRawBlocks = GBM.initializeRawBlocks[C, B, H](testWeightBlocks, testBinVecBlocks,
        treesBuff.toArray, weightsBuff.toArray, boostConf)
        .setName("Initial: Test RawPreds")
      if (treesBuff.nonEmpty) {
        testRawBlocksCheckpointer.update(newTestRawBlocks)
      }
      newTestRawBlocks
    }


    // metrics history recoder
    val trainMetricsHistory = mutable.ArrayBuffer.empty[Map[String, Double]]
    val testMetricsHistory = mutable.ArrayBuffer.empty[Map[String, Double]]


    // random number generator for drop out
    val dartRng = new Random(boostConf.getSeed)
    val dropped = mutable.Set.empty[Int]

    var iteration = 0
    var finished = false

    while (!finished && iteration < boostConf.getMaxIter) {
      val numTrees = treesBuff.length
      val logPrefix = s"Iteration $iteration:"

      // drop out
      if (boostConf.getBoostType == GBM.Dart) {
        GBM.dropTrees(dropped, boostConf, numTrees, dartRng)
        if (dropped.nonEmpty) {
          logInfo(s"$logPrefix ${dropped.size} trees dropped")
        } else {
          logInfo(s"$logPrefix skip drop")
        }
      }


      // build trees
      logInfo(s"$logPrefix start")
      val start = System.nanoTime
      val trees = buildTrees[C, B, H](trainWeightBlocks, trainLabelBlocks, trainBinVecBlocks, trainRawBlocks, subBinVecBlocks,
        weightsBuff.toArray, boostConf, iteration, dropped.toSet)
      logInfo(s"$logPrefix finished, duration: ${(System.nanoTime - start) / 1e9} sec")

      if (trees.forall(_.isEmpty)) {
        // fail to build a new tree
        logInfo(s"$logPrefix no more tree built, GBM training finished")
        finished = true

      } else {
        // update base model buffer
        GBM.updateTreeBuffer(weightsBuff, treesBuff, trees, dropped.toSet, boostConf)

        // whether to keep the weights of previous trees
        val keepWeights = boostConf.getBoostType != GBM.Dart || dropped.isEmpty

        // update train data predictions
        trainRawBlocks = GBM.updateRawBlocks[C, B, H](trainBinVecBlocks, trainRawBlocks,
          trees, weightsBuff.toArray, boostConf, keepWeights)
          .setName(s"Iter $iteration: Train RawPreds")
        trainRawBlocksCheckpointer.update(trainRawBlocks)


        if (boostConf.getEvalFunc.isEmpty) {
          // materialize predictions
          trainRawBlocks.count()
        }

        // evaluate on train data
        if (boostConf.getEvalFunc.nonEmpty) {
          val trainMetrics = GBM.evaluate[H, C, B](trainWeightBlocks, trainLabelBlocks, trainRawBlocks, boostConf)
          trainMetricsHistory.append(trainMetrics)
          logInfo(s"$logPrefix train metrics ${trainMetrics.mkString("(", ", ", ")")}")
        }


        testBlocks.foreach { case (testWeightBlocks, testLabelBlocks, testBinVecBlocks) =>
          // update test data predictions
          val newTestRawBlocks = GBM.updateRawBlocks[C, B, H](testBinVecBlocks, testRawBlocks.get,
            trees, weightsBuff.toArray, boostConf, keepWeights)
            .setName(s"Iter $iteration: Test RawPreds")

          testRawBlocks = Some(newTestRawBlocks)
          testRawBlocksCheckpointer.update(testRawBlocks.get)

          // evaluate on test data
          val testMetrics = GBM.evaluate[H, C, B](testWeightBlocks, testLabelBlocks, testRawBlocks.get, boostConf)
          testMetricsHistory.append(testMetrics)
          logInfo(s"$logPrefix test metrics ${testMetrics.mkString("(", ", ", ")")}")
        }


        // callback
        if (boostConf.getCallbackFunc.nonEmpty) {
          // using cloning to avoid model modification
          val snapshot = new GBMModel(boostConf.getObjFunc, discretizer.copy(),
            boostConf.getRawBaseScore.clone(), treesBuff.toArray.clone(),
            neh.toDouble(weightsBuff.toArray).clone())

          // callback can update boosting configuration
          boostConf.getCallbackFunc.foreach { callback =>
            if (callback.compute(boostConf, snapshot, iteration + 1,
              trainMetricsHistory.toArray.clone(), testMetricsHistory.toArray.clone())) {
              finished = true
              logInfo(s"$logPrefix callback ${callback.name} stop training")
            }
          }
        }
      }

      logInfo(s"$logPrefix finished, ${treesBuff.length} trees now")
      iteration += 1
    }

    if (iteration >= boostConf.getMaxIter) {
      logInfo(s"maxIter=${boostConf.getMaxIter} reached, GBM training finished")
    }

    subBinVecBlocks.unpersist(false)
    trainRawBlocksCheckpointer.clear(false)
    if (testRawBlocksCheckpointer != null) {
      testRawBlocksCheckpointer.clear(false)
    }

    new GBMModel(boostConf.getObjFunc, discretizer, boostConf.getRawBaseScore,
      treesBuff.toArray, neh.toDouble(weightsBuff.toArray))
  }


  def divideBinVecBlocks[C, B](binVecBlocks: RDD[KVMatrix[C, B]],
                               boostConf: BoostConfig)
                              (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                               cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B]): RDD[KVMatrix[C, B]] = {
    logInfo(s"Dividing train data into ${boostConf.getNumBaseVParts} X ${boostConf.getNumVLayers} parts")

    val numVParts = boostConf.getNumVParts

    binVecBlocks.mapPartitionsWithIndex { case (partId, iter) =>
      val numVLayers = boostConf.getNumVLayers
      val colIdsPerBaseVPart = boostConf.getBaseVCols[C]()
      val numBaseVParts = boostConf.getNumBaseVParts
      var blockId = boostConf.getBlockOffset(partId) - 1

      iter.flatMap { binVecBlock =>
        blockId += 1
        val layerId = (blockId % numVLayers).toInt
        val binVecs = binVecBlock.iterator.toArray

        colIdsPerBaseVPart.iterator.zipWithIndex
          .map { case (colIds, baseVPartId) =>
            val vPartId = baseVPartId + layerId * numBaseVParts
            val subBinVecBlock = KVMatrix.sliceAndBuild(colIds, binVecs)
            require(subBinVecBlock.size == binVecBlock.size)
            ((blockId, vPartId), subBinVecBlock)
          }
      }
    }.repartitionAndSortWithinPartitions(new Partitioner {

      override def numPartitions: Int = numVParts

      override def getPartition(key: Any): Int = key match {
        case (_, vPartId: Int) => vPartId
      }

    }).map(_._2)
  }


  def gatherByLayer[T](blocks: RDD[T],
                       blockIds: RDD[Long],
                       boostConf: BoostConfig,
                       bcBoostConf: Broadcast[BoostConfig])
                      (implicit ct: ClassTag[T]): RDD[T] = {
    val baseVPartIds = Array.range(0, boostConf.getNumBaseVParts)
    gatherByLayer(blocks, blockIds, baseVPartIds, boostConf, bcBoostConf)
  }


  def gatherByLayer[T](blocks: RDD[T],
                       blockIds: RDD[Long],
                       baseVPartIds: Array[Int],
                       boostConf: BoostConfig,
                       bcBoostConf: Broadcast[BoostConfig])
                      (implicit ct: ClassTag[T]): RDD[T] = {
    require(baseVPartIds.nonEmpty)

    val numVParts = boostConf.getNumVParts

    val copiedBlocks = if (blockIds != null) {
      blocks.zipPartitions(blockIds)({
        case (blockIter, blockIdIter) =>
          val numVLayers = bcBoostConf.value.getNumVLayers
          val numBaseVParts = bcBoostConf.value.getNumBaseVParts

          Utils.zip2(blockIter, blockIdIter)
            .flatMap { case (block, blockId) =>
              val layerId = (blockId % numVLayers).toInt
              baseVPartIds.map { baseVPartId =>
                val vPartId = baseVPartId + layerId * numBaseVParts
                ((blockId, vPartId), block)
              }
            }
      })

    } else {

      blocks.mapPartitionsWithIndex { case (partId, iter) =>
        val numVLayers = bcBoostConf.value.getNumVLayers
        val numBaseVParts = bcBoostConf.value.getNumBaseVParts
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { block =>
          blockId += 1
          val layerId = (blockId % numVLayers).toInt
          baseVPartIds.map { baseVPartId =>
            val vPartId = baseVPartId + layerId * numBaseVParts
            ((blockId, vPartId), block)
          }
        }
      }
    }


    copiedBlocks
      .repartitionAndSortWithinPartitions(new Partitioner {

        override def numPartitions: Int = numVParts

        override def getPartition(key: Any): Int = key match {
          case (_, vPartId: Int) => vPartId
        }

      }).map(_._2)
  }


  def buildTrees[C, B, H](weightBlocks: RDD[CompactArray[H]],
                          labelBlocks: RDD[ArrayBlock[H]],
                          binVecBlocks: RDD[KVMatrix[C, B]],
                          rawBlocks: RDD[ArrayBlock[H]],
                          subBinVecBlocks: RDD[KVMatrix[C, B]],
                          weights: Array[H],
                          boostConf: BoostConfig,
                          iteration: Int,
                          dropped: Set[Int])
                         (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                          cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                          ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Array[TreeModel] = {
    import Utils._

    val numTrees = boostConf.getBaseModelParallelism * boostConf.getRawSize
    logInfo(s"Iteration $iteration: Starting to create next $numTrees trees")

    val treeIdType = Utils.getTypeByRange(numTrees)
    logInfo(s"DataType of TreeId: $treeIdType")

    val nodeIdType = Utils.getTypeByRange(1 << boostConf.getMaxDepth)
    logInfo(s"DataType of NodeId: $nodeIdType")

    (treeIdType, nodeIdType) match {
      case (BYTE, BYTE) =>
        buildTreesImpl[Byte, Byte, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, weights, boostConf, iteration, dropped)

      case (BYTE, SHORT) =>
        buildTreesImpl[Byte, Short, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, weights, boostConf, iteration, dropped)

      case (BYTE, INT) =>
        buildTreesImpl[Byte, Int, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, weights, boostConf, iteration, dropped)

      case (SHORT, BYTE) =>
        buildTreesImpl[Short, Byte, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, weights, boostConf, iteration, dropped)

      case (SHORT, SHORT) =>
        buildTreesImpl[Short, Short, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, weights, boostConf, iteration, dropped)

      case (SHORT, INT) =>
        buildTreesImpl[Short, Int, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, weights, boostConf, iteration, dropped)

      case (INT, BYTE) =>
        buildTreesImpl[Int, Byte, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, weights, boostConf, iteration, dropped)

      case (INT, SHORT) =>
        buildTreesImpl[Int, Short, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, weights, boostConf, iteration, dropped)

      case (INT, INT) =>
        buildTreesImpl[Int, Int, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, weights, boostConf, iteration, dropped)
    }
  }


  /**
    * build new trees
    *
    * @param rawBlocks previous raw predictions
    * @param weights   weights of trees
    * @param boostConf boosting configuration
    * @param iteration current iteration
    * @param dropped   indices of trees which are selected to drop during building of current tree
    * @return new trees
    */
  def buildTreesImpl[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                    labelBlocks: RDD[ArrayBlock[H]],
                                    binVecBlocks: RDD[KVMatrix[C, B]],
                                    rawBlocks: RDD[ArrayBlock[H]],
                                    subBinVecBlocks: RDD[KVMatrix[C, B]],
                                    weights: Array[H],
                                    boostConf: BoostConfig,
                                    iteration: Int,
                                    dropped: Set[Int])
                                   (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                    cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                    cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                    cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                    ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Array[TreeModel] = {
    import nuh._

    val sc = weightBlocks.sparkContext

    val recoder = new ResourceRecoder

    val bcBoostConf = sc.broadcast(boostConf)
    recoder.append(bcBoostConf)

    val rawSize = boostConf.getRawSize

    val computeRaw = boostConf.getBoostType match {
      case GBM.GBTree =>
        rawSeq: Array[H] => rawSeq

      case GBM.Dart if dropped.isEmpty =>
        rawSeq: Array[H] => neh.take(rawSeq, rawSize)

      case GBM.Dart if dropped.nonEmpty =>
        val rawBase = neh.fromDouble(boostConf.getRawBaseScore)

        rawSeq: Array[H] =>
          val raw = rawBase.clone()
          Iterator.range(rawSize, rawSeq.length)
            .filterNot(i => dropped.contains(i - rawSize))
            .foreach { i => raw(i % rawSize) += rawSeq(i) * weights(i - rawSize) }
          raw
    }

    val computeGrad = (weight: H, label: Array[H], rawSeq: Array[H]) => {
      val raw = neh.toDouble(computeRaw(rawSeq))
      val score = boostConf.getObjFunc.transform(raw)
      val (grad, hess) = boostConf.getObjFunc.compute(neh.toDouble(label), score)
      require(grad.length == rawSize && hess.length == rawSize)

      val array = Array.ofDim[H](rawSize << 1)
      var i = 0
      while (i < rawSize) {
        val j = i << 1
        array(j) = neh.fromDouble(grad(i)) * weight
        array(j + 1) = neh.fromDouble(hess(i)) * weight
        i += 1
      }
      array
    }

    val computeGradBlock = (weightBlock: CompactArray[H], labelBlock: ArrayBlock[H], rawBlock: ArrayBlock[H]) => {
      require(weightBlock.size == rawBlock.size)
      require(labelBlock.size == rawBlock.size)

      val iter = Utils.zip3(weightBlock.iterator, labelBlock.iterator, rawBlock.iterator)
        .map { case (weight, label, rawSeq) => computeGrad(weight, label, rawSeq) }

      ArrayBlock.build[H](iter)
    }


    val baseConfig = BaseConfig.create(boostConf, iteration, boostConf.getSeed + iteration)
    logInfo(s"Iter $iteration: ColSelector ${baseConfig.colSelector}")

    // To alleviate memory footprint in caching layer, different schemas of intermediate dataset are designed.
    // Each `prepareTreeInput**` method will internally cache necessary datasets in a compact fashion.
    // These cached datasets are holden in `recoder`, and will be freed after training.
    val (sampledBinVecBlocks, treeIdBlocks, sampledSubBinVecBlocks, sampledBlockIds, agTreeIdBlocks, agGradBlocks) = (boostConf.getSubSampleType, boostConf.getSubSampleRate == 1) match {
      case (_, true) =>
        adaptTreeInputsForNonSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, boostConf, bcBoostConf, iteration, computeGradBlock, recoder)

      case (GBM.Block, _) =>
        adaptTreeInputsForBlockSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, boostConf, bcBoostConf, iteration, computeGradBlock, recoder)

      case (GBM.Row, _) =>
        adaptTreeInputsForInstanceSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, subBinVecBlocks, boostConf, bcBoostConf, iteration, computeGrad, recoder)
    }

    val trees = Tree.trainVertical[T, N, C, B, H](sampledBinVecBlocks, treeIdBlocks, sampledSubBinVecBlocks, sampledBlockIds, agTreeIdBlocks, agGradBlocks, boostConf, bcBoostConf, baseConfig)

    recoder.clear(false)

    System.gc()

    trees
  }


  def adaptTreeInputsForNonSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                   labelBlocks: RDD[ArrayBlock[H]],
                                                   binVecBlocks: RDD[KVMatrix[C, B]],
                                                   rawBlocks: RDD[ArrayBlock[H]],
                                                   subBinVecBlocks: RDD[KVMatrix[C, B]],
                                                   boostConf: BoostConfig,
                                                   bcBoostConf: Broadcast[BoostConfig],
                                                   iteration: Int,
                                                   computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                                   recoder: ResourceRecoder)
                                                  (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                   cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                   cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                   cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                   ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]) = {
    import RDDFunctions._

    val sc = weightBlocks.sparkContext


    val treeIdBlocks = weightBlocks
      .mapPartitions { iter =>
        val blockSize = bcBoostConf.value.getBlockSize
        val numTrees = bcBoostConf.value.getNumTrees
        val treeIds = Array.tabulate(numTrees)(int.fromInt)
        val defaultTreeIdBlock = ArrayBlock.fill(blockSize, treeIds)
        iter.map { weightBlock =>
          if (weightBlock.size == blockSize) {
            defaultTreeIdBlock
          } else {
            ArrayBlock.fill(weightBlock.size, treeIds)
          }
        }
      }.setName(s"Iter $iteration: TreeIds")

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(treeIdBlocks)


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .map { case (weightBlock, labelBlock, rawBlock) =>
        computeGradBlock(weightBlock, labelBlock, rawBlock)
      }.setName(s"Iter $iteration: Grads")

    val agGradBlocks = gatherByLayer(gradBlocks, null, boostConf, bcBoostConf)
      .setName(s"Iter $iteration: Grads (Gathered)")

    agGradBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agGradBlocks)


    val agTreeIdBlocks = agGradBlocks
      .mapPartitions { iter =>
        val blockSize = bcBoostConf.value.getBlockSize
        val numTrees = bcBoostConf.value.getNumTrees
        val treeIds = Array.tabulate(numTrees)(int.fromInt)
        val defaultTreeIdBlock = ArrayBlock.fill(blockSize, treeIds)
        iter.map { gradBlock =>
          if (gradBlock.size == blockSize) {
            defaultTreeIdBlock
          } else {
            ArrayBlock.fill(gradBlock.size, treeIds)
          }
        }
      }.setName(s"Iter $iteration: TreeIds (Gathered)")

    agTreeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agTreeIdBlocks)

    (binVecBlocks, treeIdBlocks, subBinVecBlocks, null, agTreeIdBlocks, agGradBlocks)
  }


  def adaptTreeInputsForBlockSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                     labelBlocks: RDD[ArrayBlock[H]],
                                                     binVecBlocks: RDD[KVMatrix[C, B]],
                                                     rawBlocks: RDD[ArrayBlock[H]],
                                                     subBinVecBlocks: RDD[KVMatrix[C, B]],
                                                     boostConf: BoostConfig,
                                                     bcBoostConf: Broadcast[BoostConfig],
                                                     iteration: Int,
                                                     computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                                     recoder: ResourceRecoder)
                                                    (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                     cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                     cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                     cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                     ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]) = {
    import RDDFunctions._

    val sc = weightBlocks.sparkContext

    val blockSelector = Selector.create(boostConf.getSubSampleRate, boostConf.getNumBlocks,
      boostConf.getBaseModelParallelism, 1, boostConf.getSeed + iteration)
    logInfo(s"Iter $iteration: BlockSelector $blockSelector")

    val bcBlockSelector = sc.broadcast(blockSelector)
    recoder.append(bcBlockSelector)


    val sampledBinVecBlocks = binVecBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSelector = bcBlockSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { binVecBlock =>
          blockId += 1
          if (blockSelector.contains[Long](blockId)) {
            Iterator.single(binVecBlock)
          } else {
            Iterator.empty
          }
        }
      }.setName(s"Iter $iteration: BinVecs (Block-Sampled)")

    sampledBinVecBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(sampledBinVecBlocks)


    val treeIdBlocks = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSelector = bcBlockSelector.value
        val computeTreeIds = bcBoostConf.value.computeTreeIds[T]()
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { weightBlock =>
          blockId += 1
          val baseIds = blockSelector.index[T, Long](blockId)
          if (baseIds.nonEmpty) {
            val treeIds = computeTreeIds(baseIds)
            val treeIdBlock = ArrayBlock.fill(weightBlock.size, treeIds)
            Iterator.single(treeIdBlock)
          } else {
            Iterator.empty
          }
        }
      }.setName(s"Iter $iteration: TreeIds (Block-Sampled)")

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(treeIdBlocks)


    val sampledBlockIds = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSelector = bcBlockSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { _ =>
          blockId += 1

          if (blockSelector.contains[Long](blockId)) {
            Iterator.single(blockId)
          } else {
            Iterator.empty
          }
        }
      }.setName(s"Iter $iteration: BlockIds (Block-Sampled)")

    sampledBlockIds.persist(boostConf.getStorageLevel1)
    recoder.append(sampledBlockIds)


    val agTreeIdBlocks = gatherByLayer(treeIdBlocks, sampledBlockIds, boostConf, bcBoostConf)
      .setName(s"Iter $iteration: TreeIds (Block-Sampled) (Gathered)")

    agTreeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agTreeIdBlocks)


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .mapPartitionsWithIndex {
        case (partId, iter) =>
          val blockSelector = bcBlockSelector.value
          var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

          iter.flatMap {
            case (weightBlock, labelBlock, rawBlock) =>
              blockId += 1
              if (blockSelector.contains[Long](blockId)) {
                val gradBlock = computeGradBlock(weightBlock, labelBlock, rawBlock)
                Iterator.single(gradBlock)
              } else {
                Iterator.empty
              }
          }
      }.setName(s"Iter $iteration: Grads (Block-Sampled)")

    val agGradBlocks = gatherByLayer(gradBlocks, sampledBlockIds, boostConf, bcBoostConf)
      .setName(s"Iter $iteration: Grads (Block-Sampled) (Gathered)")

    agGradBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agGradBlocks)


    val sampledSubBinVecBlocks = subBinVecBlocks
      .mapPartitionsWithIndex { case (vPartId, iter) =>
        val numVLayers = bcBoostConf.value.getNumVLayers
        val blockSelector = bcBlockSelector.value
        var blockId = bcBoostConf.value.getVBlockOffset(vPartId) - numVLayers

        iter.flatMap { subBinVecBlock =>
          blockId += numVLayers
          if (blockSelector.contains[Long](blockId)) {
            Iterator.single(subBinVecBlock)
          } else {
            Iterator.empty
          }
        }
      }.setName(s"Iter $iteration: SubBinVecs (Block-Sampled)")

    sampledSubBinVecBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(sampledSubBinVecBlocks)


    (sampledBinVecBlocks, treeIdBlocks, sampledSubBinVecBlocks, sampledBlockIds, agTreeIdBlocks, agGradBlocks)
  }


  def adaptTreeInputsForInstanceSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                        labelBlocks: RDD[ArrayBlock[H]],
                                                        binVecBlocks: RDD[KVMatrix[C, B]],
                                                        rawBlocks: RDD[ArrayBlock[H]],
                                                        subBinVecBlocks: RDD[KVMatrix[C, B]],
                                                        boostConf: BoostConfig,
                                                        bcBoostConf: Broadcast[BoostConfig],
                                                        iteration: Int,
                                                        computeGrad: (H, Array[H], Array[H]) => Array[H],
                                                        recoder: ResourceRecoder)
                                                       (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                        cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                        cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                        cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                        ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]) = {
    import RDDFunctions._

    val sc = weightBlocks.sparkContext

    val rowSelector = Selector.create(boostConf.getSubSampleRate, boostConf.getNumBlocks * boostConf.getBlockSize,
      boostConf.getBaseModelParallelism, 1, boostConf.getSeed + iteration)
    logInfo(s"Iter $iteration: RowSelector $rowSelector")

    val bcRowSelector = sc.broadcast(rowSelector)
    recoder.append(bcRowSelector)

    val sampledBinVecBlocks = binVecBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val rowSelector = bcRowSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.map { binVecBlock =>
          blockId += 1
          var rowId = blockId * blockSize - 1

          val iter2 = binVecBlock.iterator
            .flatMap { binVec =>
              rowId += 1
              if (rowSelector.contains[Long](rowId)) {
                Iterator.single(binVec)
              } else {
                Iterator.empty
              }
            }
          KVMatrix.build[C, B](iter2)
        }
      }.setName(s"Iter $iteration: BinVecs (Row-Sampled)")

    sampledBinVecBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(sampledBinVecBlocks)


    val treeIdBlocks = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val computeTreeIds = bcBoostConf.value.computeTreeIds[T]()
        val rowSelector = bcRowSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.map { weightBlock =>
          blockId += 1
          var rowId = blockId * blockSize - 1

          val iter2 = weightBlock.iterator
            .flatMap { _ =>
              rowId += 1
              val baseIds = rowSelector.index[T, Long](rowId)
              if (baseIds.nonEmpty) {
                val treeIds = computeTreeIds(baseIds)
                Iterator.single(treeIds)
              } else {
                Iterator.empty
              }
            }
          ArrayBlock.build[T](iter2)
        }
      }.setName(s"Iter $iteration: TreeIds (Row-Sampled)")

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(treeIdBlocks)

    val agTreeIdBlocks = gatherByLayer(treeIdBlocks, null, boostConf, bcBoostConf)
      .setName(s"Iter $iteration: TreeIds (Instance-Sampled) (Gathered)")

    agTreeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agTreeIdBlocks)


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val rowSelector = bcRowSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.map { case (weightBlock, labelBlock, rawBlock) =>
          require(weightBlock.size == rawBlock.size)
          require(labelBlock.size == rawBlock.size)
          blockId += 1
          var rowId = blockId * blockSize - 1

          val iter2 = Utils.zip3(weightBlock.iterator, labelBlock.iterator, rawBlock.iterator)
            .flatMap { case (weight, label, rawSeq) =>
              rowId += 1
              if (rowSelector.contains[Long](rowId)) {
                val grad = computeGrad(weight, label, rawSeq)
                Iterator.single(grad)
              } else {
                Iterator.empty
              }
            }
          ArrayBlock.build[H](iter2)
        }
      }.setName(s"Iter $iteration: Grads (Row-Sampled)")

    val agGradBlocks = gatherByLayer(gradBlocks, null, boostConf, bcBoostConf)
      .setName(s"Iter $iteration: Grads (Row-Sampled) (AllGathered)")

    agGradBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agGradBlocks)


    val sampledSubBinVecBlocks = subBinVecBlocks
      .mapPartitionsWithIndex { case (vPartId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val numVLayers = bcBoostConf.value.getNumVLayers
        val rowSelector = bcRowSelector.value
        val localColIds = bcBoostConf.value.getVCols[C](vPartId)
        var blockId = bcBoostConf.value.getVBlockOffset(vPartId) - numVLayers

        iter.map { subBinVecBlock =>
          blockId += numVLayers
          var rowId = blockId * blockSize - 1

          val iter2 = subBinVecBlock.iterator
            .flatMap { subBinVec =>
              rowId += 1
              if (rowSelector.contains[Long](rowId)) {
                Iterator.single(subBinVec)
              } else {
                Iterator.empty
              }
            }
          KVMatrix.sliceAndBuild[C, B](localColIds, iter2)
        }
      }.setName(s"Iter $iteration: SubBinVecs (Row-Sampled)")

    sampledSubBinVecBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(sampledSubBinVecBlocks)

    (sampledBinVecBlocks, treeIdBlocks, sampledSubBinVecBlocks, null, agTreeIdBlocks, agGradBlocks)
  }
}



