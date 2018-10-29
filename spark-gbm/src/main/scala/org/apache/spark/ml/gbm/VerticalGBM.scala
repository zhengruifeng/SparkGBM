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
import org.apache.spark.sql._


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
    import Utils._

    val sc = trainBlocks._1.sparkContext
    val parallelism = boostConf.getRealParallelism(boostConf.getReduceParallelism, sc.defaultParallelism)
    boostConf.updateVColsInfo(parallelism)

    val maxCols = boostConf.getVCols[C]().map(_.length).max

    Utils.getTypeByRange(maxCols * boostConf.getBlockSize) match {
      case BYTE =>
        boostImpl[C, B, H, Byte](trainBlocks, testBlocks, boostConf, discretizer, initialModel)

      case SHORT =>
        boostImpl[C, B, H, Short](trainBlocks, testBlocks, boostConf, discretizer, initialModel)

      case INT =>
        boostImpl[C, B, H, Int](trainBlocks, testBlocks, boostConf, discretizer, initialModel)
    }
  }


  def boostImpl[C, B, H, G](trainBlocks: (RDD[CompactArray[H]], RDD[ArrayBlock[H]], RDD[KVMatrix[C, B]]),
                            testBlocks: Option[(RDD[CompactArray[H]], RDD[ArrayBlock[H]], RDD[KVMatrix[C, B]])],
                            boostConf: BoostConfig,
                            discretizer: Discretizer,
                            initialModel: Option[GBMModel])
                           (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                            cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                            ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H],
                            cg: ClassTag[G], ing: Integral[G], neg: NumericExt[G]): GBMModel = {

    val spark = SparkSession.builder().getOrCreate()
    val sc = spark.sparkContext

    // train blocks
    val (trainWeightBlocks, trainLabelBlocks, trainBinVecBlocks) = trainBlocks
    GBM.touchWeightBlocksAndUpdateSizeInfo[H](trainWeightBlocks, boostConf)

    // test blocks
    testBlocks.foreach { case (testWeightBlocks, _, _) => GBM.touchWeightBlocks(testWeightBlocks, boostConf) }


    // vertical train sub-binvec blocks
    val subBinVecBlocks = divideBinVecBlocks[C, B, G](trainBinVecBlocks, boostConf)
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
      val trees = buildTrees[C, B, H, G](trainWeightBlocks, trainLabelBlocks, trainBinVecBlocks, subBinVecBlocks, trainRawBlocks,
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
            if (callback.compute(spark, boostConf, snapshot, iteration + 1,
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

    subBinVecBlocks.unpersist(true)
    trainRawBlocksCheckpointer.clear(true)
    if (testRawBlocksCheckpointer != null) {
      testRawBlocksCheckpointer.clear(true)
    }

    new GBMModel(boostConf.getObjFunc, discretizer, boostConf.getRawBaseScore,
      treesBuff.toArray, neh.toDouble(weightsBuff.toArray))
  }


  def divideBinVecBlocks[C, B, G](trainBinVecBlocks: RDD[KVMatrix[C, B]],
                                  boostConf: BoostConfig)
                                 (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                  cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                  cg: ClassTag[G], ing: Integral[G], neg: NumericExt[G]): RDD[KVVector[G, B]] = {

    val numVParts = boostConf.getNumVParts
    val colIdsPerVPart = boostConf.getVCols[C]()
    require(numVParts == colIdsPerVPart.length)

    trainBinVecBlocks.mapPartitionsWithIndex { case (partId, iter) =>
      var localBlockId = -1L

      iter.flatMap { binVecBlock =>
        localBlockId += 1

        val binVecs = binVecBlock.iterator.toSeq

        colIdsPerVPart.iterator.zipWithIndex.map { case (colIds, vPartId) =>

          val values = binVecs.iterator.flatMap { binVec =>
            val sliced = binVec.slice(nec.toInt(colIds))
            sliced.iterator.map(_._2)
          }.toArray

          val subBinVecBlock = KVVector.dense[G, B](values).compress
          require(subBinVecBlock.size == colIds.length * binVecs.length)

          ((partId, localBlockId, vPartId), subBinVecBlock)
        }
      }

    }.repartitionAndSortWithinPartitions(new Partitioner {

      override def numPartitions: Int = numVParts

      override def getPartition(key: Any): Int = key match {
        case (_, _, vPartId: Int) => vPartId
      }

    }).map(_._2)
  }


  def buildTrees[C, B, H, G](weightBlocks: RDD[CompactArray[H]],
                             labelBlocks: RDD[ArrayBlock[H]],
                             binVecBlocks: RDD[KVMatrix[C, B]],
                             subBinVecBlocks: RDD[KVVector[G, B]],
                             rawBlocks: RDD[ArrayBlock[H]],
                             weights: Array[H],
                             boostConf: BoostConfig,
                             iteration: Int,
                             dropped: Set[Int])
                            (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                             cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                             ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H],
                             cg: ClassTag[G], ing: Integral[G], neg: NumericExt[G]): Array[TreeModel] = {
    import Utils._

    val numTrees = boostConf.getBaseModelParallelism * boostConf.getRawSize
    logInfo(s"Iteration $iteration: Starting to create next $numTrees trees")

    val treeIdType = Utils.getTypeByRange(numTrees)
    logInfo(s"DataType of TreeId: $treeIdType")

    val nodeIdType = Utils.getTypeByRange(1 << boostConf.getMaxDepth)
    logInfo(s"DataType of NodeId: $nodeIdType")

    (treeIdType, nodeIdType) match {
      case (BYTE, BYTE) =>
        buildTreesImpl[Byte, Byte, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (BYTE, SHORT) =>
        buildTreesImpl[Byte, Short, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (BYTE, INT) =>
        buildTreesImpl[Byte, Int, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (SHORT, BYTE) =>
        buildTreesImpl[Short, Byte, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (SHORT, SHORT) =>
        buildTreesImpl[Short, Short, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (SHORT, INT) =>
        buildTreesImpl[Short, Int, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (INT, BYTE) =>
        buildTreesImpl[Int, Byte, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (INT, SHORT) =>
        buildTreesImpl[Int, Short, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (INT, INT) =>
        buildTreesImpl[Int, Int, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)
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
  def buildTreesImpl[T, N, C, B, H, G](weightBlocks: RDD[CompactArray[H]],
                                       labelBlocks: RDD[ArrayBlock[H]],
                                       binVecBlocks: RDD[KVMatrix[C, B]],
                                       subBinVecBlocks: RDD[KVVector[G, B]],
                                       rawBlocks: RDD[ArrayBlock[H]],
                                       weights: Array[H],
                                       boostConf: BoostConfig,
                                       iteration: Int,
                                       dropped: Set[Int])
                                      (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                       cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                       cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                       cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                       ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H],
                                       cg: ClassTag[G], ing: Integral[G], neg: NumericExt[G]): Array[TreeModel] = {
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
        rawSeq: Array[H] => rawSeq.take(rawSize)

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


    val baseConfig = BaseConfig.create(boostConf, iteration, boostConf.getBaseModelParallelism, boostConf.getSeed + iteration)

    // To alleviate memory footprint in caching layer, different schemas of intermediate dataset are designed.
    // Each `prepareTreeInput**` method will internally cache necessary datasets in a compact fashion.
    // These cached datasets are holden in `recoder`, and will be freed after training.
    val (sampledBinVecBlocks, treeIdBlocks, sampledSubBinVecBlocks, agTreeIdBlocks, agGradBlocks) = (boostConf.getSubSampleType, boostConf.getSubSampleRate == 1) match {
      case (_, true) =>
        adaptTreeInputsForNonSampling[T, N, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, boostConf, bcBoostConf, iteration, computeGradBlock, recoder)

      case (GBM.Block, _) =>
        adaptTreeInputsForBlockSampling[T, N, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, boostConf, bcBoostConf, iteration, computeGradBlock, recoder)

      case (GBM.Instance, _) =>
        adaptTreeInputsForInstanceSampling[T, N, C, B, H, G](weightBlocks, labelBlocks, binVecBlocks, subBinVecBlocks, rawBlocks, boostConf, bcBoostConf, iteration, computeGrad, recoder)
    }

    val trees = Tree.trainVertical[T, N, C, B, H, G](sampledBinVecBlocks, treeIdBlocks, sampledSubBinVecBlocks, agTreeIdBlocks, agGradBlocks, boostConf, bcBoostConf, baseConfig)

    recoder.clear(true)

    trees
  }


  def adaptTreeInputsForNonSampling[T, N, C, B, H, G](weightBlocks: RDD[CompactArray[H]],
                                                      labelBlocks: RDD[ArrayBlock[H]],
                                                      binVecBlocks: RDD[KVMatrix[C, B]],
                                                      subBinVecBlocks: RDD[KVVector[G, B]],
                                                      rawBlocks: RDD[ArrayBlock[H]],
                                                      boostConf: BoostConfig,
                                                      bcBoostConf: Broadcast[BoostConfig],
                                                      iteration: Int,
                                                      computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                                      recoder: ResourceRecoder)
                                                     (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                      cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                      cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                      cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                      ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H],
                                                      cg: ClassTag[G], ing: Integral[G], neg: NumericExt[G]) = {
    import RDDFunctions._

    val numVParts = boostConf.getNumVParts
    val rawSize = boostConf.getRawSize
    val numBaseModels = boostConf.getBaseModelParallelism
    val numTrees = numBaseModels * rawSize
    val blockSize = boostConf.getBlockSize

    val treeIdBlocks = weightBlocks
      .mapPartitions { iter =>
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

    val agGradBlocks = gradBlocks
      .allgather(numVParts)
      .setName(s"Iter $iteration: Grads (AllGathered)")

    agGradBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agGradBlocks)


    val agTreeIdBlocks = agGradBlocks.mapPartitions { iter =>
      val treeIds = Array.tabulate(numTrees)(int.fromInt)
      val defaultTreeIdBlock = ArrayBlock.fill(blockSize, treeIds)
      iter.map { gradBlock =>
        if (gradBlock.size == blockSize) {
          defaultTreeIdBlock
        } else {
          ArrayBlock.fill(gradBlock.size, treeIds)
        }
      }
    }.setName(s"Iter $iteration: TreeIds (AllGathered)")

    agTreeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agTreeIdBlocks)

    (binVecBlocks, treeIdBlocks, subBinVecBlocks, agTreeIdBlocks, agGradBlocks)
  }


  def adaptTreeInputsForBlockSampling[T, N, C, B, H, G](weightBlocks: RDD[CompactArray[H]],
                                                        labelBlocks: RDD[ArrayBlock[H]],
                                                        binVecBlocks: RDD[KVMatrix[C, B]],
                                                        subBinVecBlocks: RDD[KVVector[G, B]],
                                                        rawBlocks: RDD[ArrayBlock[H]],
                                                        boostConf: BoostConfig,
                                                        bcBoostConf: Broadcast[BoostConfig],
                                                        iteration: Int,
                                                        computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                                        recoder: ResourceRecoder)
                                                       (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                        cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                        cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                        cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                        ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H],
                                                        cg: ClassTag[G], ing: Integral[G], neg: NumericExt[G]) = {
    import RDDFunctions._

    val blockSelector = Selector.create(boostConf.getSubSampleRate, boostConf.getNumBlocks,
      boostConf.getBaseModelParallelism, 1, boostConf.getSeed + iteration)
    logInfo(s"Iteration $iteration, blockSelector $blockSelector")


    val sampledBinVecBlocks = binVecBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val numBaseModels = bcBoostConf.value.getBaseModelParallelism
        var blockId = bcBoostConf.value.getBlockIdOffsetPerPart(partId) - 1

        iter.flatMap { binVecBlock =>
          blockId += 1

          val selected = Iterator.range(0, numBaseModels)
            .exists { i => blockSelector.contains(i, blockId) }

          if (selected) {
            Iterator.single(binVecBlock)
          } else {
            Iterator.empty
          }
        }
      }.setName(s"Iter $iteration: BinVecs (Block-Sampled)")

    if (boostConf.getStorageStrategy == GBM.Eager) {
      sampledBinVecBlocks.persist(boostConf.getStorageLevel1)
      recoder.append(sampledBinVecBlocks)
    }


    val treeIdBlocks = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val numBaseModels = bcBoostConf.value.getBaseModelParallelism
        val computeTreeIds = bcBoostConf.value.computeTreeIds[T]()

        var blockId = bcBoostConf.value.getBlockIdOffsetPerPart(partId) - 1

        iter.flatMap { weightBlock =>
          blockId += 1

          val baseIds = Array.range(0, numBaseModels)
            .filter { i => blockSelector.contains(i, blockId) }

          if (baseIds.nonEmpty) {
            val treeIds = computeTreeIds(net.fromInt(baseIds))
            val treeIdBlock = ArrayBlock.fill(weightBlock.size, treeIds)
            Iterator.single(treeIdBlock)
          } else {
            Iterator.empty
          }
        }
      }.setName(s"Iter $iteration: TreeIds (Block-Sampled)")

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(treeIdBlocks)


    val agTreeIdBlocks = treeIdBlocks
      .allgather(boostConf.getNumVParts)
      .setName(s"Iter $iteration: TreeIds (Block-Sampled) (AllGathered)")

    agTreeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agTreeIdBlocks)


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val numBaseModels = bcBoostConf.value.getBaseModelParallelism
        var blockId = bcBoostConf.value.getBlockIdOffsetPerPart(partId) - 1

        iter.flatMap { case (weightBlock, labelBlock, rawBlock) =>
          blockId += 1

          val selected = Iterator.range(0, numBaseModels)
            .exists { i => blockSelector.contains(i, blockId) }

          if (selected) {
            val gradBlock = computeGradBlock(weightBlock, labelBlock, rawBlock)
            Iterator.single(gradBlock)
          } else {
            Iterator.empty
          }
        }
      }.setName(s"Iter $iteration: Grads (Block-Sampled)")

    val agGradBlocks = gradBlocks
      .allgather(boostConf.getNumVParts)
      .setName(s"Iter $iteration: Grads (Block-Sampled) (AllGathered)")

    agGradBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agGradBlocks)


    val sampledSubBinVecBlocks = subBinVecBlocks.mapPartitions { iter =>
      val numBaseModels = bcBoostConf.value.getBaseModelParallelism
      var blockId = -1L

      iter.flatMap { subBinVecBlock =>
        blockId += 1

        val selected = Iterator.range(0, numBaseModels)
          .exists { i => blockSelector.contains(i, blockId) }

        if (selected) {
          Iterator.single(subBinVecBlock)
        } else {
          Iterator.empty
        }
      }
    }.setName(s"Iter $iteration: SubBinVecs (Block-Sampled)")

    if (boostConf.getStorageStrategy == GBM.Eager) {
      sampledSubBinVecBlocks.persist(boostConf.getStorageLevel1)
      recoder.append(sampledSubBinVecBlocks)
    }


    (sampledBinVecBlocks, treeIdBlocks, sampledSubBinVecBlocks, agTreeIdBlocks, agGradBlocks)
  }


  def adaptTreeInputsForInstanceSampling[T, N, C, B, H, G](weightBlocks: RDD[CompactArray[H]],
                                                           labelBlocks: RDD[ArrayBlock[H]],
                                                           binVecBlocks: RDD[KVMatrix[C, B]],
                                                           subBinVecBlocks: RDD[KVVector[G, B]],
                                                           rawBlocks: RDD[ArrayBlock[H]],
                                                           boostConf: BoostConfig,
                                                           bcBoostConf: Broadcast[BoostConfig],
                                                           iteration: Int,
                                                           computeGrad: (H, Array[H], Array[H]) => Array[H],
                                                           recoder: ResourceRecoder)
                                                          (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                           cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                           cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                           cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                           ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H],
                                                           cg: ClassTag[G], ing: Integral[G], neg: NumericExt[G]) = {
    import RDDFunctions._

    val instanceSelector = Selector.create(boostConf.getSubSampleRate, boostConf.getNumInstances,
      boostConf.getBaseModelParallelism, 1, boostConf.getSeed + iteration)
    logInfo(s"Iteration $iteration, instanceSelector $instanceSelector")


    val sampledBinVecBlocks = binVecBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val numBaseModels = bcBoostConf.value.getBaseModelParallelism
        val blockSize = bcBoostConf.value.getBlockSize
        var instanceId = bcBoostConf.value.getInstanceOffsetPerPart(partId) - 1

        iter.flatMap(_.iterator)
          .flatMap { binVec =>
            instanceId += 1

            val selected = Iterator.range(0, numBaseModels)
              .exists { i => instanceSelector.contains(i, instanceId) }

            if (selected) {
              Iterator.single(binVec)
            } else {
              Iterator.empty
            }
          }.grouped(blockSize)
          .map(KVMatrix.build[C, B])
      }.setName(s"Iter $iteration: BinVecs (Instance-Sampled)")

    sampledBinVecBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(sampledBinVecBlocks)


    val treeIdBlocks = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val numBaseModels = bcBoostConf.value.getBaseModelParallelism
        val blockSize = bcBoostConf.value.getBlockSize
        val computeTreeIds = bcBoostConf.value.computeTreeIds[T]()

        var instanceId = bcBoostConf.value.getInstanceOffsetPerPart(partId) - 1

        iter.flatMap(_.iterator)
          .flatMap { _ =>
            instanceId += 1

            val baseIds = Array.range(0, numBaseModels)
              .filter { i => instanceSelector.contains(i, instanceId) }

            if (baseIds.nonEmpty) {
              val treeIds = computeTreeIds(net.fromInt(baseIds))
              Iterator.single(treeIds)
            } else {
              Iterator.empty
            }
          }.grouped(blockSize)
          .map(ArrayBlock.build[T])
      }.setName(s"Iter $iteration: TreeIds (Instance-Sampled)")

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(treeIdBlocks)


    val agTreeIdBlocks = treeIdBlocks
      .allgather(boostConf.getNumVParts)
      .setName(s"Iter $iteration: TreeIds (Instance-Sampled) (AllGathered)")

    agTreeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agTreeIdBlocks)


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val numBaseModels = bcBoostConf.value.getBaseModelParallelism
        val blockSize = bcBoostConf.value.getBlockSize

        var instanceId = bcBoostConf.value.getInstanceOffsetPerPart(partId) - 1

        iter.flatMap { case (weightBlock, labelBlock, rawBlock) =>
          require(weightBlock.size == rawBlock.size)
          require(labelBlock.size == rawBlock.size)

          Utils.zip3(weightBlock.iterator, labelBlock.iterator, rawBlock.iterator)

        }.flatMap { case (weight, label, rawSeq) =>
          instanceId += 1

          val selected = Iterator.range(0, numBaseModels)
            .exists { i => instanceSelector.contains(i, instanceId) }

          if (selected) {
            val grad = computeGrad(weight, label, rawSeq)
            Iterator.single(grad)
          } else {
            Iterator.empty
          }
        }.grouped(blockSize)
          .map(ArrayBlock.build[H])

      }.setName(s"Iter $iteration: Grads (Instance-Sampled)")

    val agGradBlocks = gradBlocks
      .allgather(boostConf.getNumVParts)
      .setName(s"Iter $iteration: Grads (Instance-Sampled) (AllGathered)")

    agGradBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(agGradBlocks)


    val sampledSubBinVecBlocks = subBinVecBlocks
      .mapPartitionsWithIndex { case (vPartId, iter) =>
        val numBaseModels = bcBoostConf.value.getBaseModelParallelism
        val blockSize = bcBoostConf.value.getBlockSize
        val localColIds = bcBoostConf.value.getVCols[C](vPartId)
        val numLocalCols = localColIds.length

        var instanceId = -1L

        iter.flatMap { subBinVecBlock =>
          require(subBinVecBlock.size % numLocalCols == 0)

          subBinVecBlock.iterator
            .map(_._2).grouped(numLocalCols)

        }.flatMap { values =>
          instanceId += 1

          val selected = Iterator.range(0, numBaseModels)
            .exists { i => instanceSelector.contains(i, instanceId) }

          if (selected) {
            Iterator.single(values)
          } else {
            Iterator.empty
          }

        }.grouped(blockSize)
          .map { seqs => KVVector.dense[G, B](seqs.flatten.toArray).compress }
      }.setName(s"Iter $iteration: SubBinVecs (Instance-Sampled)")

    sampledSubBinVecBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(sampledSubBinVecBlocks)

    (sampledBinVecBlocks, treeIdBlocks, sampledSubBinVecBlocks, agTreeIdBlocks, agGradBlocks)
  }
}



