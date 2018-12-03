package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.rdd._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.util.QuantileSummaries


object HorizontalGBM extends Logging {

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

    // train blocks
    val (trainWeightBlocks, trainLabelBlocks, trainBinVecBlocks) = trainBlocks
    GBM.touchBlocksAndUpdatePartInfo[H](trainWeightBlocks, boostConf, true)

    // test blocks
    testBlocks.foreach { case (testWeightBlocks, _, _) =>
      GBM.touchBlocksAndUpdatePartInfo(testWeightBlocks, boostConf, false)
    }


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
      val logPrefix = s"Iter $iteration:"

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
      val tic = System.nanoTime()
      val trees = buildTrees[C, B, H](trainWeightBlocks, trainLabelBlocks, trainBinVecBlocks, trainRawBlocks,
        weightsBuff.toArray, boostConf, iteration, dropped.toSet)
      logInfo(s"$logPrefix finished, duration: ${(System.nanoTime() - tic) / 1e9} sec")

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

    trainRawBlocksCheckpointer.clear(false)
    if (testRawBlocksCheckpointer != null) {
      testRawBlocksCheckpointer.clear(false)
    }

    new GBMModel(boostConf.getObjFunc, discretizer, boostConf.getRawBaseScore,
      treesBuff.toArray, neh.toDouble(weightsBuff.toArray))
  }


  def buildTrees[C, B, H](weightBlocks: RDD[CompactArray[H]],
                          labelBlocks: RDD[ArrayBlock[H]],
                          binVecBlocks: RDD[KVMatrix[C, B]],
                          rawBlocks: RDD[ArrayBlock[H]],
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
        buildTreesImpl[Byte, Byte, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (BYTE, SHORT) =>
        buildTreesImpl[Byte, Short, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (BYTE, INT) =>
        buildTreesImpl[Byte, Int, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (SHORT, BYTE) =>
        buildTreesImpl[Short, Byte, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (SHORT, SHORT) =>
        buildTreesImpl[Short, Short, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (SHORT, INT) =>
        buildTreesImpl[Short, Int, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (INT, BYTE) =>
        buildTreesImpl[Int, Byte, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (INT, SHORT) =>
        buildTreesImpl[Int, Short, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case (INT, SHORT) =>
        buildTreesImpl[Int, Int, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)
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

    val cleaner = new ResourceCleaner

    val bcBoostConf = sc.broadcast(boostConf)
    cleaner.registerBroadcastedObjects(bcBoostConf)

    val rawSize = boostConf.getRawSize
    val objFunc = Utils.deepCopy(boostConf.getObjFunc)

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
      val score = objFunc.transform(raw)
      val (grad, hess) = objFunc.compute(neh.toDouble(label), score)
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
    val (sampledBinVecBlocks, treeIdBlocks, gradBlocks) = boostConf.getSubSampleType match {
      case GBM.Goss =>
        adaptTreeInputsForGoss[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, boostConf, bcBoostConf, iteration, computeGradBlock, cleaner)

      case _ if boostConf.getSubSampleRate == 1 =>
        adaptTreeInputsForNonSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, boostConf, bcBoostConf, iteration, computeGradBlock, cleaner)

      case GBM.Partition =>
        adaptTreeInputsForPartitionSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, boostConf, bcBoostConf, iteration, computeGradBlock, cleaner)

      case GBM.Block =>
        adaptTreeInputsForBlockSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, boostConf, bcBoostConf, iteration, computeGradBlock, cleaner)

      case GBM.Row =>
        adaptTreeInputsForRowSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, boostConf, bcBoostConf, iteration, computeGrad, cleaner)
    }


    val trees = Tree.trainHorizontal[T, N, C, B, H](sampledBinVecBlocks, treeIdBlocks, gradBlocks, boostConf, bcBoostConf, baseConfig)

    cleaner.clear(false)

    System.gc()

    trees
  }


  def adaptTreeInputsForGoss[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                            labelBlocks: RDD[ArrayBlock[H]],
                                            binVecBlocks: RDD[KVMatrix[C, B]],
                                            rawBlocks: RDD[ArrayBlock[H]],
                                            boostConf: BoostConfig,
                                            bcBoostConf: Broadcast[BoostConfig],
                                            iteration: Int,
                                            computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                            cleaner: ResourceCleaner)
                                           (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                            cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                            cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                            cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                            ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]) = {
    import RDDFunctions._
    import nuh._

    val sc = weightBlocks.sparkContext

    val otherSelector = Selector.create(boostConf.getOtherSampleRate, boostConf.getNumBlocks * boostConf.getBlockSize,
      boostConf.getBaseModelParallelism, 1, boostConf.getSeed + iteration)
    logInfo(s"Iter $iteration: OtherSelector $otherSelector")

    val bcOtherSelector = sc.broadcast(otherSelector)
    cleaner.registerBroadcastedObjects(bcOtherSelector)


    // here all base models share the same gradient,
    // so we do not need to compute quantile for each one.
    val gradNormBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .map { case (weightBlock, labelBlock, rawBlock) =>
        val gradBlock = computeGradBlock(weightBlock, labelBlock, rawBlock)

        val normBlock = gradBlock.iterator
          .map { gradHess =>
            var gradNorm = nuh.zero
            var i = 0
            while (i < gradHess.length) {
              gradNorm += gradHess(i) * gradHess(i)
              i += 2
            }
            gradNorm
          }.toArray

        require(gradBlock.size == normBlock.length)
        (gradBlock, normBlock)
      }.setName(s"Iter $iteration: GradWithNorms")

    gradNormBlocks.persist(boostConf.getStorageLevel2)
    cleaner.registerCachedRDDs(gradNormBlocks)


    val tic = System.nanoTime()
    logInfo(s"Iteration $iteration: start to compute the threshold of top gradients")

    val summary = gradNormBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        var s = new QuantileSummaries(QuantileSummaries.defaultCompressThreshold,
          QuantileSummaries.defaultRelativeError)

        iter.flatMap(_._2).foreach { v => s = s.insert(nuh.toDouble(v)) }
        s = s.compress

        if (s.count > 0) {
          Iterator.single(s)
        } else if (partId == 0) {
          // avoid `treeReduce` on empty RDD
          Iterator.single(s)
        } else {
          Iterator.empty
        }

      }.treeReduce(f = _.merge(_).compress,
      depth = boostConf.getAggregationDepth)

    val threshold = neh.fromDouble(summary.query(1 - boostConf.getTopRate).get)
    logInfo(s"Iteration $iteration: threshold for top gradients: ${neh.sqrt(threshold)}, " +
      s"duration ${(System.nanoTime() - tic) / 1e9} seconds")


    val sampledBinVecBlocks = binVecBlocks.zip(gradNormBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val otherSelector = bcOtherSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { case (binVecBlock, (_, normBlock)) =>
          require(binVecBlock.size == normBlock.length)
          blockId += 1
          var rowId = blockId * blockSize - 1

          Utils.zip2(binVecBlock.iterator, normBlock.iterator)
            .flatMap { case (binVec, norm) =>
              rowId += 1
              if (norm >= threshold ||
                otherSelector.contains[Long](rowId)) {
                Iterator.single(binVec)
              } else {
                Iterator.empty
              }
            }
        }.grouped(blockSize)
          .map(KVMatrix.build[C, B])
      }.setName(s"Iter $iteration: BinVecs (GOSS)")

    sampledBinVecBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(sampledBinVecBlocks)


    val treeIdBlocks = gradNormBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val numTrees = bcBoostConf.value.getNumTrees
        val topTreeIds = Array.tabulate(numTrees)(int.fromInt)
        val computeTreeIds = bcBoostConf.value.computeTreeIds[T]()
        val otherSelector = bcOtherSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { case (_, normBlock) =>
          blockId += 1
          var rowId = blockId * blockSize - 1

          normBlock.iterator
            .flatMap { norm =>
              rowId += 1
              if (norm >= threshold) {
                Iterator.single(topTreeIds)
              } else {
                val baseIds = otherSelector.index[T, Long](rowId)
                if (baseIds.nonEmpty) {
                  val treeIds = computeTreeIds(baseIds)
                  Iterator.single(treeIds)
                } else {
                  Iterator.empty
                }
              }
            }
        }.grouped(blockSize)
          .map(ArrayBlock.build[T])
      }.setName(s"Iter $iteration: TreeIds (GOSS)")

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(treeIdBlocks)


    val gradBlocks = gradNormBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val weightScale = neh.fromDouble(bcBoostConf.value.getOtherReweight)
        val otherSelector = bcOtherSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { case (gradBlock, normBlock) =>
          require(gradBlock.size == normBlock.length)
          blockId += 1
          var rowId = blockId * blockSize - 1

          Utils.zip2(gradBlock.iterator, normBlock.iterator)
            .flatMap { case (grad, norm) =>
              rowId += 1
              if (norm >= threshold) {
                Iterator.single(grad)

              } else if (otherSelector.contains[Long](rowId)) {
                var i = 0
                while (i < grad.length) {
                  grad(i) *= weightScale
                  i += 1
                }
                Iterator.single(grad)

              } else {
                Iterator.empty
              }
            }
        }.grouped(blockSize)
          .map(ArrayBlock.build[H])
      }.setName(s"Iter $iteration: Grads (GOSS)")


    gradBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(gradBlocks)


    (sampledBinVecBlocks, treeIdBlocks, gradBlocks)
  }


  def adaptTreeInputsForNonSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                   labelBlocks: RDD[ArrayBlock[H]],
                                                   binVecBlocks: RDD[KVMatrix[C, B]],
                                                   rawBlocks: RDD[ArrayBlock[H]],
                                                   boostConf: BoostConfig,
                                                   bcBoostConf: Broadcast[BoostConfig],
                                                   iteration: Int,
                                                   computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                                   cleaner: ResourceCleaner)
                                                  (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                   cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                   cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                   cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                   ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]) = {
    import RDDFunctions._

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
    cleaner.registerCachedRDDs(treeIdBlocks)


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .map { case (weightBlock, labelBlock, rawBlock) =>
        computeGradBlock(weightBlock, labelBlock, rawBlock)
      }.setName(s"Iter $iteration: Grads")

    gradBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(gradBlocks)

    (binVecBlocks, treeIdBlocks, gradBlocks)
  }


  def adaptTreeInputsForPartitionSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                         labelBlocks: RDD[ArrayBlock[H]],
                                                         binVecBlocks: RDD[KVMatrix[C, B]],
                                                         rawBlocks: RDD[ArrayBlock[H]],
                                                         boostConf: BoostConfig,
                                                         bcBoostConf: Broadcast[BoostConfig],
                                                         iteration: Int,
                                                         computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                                         cleaner: ResourceCleaner)
                                                        (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                         cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                         cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                         cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                         ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]) = {
    import RDDFunctions._

    val sc = weightBlocks.sparkContext

    val partSelector = Selector.create(boostConf.getSubSampleRate, weightBlocks.getNumPartitions,
      boostConf.getBaseModelParallelism, 1, boostConf.getSeed + iteration)
    logInfo(s"Iter $iteration: PartSelector $partSelector")

    val bcPartSelector = sc.broadcast(partSelector)
    cleaner.registerBroadcastedObjects(bcPartSelector)


    val sampledBinVecBlocks = binVecBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val partSelector = bcPartSelector.value

        if (partSelector.contains[Int](partId)) {
          iter
        } else {
          Iterator.empty
        }
      }.setName(s"Iter $iteration: BinVecs (Partition-Sampled)")


    val treeIdBlocks = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val partSelector = bcPartSelector.value
        val baseIds = partSelector.index[T, Int](partId)

        if (baseIds.nonEmpty) {
          val computeTreeIds = bcBoostConf.value.computeTreeIds[T]()
          val treeIds = computeTreeIds(baseIds)
          val defaultTreeIdBlock = ArrayBlock.fill(blockSize, treeIds)

          iter.map { weightBlock =>
            if (weightBlock.size == blockSize) {
              defaultTreeIdBlock
            } else {
              ArrayBlock.fill(weightBlock.size, treeIds)
            }
          }
        } else {
          Iterator.empty
        }
      }.setName(s"Iter $iteration: TreeIds (Partition-Sampled)")

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(treeIdBlocks)


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val partSelector = bcPartSelector.value

        if (partSelector.contains[Int](partId)) {
          iter.map { case (weightBlock, labelBlock, rawBlock) =>
            computeGradBlock(weightBlock, labelBlock, rawBlock)
          }
        } else {
          Iterator.empty
        }
      }.setName(s"Iter $iteration: Grads (Partition-Sampled)")

    gradBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(gradBlocks)


    (sampledBinVecBlocks, treeIdBlocks, gradBlocks)
  }


  def adaptTreeInputsForBlockSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                     labelBlocks: RDD[ArrayBlock[H]],
                                                     binVecBlocks: RDD[KVMatrix[C, B]],
                                                     rawBlocks: RDD[ArrayBlock[H]],
                                                     boostConf: BoostConfig,
                                                     bcBoostConf: Broadcast[BoostConfig],
                                                     iteration: Int,
                                                     computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                                     cleaner: ResourceCleaner)
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
    cleaner.registerBroadcastedObjects(bcBlockSelector)


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
    cleaner.registerCachedRDDs(sampledBinVecBlocks)


    val treeIdBlocks = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val computeTreeIds = bcBoostConf.value.computeTreeIds[T]()
        val blockSelector = bcBlockSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { weightBlock =>
          blockId += 1
          val baseIds = blockSelector.index[T, Long](blockId)
          if (baseIds.nonEmpty) {
            val treeIds = computeTreeIds(baseIds)
            Iterator.single(ArrayBlock.fill(weightBlock.size, treeIds))
          } else {
            Iterator.empty
          }
        }
      }.setName(s"Iter $iteration: TreeIds (Block-Sampled)")

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(treeIdBlocks)


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSelector = bcBlockSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { case (weightBlock, labelBlock, rawBlock) =>
          blockId += 1
          if (blockSelector.contains[Long](blockId)) {
            val gradBlock = computeGradBlock(weightBlock, labelBlock, rawBlock)
            Iterator.single(gradBlock)
          } else {
            Iterator.empty
          }
        }
      }.setName(s"Iter $iteration: Grads (Block-Sampled)")

    gradBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(gradBlocks)


    (sampledBinVecBlocks, treeIdBlocks, gradBlocks)
  }


  def adaptTreeInputsForRowSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                   labelBlocks: RDD[ArrayBlock[H]],
                                                   binVecBlocks: RDD[KVMatrix[C, B]],
                                                   rawBlocks: RDD[ArrayBlock[H]],
                                                   boostConf: BoostConfig,
                                                   bcBoostConf: Broadcast[BoostConfig],
                                                   iteration: Int,
                                                   computeGrad: (H, Array[H], Array[H]) => Array[H],
                                                   cleaner: ResourceCleaner)
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
    cleaner.registerBroadcastedObjects(bcRowSelector)


    val sampledBinVecBlocks = binVecBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val rowSelector = bcRowSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { binVecBlock =>
          blockId += 1
          var rowId = blockId * blockSize - 1

          binVecBlock.iterator
            .flatMap { binVec =>
              rowId += 1
              if (rowSelector.contains[Long](rowId)) {
                Iterator.single(binVec)
              } else {
                Iterator.empty
              }
            }
        }.grouped(blockSize)
          .map(KVMatrix.build[C, B])
      }.setName(s"Iter $iteration: BinVecs (Row-Sampled)")

    sampledBinVecBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(sampledBinVecBlocks)


    val treeIdBlocks = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val computeTreeIds = bcBoostConf.value.computeTreeIds[T]()
        val rowSelector = bcRowSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { weightBlock =>
          blockId += 1
          var rowId = blockId * blockSize - 1

          Iterator.range(0, weightBlock.size)
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
        }.grouped(blockSize)
          .map(ArrayBlock.build[T])
      }.setName(s"Iter $iteration: TreeIds (Row-Sampled)")

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(treeIdBlocks)


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val blockSize = bcBoostConf.value.getBlockSize
        val rowSelector = bcRowSelector.value
        var blockId = bcBoostConf.value.getBlockOffset(partId) - 1

        iter.flatMap { case (weightBlock, labelBlock, rawBlock) =>
          require(weightBlock.size == rawBlock.size)
          require(labelBlock.size == rawBlock.size)
          blockId += 1
          var rowId = blockId * blockSize - 1

          Utils.zip3(weightBlock.iterator, labelBlock.iterator, rawBlock.iterator)
            .flatMap { case (weight, label, rawSeq) =>
              rowId += 1
              if (rowSelector.contains[Long](rowId)) {
                val grad = computeGrad(weight, label, rawSeq)
                Iterator.single(grad)
              } else {
                Iterator.empty
              }
            }
        }.grouped(blockSize)
          .map(ArrayBlock.build[H])
      }.setName(s"Iter $iteration: Grads (Row-Sampled)")

    gradBlocks.persist(boostConf.getStorageLevel1)
    cleaner.registerCachedRDDs(gradBlocks)


    (sampledBinVecBlocks, treeIdBlocks, gradBlocks)
  }
}


