package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.rdd._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.util.QuantileSummaries
import org.apache.spark.util.random.XORShiftRandom


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

    val spark = SparkSession.builder().getOrCreate()
    val sc = spark.sparkContext

    val rawBase = boostConf.getRawBaseScore

    // train blocks
    val (trainWeightBlocks, trainLabelBlocks, trainBinVecBlocks) = trainBlocks
    GBM.touchWeightBlocks[H](trainWeightBlocks, boostConf)

    // test blocks
    testBlocks.foreach { case (testWeightBlocks, _, _) => GBM.touchWeightBlocks(testWeightBlocks, boostConf) }


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
      .setName("Train Raw Blocks (Iteration 0)")
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
        .setName("Test Raw Blocks (Iteration 0)")
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

    var iter = 0
    var finished = false

    while (!finished && iter < boostConf.getMaxIter) {
      val numTrees = treesBuff.length
      val logPrefix = s"Iteration $iter:"

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
      val trees = buildTrees[C, B, H](trainWeightBlocks, trainLabelBlocks, trainBinVecBlocks, trainRawBlocks,
        weightsBuff.toArray, boostConf, iter, dropped.toSet)
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
          .setName(s"Train Raw Blocks (Iteration ${iter + 1})")
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
            .setName(s"Test Raw Blocks (Iteration ${iter + 1})")

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
            rawBase.clone(), treesBuff.toArray.clone(), neh.toDouble(weightsBuff.toArray).clone())

          // callback can update boosting configuration
          boostConf.getCallbackFunc.foreach { callback =>
            if (callback.compute(spark, boostConf, snapshot, iter + 1,
              trainMetricsHistory.toArray.clone(), testMetricsHistory.toArray.clone())) {
              finished = true
              logInfo(s"$logPrefix callback ${callback.name} stop training")
            }
          }
        }
      }

      logInfo(s"$logPrefix finished, ${treesBuff.length} trees now")
      iter += 1
    }

    if (iter >= boostConf.getMaxIter) {
      logInfo(s"maxIter=${boostConf.getMaxIter} reached, GBM training finished")
    }

    trainRawBlocksCheckpointer.clear()
    if (testRawBlocksCheckpointer != null) {
      testRawBlocksCheckpointer.clear()
    }

    new GBMModel(boostConf.getObjFunc, discretizer, rawBase,
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

    val recoder = new ResourceRecoder

    val bcBoostConf = sc.broadcast(boostConf)
    recoder.append(bcBoostConf)


    val numBaseModels = boostConf.getBaseModelParallelism
    val rawBase = neh.fromDouble(boostConf.getRawBaseScore)
    val rawSize = boostConf.getRawSize

    val computeRaw = boostConf.getBoostType match {
      case GBM.GBTree =>
        rawSeq: Array[H] => rawSeq

      case GBM.Dart if dropped.isEmpty =>
        rawSeq: Array[H] => rawSeq.take(rawSize)

      case GBM.Dart if dropped.nonEmpty =>
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

      val iter = weightBlock.iterator
        .zip(labelBlock.iterator)
        .zip(rawBlock.iterator)
        .map { case ((weight, label), rawSeq) => computeGrad(weight, label, rawSeq) }

      val gradBlock = ArrayBlock.build[H](iter)
      require(gradBlock.size == rawBlock.size)
      gradBlock
    }


    val baseConfig = BaseConfig.create(boostConf, iteration, numBaseModels, boostConf.getSeed + iteration)

    // To alleviate memory footprint in caching layer, different schemas of intermediate dataset are designed.
    // Each `prepareTreeInput**` method will internally cache necessary datasets in a compact fashion.
    // These cached datasets are holden in `recoder`, and will be freed after training.
    val (sampledBinVecBlocks, treeIdBlocks, gradBlocks) = boostConf.getSubSampleType match {
      case GBM.Goss =>
        adaptTreeInputsForGoss[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, boostConf, iteration, computeGradBlock, recoder)

      case _ if boostConf.getSubSampleRate == 1 =>
        adaptTreeInputsForNonSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, boostConf, iteration, computeGradBlock, recoder)

      case GBM.Partition =>
        adaptTreeInputsForPartitionSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, boostConf, iteration, computeGradBlock, recoder)

      case GBM.Block =>
        adaptTreeInputsForBlockSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, boostConf, iteration, computeGradBlock, recoder)

      case GBM.Instance =>
        adaptTreeInputsForInstanceSampling[T, N, C, B, H](weightBlocks, labelBlocks, binVecBlocks, rawBlocks, boostConf, iteration, computeGrad, recoder)
    }


    val trees = Tree.trainHorizontal[T, N, C, B, H](sampledBinVecBlocks, treeIdBlocks, gradBlocks, boostConf, bcBoostConf, baseConfig)

    recoder.clear(true)

    trees
  }


  def adaptTreeInputsForGoss[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                            labelBlocks: RDD[ArrayBlock[H]],
                                            binVecBlocks: RDD[KVMatrix[C, B]],
                                            rawBlocks: RDD[ArrayBlock[H]],
                                            boostConf: BoostConfig,
                                            iteration: Int,
                                            computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                            recoder: ResourceRecoder)
                                           (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                            cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                            cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                            cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                            ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): (RDD[KVMatrix[C, B]], RDD[ArrayBlock[T]], RDD[ArrayBlock[H]]) = {
    import RDDFunctions._
    import nuh._

    val blockSize = boostConf.getBlockSize
    val rawSize = boostConf.getRawSize
    val numBaseModels = boostConf.getBaseModelParallelism
    val numTrees = numBaseModels * rawSize
    val lowSample = 1 / boostConf.computeOtherReweight
    val seedOffset = boostConf.getSeed + iteration
    val weightScale = neh.fromDouble(boostConf.computeOtherReweight)
    val computeTreeIds = GBM.getTreeIds[T](rawSize)


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
      }.setName(s"GradNormBlocks (iteration $iteration)")

    gradNormBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(gradNormBlocks)


    val start = System.nanoTime
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
      s"duration ${(System.nanoTime - start) / 1e9} seconds")


    val sampledBinVecBlocks = binVecBlocks.zip(gradNormBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)

        iter.flatMap { case (binVecBlock, (_, normBlock)) =>
          require(binVecBlock.size == normBlock.length)

          binVecBlock.iterator.zip(normBlock.iterator)
            .flatMap { case (binVec, norm) =>
              if (norm >= threshold) {
                Iterator.single(binVec)
              } else {
                val baseIds = Array.range(0, numBaseModels)
                  .filter(_ => sampleRNG.nextDouble < lowSample)
                if (baseIds.nonEmpty) {
                  Iterator.single(binVec)
                } else {
                  Iterator.empty
                }
              }
            }
        }.grouped(blockSize)
          .map(KVMatrix.build[C, B])
      }.setName(s"BinVecBlocks (iteration $iteration) (GOSS)")


    val treeIdBlocks = gradNormBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)
        val topTreeIds = Array.tabulate(numTrees)(int.fromInt)

        iter.flatMap { case (_, normBlock) =>
          normBlock.iterator.flatMap { norm =>
            if (norm >= threshold) {
              Iterator.single(topTreeIds)
            } else {
              val baseIds = Array.range(0, numBaseModels)
                .filter(_ => sampleRNG.nextDouble < lowSample)
              if (baseIds.nonEmpty) {
                val treeIds = computeTreeIds(net.fromInt(baseIds))
                Iterator.single(treeIds)
              } else {
                Iterator.empty
              }
            }
          }.grouped(blockSize)
            .map(ArrayBlock.build[T])
        }
      }.setName(s"TreeIdBlocks (iteration $iteration) (GOSS)")


    val gradBlocks = gradNormBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)

        iter.flatMap { case (gradBlock, normBlock) =>
          require(gradBlock.size == normBlock.length)

          gradBlock.iterator.zip(normBlock.iterator)
            .flatMap { case (grad, norm) =>
              if (norm >= threshold) {
                Iterator.single(grad)
              } else {
                val baseIds = Array.range(0, numBaseModels)
                  .filter(_ => sampleRNG.nextDouble < lowSample)
                if (baseIds.nonEmpty) {
                  Iterator.single(grad)
                } else {
                  Iterator.empty
                }
              }
            }.map { grad =>
            var i = 0
            while (i < grad.length) {
              grad(i) *= weightScale
              i += 1
            }
            grad
          }.grouped(blockSize)
            .map(ArrayBlock.build[H])
        }
      }.setName(s"GradBlocks (iteration $iteration) (GOSS)")


    gradBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(gradBlocks)

    sampledBinVecBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(sampledBinVecBlocks)

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(treeIdBlocks)

    (sampledBinVecBlocks, treeIdBlocks, gradBlocks)
  }


  def adaptTreeInputsForNonSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                   labelBlocks: RDD[ArrayBlock[H]],
                                                   binVecBlocks: RDD[KVMatrix[C, B]],
                                                   rawBlocks: RDD[ArrayBlock[H]],
                                                   boostConf: BoostConfig,
                                                   iteration: Int,
                                                   computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                                   recoder: ResourceRecoder)
                                                  (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                   cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                   cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                   cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                   ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): (RDD[KVMatrix[C, B]], RDD[ArrayBlock[T]], RDD[ArrayBlock[H]]) = {
    import RDDFunctions._

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
      }.setName(s"TreeIdBlocks (iteration $iteration)")

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(treeIdBlocks)

    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .map { case (weightBlock, labelBlock, rawBlock) =>
        computeGradBlock(weightBlock, labelBlock, rawBlock)
      }.setName(s"GradBlocks (iteration $iteration)")


    boostConf.getStorageStrategy match {
      case GBM.Upstream =>
        gradBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(gradBlocks)

      case GBM.Eager =>
        treeIdBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(treeIdBlocks)

        gradBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(gradBlocks)
    }

    gradBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(gradBlocks)

    (binVecBlocks, treeIdBlocks, gradBlocks)
  }


  def adaptTreeInputsForPartitionSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                         labelBlocks: RDD[ArrayBlock[H]],
                                                         binVecBlocks: RDD[KVMatrix[C, B]],
                                                         rawBlocks: RDD[ArrayBlock[H]],
                                                         boostConf: BoostConfig,
                                                         iteration: Int,
                                                         computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                                         recoder: ResourceRecoder)
                                                        (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                         cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                         cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                         cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                         ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): (RDD[KVMatrix[C, B]], RDD[ArrayBlock[T]], RDD[ArrayBlock[H]]) = {
    import RDDFunctions._

    val rawSize = boostConf.getRawSize
    val numBaseModels = boostConf.getBaseModelParallelism
    val subSample = boostConf.getSubSampleRate
    val blockSize = boostConf.getBlockSize
    val seedOffset = boostConf.getSeed + iteration
    val computeTreeIds = GBM.getTreeIds[T](rawSize)


    val sampledBinVecBlocks = binVecBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)
        val baseIds = Array.range(0, numBaseModels)
          .filter(_ => sampleRNG.nextDouble < subSample)

        if (baseIds.nonEmpty) {
          iter
        } else {
          Iterator.empty
        }
      }.setName(s"BinVecBlocks (iteration $iteration) (Partition-Sampled)")


    val treeIdBlocks = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)
        val baseIds = Array.range(0, numBaseModels)
          .filter(_ => sampleRNG.nextDouble < subSample)

        if (baseIds.nonEmpty) {
          val treeIds = computeTreeIds(net.fromInt(baseIds))
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
      }.setName(s"TreeIdBlocks (iteration $iteration) (Partition-Sampled)")


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)
        val baseIds = Array.range(0, numBaseModels)
          .filter(_ => sampleRNG.nextDouble < subSample)

        if (baseIds.nonEmpty) {
          iter.map { case (weightBlock, labelBlock, rawBlock) =>
            computeGradBlock(weightBlock, labelBlock, rawBlock)
          }
        } else {
          Iterator.empty
        }
      }.setName(s"GradBlocks (iteration $iteration) (Partition-Sampled)")


    boostConf.getStorageStrategy match {
      case GBM.Upstream =>
        gradBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(gradBlocks)

      case GBM.Eager =>
        sampledBinVecBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(sampledBinVecBlocks)

        treeIdBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(treeIdBlocks)

        gradBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(gradBlocks)
    }

    (sampledBinVecBlocks, treeIdBlocks, gradBlocks)
  }


  def adaptTreeInputsForBlockSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                     labelBlocks: RDD[ArrayBlock[H]],
                                                     binVecBlocks: RDD[KVMatrix[C, B]],
                                                     rawBlocks: RDD[ArrayBlock[H]],
                                                     boostConf: BoostConfig,
                                                     iteration: Int,
                                                     computeGradBlock: (CompactArray[H], ArrayBlock[H], ArrayBlock[H]) => ArrayBlock[H],
                                                     recoder: ResourceRecoder)
                                                    (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                     cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                     cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                     cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                     ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): (RDD[KVMatrix[C, B]], RDD[ArrayBlock[T]], RDD[ArrayBlock[H]]) = {
    import RDDFunctions._

    val rawSize = boostConf.getRawSize
    val numBaseModels = boostConf.getBaseModelParallelism
    val seedOffset = boostConf.getSeed + iteration
    val subSample = boostConf.getSubSampleRate
    val computeTreeIds = GBM.getTreeIds[T](rawSize)


    val sampledBinVecBlocks = binVecBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)
        iter.filter { _ =>
          val baseIds = Array.range(0, numBaseModels)
            .filter(_ => sampleRNG.nextDouble < subSample)
          baseIds.nonEmpty
        }
      }.setName(s"BinVecBlocks (iteration $iteration) (Block-Sampled)")


    val treeIdBlocks = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)

        iter.flatMap { weightBlock =>
          val baseIds = Array.range(0, numBaseModels)
            .filter(_ => sampleRNG.nextDouble < subSample)

          if (baseIds.nonEmpty) {
            val treeIds = computeTreeIds(net.fromInt(baseIds))
            Iterator.single(ArrayBlock.fill(weightBlock.size, treeIds))
          } else {
            Iterator.empty
          }
        }
      }.setName(s"TreeIdBlocks (iteration $iteration) (Block-Sampled)")


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)

        iter.flatMap { case (weightBlock, labelBlock, rawBlock) =>
          val baseIds = Array.range(0, numBaseModels)
            .filter(_ => sampleRNG.nextDouble < subSample)

          if (baseIds.nonEmpty) {
            val gradBlock = computeGradBlock(weightBlock, labelBlock, rawBlock)
            Iterator.single(gradBlock)
          } else {
            Iterator.empty
          }
        }
      }.setName(s"GradBlocks (iteration $iteration) (Block-Sampled)")


    boostConf.getStorageStrategy match {
      case GBM.Upstream =>
        gradBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(gradBlocks)

      case GBM.Eager =>
        sampledBinVecBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(sampledBinVecBlocks)

        treeIdBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(treeIdBlocks)

        gradBlocks.persist(boostConf.getStorageLevel1)
        recoder.append(gradBlocks)
    }

    (sampledBinVecBlocks, treeIdBlocks, gradBlocks)
  }


  def adaptTreeInputsForInstanceSampling[T, N, C, B, H](weightBlocks: RDD[CompactArray[H]],
                                                        labelBlocks: RDD[ArrayBlock[H]],
                                                        binVecBlocks: RDD[KVMatrix[C, B]],
                                                        rawBlocks: RDD[ArrayBlock[H]],
                                                        boostConf: BoostConfig,
                                                        iteration: Int,
                                                        computeGrad: (H, Array[H], Array[H]) => Array[H],
                                                        recoder: ResourceRecoder)
                                                       (implicit ct: ClassTag[T], int: Integral[T], net: NumericExt[T],
                                                        cn: ClassTag[N], inn: Integral[N], nen: NumericExt[N],
                                                        cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                        cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                                        ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): (RDD[KVMatrix[C, B]], RDD[ArrayBlock[T]], RDD[ArrayBlock[H]]) = {
    import RDDFunctions._

    val rawSize = boostConf.getRawSize
    val numBaseModels = boostConf.getBaseModelParallelism
    val blockSize = boostConf.getBlockSize
    val seedOffset = boostConf.getSeed + iteration
    val subSample = boostConf.getSubSampleRate
    val computeTreeIds = GBM.getTreeIds[T](rawSize)


    val sampledBinVecBlocks = binVecBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)

        iter.flatMap { binVecBlock =>
          binVecBlock.iterator
            .filter { _ =>
              val baseIds = Array.range(0, numBaseModels)
                .filter(_ => sampleRNG.nextDouble < subSample)
              baseIds.nonEmpty
            }
        }.grouped(blockSize)
          .map(KVMatrix.build[C, B])
      }.setName(s"BinVecBlocks (iteration $iteration) (Instance-Sampled)")


    val treeIdBlocks = weightBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)

        iter.flatMap { weightBlock =>
          Iterator.range(0, weightBlock.size)
            .flatMap { _ =>
              val baseIds = Array.range(0, numBaseModels)
                .filter(_ => sampleRNG.nextDouble < subSample)

              if (baseIds.nonEmpty) {
                val treeIds = computeTreeIds(net.fromInt(baseIds))
                Iterator.single(treeIds)
              } else {
                Iterator.empty
              }
            }
        }.grouped(blockSize)
          .map(ArrayBlock.build[T])
      }.setName(s"TreeIdBlocks (iteration $iteration) (Instance-Sampled)")


    val gradBlocks = weightBlocks.zip2(labelBlocks, rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNG = new XORShiftRandom(seedOffset + partId)

        iter.flatMap { case (weightBlock, labelBlock, rawBlock) =>
          require(weightBlock.size == rawBlock.size)
          require(labelBlock.size == rawBlock.size)

          Utils.zip3(weightBlock.iterator, labelBlock.iterator, rawBlock.iterator)
            .flatMap { case (weight, label, rawSeq) =>
              val baseIds = Array.range(0, numBaseModels)
                .filter(_ => sampleRNG.nextDouble < subSample)

              if (baseIds.nonEmpty) {
                val grad = computeGrad(weight, label, rawSeq)
                Iterator.single(grad)
              } else {
                Iterator.empty
              }
            }
        }.grouped(blockSize)
          .map(ArrayBlock.build[H])
      }.setName(s"GradBlocks (iteration $iteration) (Instance-Sampled)")


    gradBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(gradBlocks)

    sampledBinVecBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(sampledBinVecBlocks)

    treeIdBlocks.persist(boostConf.getStorageLevel1)
    recoder.append(treeIdBlocks)

    (sampledBinVecBlocks, treeIdBlocks, gradBlocks)
  }
}


