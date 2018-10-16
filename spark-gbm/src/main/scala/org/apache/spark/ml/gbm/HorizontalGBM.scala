package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.util.QuantileSummaries
import org.apache.spark.util.random.XORShiftRandom


object HorizontalGBM extends Logging {

  /**
    * implementation of GBM, train a GBMModel, with given types
    *
    * @param trainBlocks  training blocks containing (weight+Label, binVec)
    * @param testBlocks   validation blocks containing (weight+Label, binVec)
    * @param boostConf    boosting configuration
    * @param discretizer  discretizer to convert raw features into bins
    * @param initialModel inital model
    * @return the model
    */
  def boost[C, B, H](trainBlocks: (RDD[(CompactArray[H], ArrayBlock[H])], RDD[KVMatrix[C, B]]),
                     testBlocks: Option[(RDD[(CompactArray[H], ArrayBlock[H])], RDD[KVMatrix[C, B]])],
                     boostConf: BoostConfig,
                     discretizer: Discretizer,
                     initialModel: Option[GBMModel])
                    (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                     cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                     ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): GBMModel = {

    val spark = SparkSession.builder().getOrCreate()
    val sc = spark.sparkContext

    val rawBase = boostConf.computeRawBaseScore

    // train blocks
    val (trainWeightLabelBlocks, trainBinVecBlocks) = trainBlocks
    GBM.touchBlocks[H](trainWeightLabelBlocks, boostConf)

    // test blocks
    testBlocks.foreach { case (testWeightLabelBlocks, _) => GBM.touchBlocks(testWeightLabelBlocks, boostConf) }


    // init tree buffer
    val weightsBuff = mutable.ArrayBuffer.empty[H]
    val treesBuff = mutable.ArrayBuffer.empty[TreeModel]
    initialModel.foreach { model =>
      weightsBuff.appendAll(neh.fromDouble(model.weights))
      treesBuff.appendAll(model.trees)
    }


    // raw scores and checkpointer
    var trainRawBlocks = GBM.initializeRawBlocks[C, B, H](trainBinVecBlocks,
      treesBuff.toArray, weightsBuff.toArray, boostConf)
      .setName("Train Raw Blocks (Iteration 0)")
    val trainRawBlocksCheckpointer = new Checkpointer[ArrayBlock[H]](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel)
    if (treesBuff.nonEmpty) {
      trainRawBlocksCheckpointer.update(trainRawBlocks)
    }


    // raw scores and checkpointer for test data
    val testRawBlocksCheckpointer = testBlocks.map(_ => new Checkpointer[ArrayBlock[H]](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel))
    var testRawBlocks = testBlocks.map { case (_, testBinVecBlocks) =>
      val newTestRawBlocks = GBM.initializeRawBlocks[C, B, H](testBinVecBlocks,
        treesBuff.toArray, weightsBuff.toArray, boostConf)
        .setName("Test Raw Blocks (Iteration 0)")
      if (treesBuff.nonEmpty) {
        testRawBlocksCheckpointer.get.update(newTestRawBlocks)
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
      val trees = buildTrees[C, B, H](trainWeightLabelBlocks, trainBinVecBlocks, trainRawBlocks,
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
          val trainMetrics = GBM.evaluate[H, C, B](trainWeightLabelBlocks, trainRawBlocks, boostConf)
          trainMetricsHistory.append(trainMetrics)
          logInfo(s"$logPrefix train metrics ${trainMetrics.mkString("(", ", ", ")")}")
        }


        testBlocks.foreach { case (testWeightLabelBlocks, testBinVecBlocks) =>
          // update test data predictions
          val newTestRawBlocks = GBM.updateRawBlocks[C, B, H](testBinVecBlocks, testRawBlocks.get,
            trees, weightsBuff.toArray, boostConf, keepWeights)
            .setName(s"Test Raw Blocks (Iteration ${iter + 1})")

          testRawBlocks = Some(newTestRawBlocks)
          testRawBlocksCheckpointer.get.update(testRawBlocks.get)

          // evaluate on test data
          val testMetrics = GBM.evaluate[H, C, B](testWeightLabelBlocks, testRawBlocks.get, boostConf)
          testMetricsHistory.append(testMetrics)
          logInfo(s"$logPrefix test metrics ${testMetrics.mkString("(", ", ", ")")}")
        }


        // callback
        if (boostConf.getCallbackFunc.nonEmpty) {
          // using cloning to avoid model modification
          val snapshot = new GBMModel(boostConf.getObjFunc, discretizer.clone(),
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
    testRawBlocksCheckpointer.foreach(_.clear())

    new GBMModel(boostConf.getObjFunc, discretizer, rawBase,
      treesBuff.toArray, neh.toDouble(weightsBuff.toArray))
  }


  def buildTrees[C, B, H](weightLabelBlocks: RDD[(CompactArray[H], ArrayBlock[H])],
                          binVecBlocks: RDD[KVMatrix[C, B]],
                          rawBlocks: RDD[ArrayBlock[H]],
                          weights: Array[H],
                          boostConf: BoostConfig,
                          iteration: Int,
                          dropped: Set[Int])
                         (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                          cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                          ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Array[TreeModel] = {

    val numTrees = boostConf.getBaseModelParallelism * boostConf.getRawSize
    logInfo(s"Iteration $iteration: Starting to create next $numTrees trees")

    val treeIdType = Utils.getTypeByRange(numTrees)
    logInfo(s"DataType of TreeId: $treeIdType")

    val nodeIdType = Utils.getTypeByRange(1 << boostConf.getMaxDepth)
    logInfo(s"DataType of NodeId: $nodeIdType")

    (treeIdType, nodeIdType) match {
      case ("Byte", "Byte") =>
        buildTreesImpl[Byte, Byte, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case ("Byte", "Short") =>
        buildTreesImpl[Byte, Short, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case ("Byte", "Int") =>
        buildTreesImpl[Byte, Int, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case ("Short", "Byte") =>
        buildTreesImpl[Short, Byte, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case ("Short", "Short") =>
        buildTreesImpl[Short, Short, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case ("Short", "Int") =>
        buildTreesImpl[Short, Int, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case ("Int", "Byte") =>
        buildTreesImpl[Int, Byte, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case ("Int", "Short") =>
        buildTreesImpl[Int, Short, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)

      case ("Int", "Int") =>
        buildTreesImpl[Int, Int, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, weights, boostConf, iteration, dropped)
    }
  }


  /**
    * build new trees
    *
    * @param weightLabelBlocks blockified instances containing (weight, labels)
    * @param rawBlocks         previous raw predictions
    * @param weights           weights of trees
    * @param boostConf         boosting configuration
    * @param iteration         current iteration
    * @param dropped           indices of trees which are selected to drop during building of current tree
    * @return new trees
    */
  def buildTreesImpl[T, N, C, B, H](weightLabelBlocks: RDD[(CompactArray[H], ArrayBlock[H])],
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

    val numBaseModels = boostConf.getBaseModelParallelism
    val numTrees = numBaseModels * boostConf.getRawSize

    val rawBase = neh.fromDouble(boostConf.computeRawBaseScore)
    val rawSize = boostConf.getRawSize

    val computeRaw = boostConf.getBoostType match {
      case GBM.GBTree =>
        rawSeq: Array[H] => rawSeq

      case GBM.Goss =>
        rawSeq: Array[H] => rawSeq

      case GBM.Dart if dropped.isEmpty =>
        rawSeq: Array[H] => rawSeq.take(rawSize)

      case GBM.Dart if dropped.nonEmpty =>
        rawSeq: Array[H] =>
          val raw = rawBase.clone
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


    val recoder = new ResourceRecoder

    val baseConfig = BaseConfig.create(boostConf, iteration, numBaseModels, boostConf.getSeed + iteration)

    // To alleviate memory footprint in caching layer, different schemas of intermediate dataset are designed.
    // Each `prepareTreeInput**` method will internally cache necessary datasets in a compact fashion.
    // These cached datasets are holden in `recoder`, and will be freed after training.
    val data = (boostConf.getSubSampleType, boostConf.getSubSampleRate == 1) match {
      case (GBM.Goss, _) =>
        adaptTreeInputsForGoss[T, N, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, boostConf, iteration, computeGradBlock, recoder)

      case (_, true) =>
        adaptTreeInputsForNonSampling[T, N, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, boostConf, iteration, computeGradBlock, recoder)

      case (GBM.Partition, _) =>
        adaptTreeInputsForPartitionSampling[T, N, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, boostConf, iteration, computeGradBlock, recoder)

      case (GBM.Block, _) =>
        adaptTreeInputsForBlockSampling[T, N, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, boostConf, iteration, computeGradBlock, recoder)

      case (GBM.Instance, _) =>
        adaptTreeInputsForInstanceSampling[T, N, C, B, H](weightLabelBlocks, binVecBlocks, rawBlocks, boostConf, iteration, computeGrad, recoder)
    }


    val trees = Tree.trainHorizontal[T, N, C, B, H](data, boostConf, baseConfig)

    recoder.clear()

    trees
  }


  def adaptTreeInputsForNonSampling[T, N, C, B, H](weightLabelBlocks: RDD[(CompactArray[H], ArrayBlock[H])],
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
                                                   ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[(KVVector[C, B], Array[T], Array[H])] = {
    val rawSize = boostConf.getRawSize
    val numBaseModels = boostConf.getBaseModelParallelism
    val numTrees = numBaseModels * rawSize

    val gradBlocks = weightLabelBlocks.zip(rawBlocks)
      .map { case ((weightBlock, labelBlock), rawBlock) => computeGradBlock(weightBlock, labelBlock, rawBlock) }

    gradBlocks.setName(s"GradientBlocks (iteration $iteration)")
    gradBlocks.persist(boostConf.getStorageLevel)
    recoder.append(gradBlocks)

    binVecBlocks.zip(gradBlocks).mapPartitions { iter =>
      val treeIds = Array.tabulate(numTrees)(int.fromInt)

      iter.flatMap { case (binVecBlock, gradBlock) =>
        require(binVecBlock.size == gradBlock.size,
          s"size of bin vectors: ${binVecBlock.size} != size of gradients : ${gradBlock.size}")

        binVecBlock.iterator.zip(gradBlock.iterator)
          .map { case (binVec, grad) => (binVec, treeIds, grad) }
      }
    }.setName(s"Gradients with TreeIds (iteration $iteration)")
  }


  def adaptTreeInputsForGoss[T, N, C, B, H](weightLabelBlocks: RDD[(CompactArray[H], ArrayBlock[H])],
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
                                            ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[(KVVector[C, B], Array[T], Array[H])] = {
    import nuh._

    val rawSize = boostConf.getRawSize
    val numBaseModels = boostConf.getBaseModelParallelism
    val numTrees = numBaseModels * rawSize

    val gradBlocks = weightLabelBlocks.zip(rawBlocks)
      .map { case ((weightBlock, labelBlock), rawBlock) =>

        val gradBlock = computeGradBlock(weightBlock, labelBlock, rawBlock)

        val gradNorms = gradBlock.iterator.map { gradHess =>
          var gradNorm = nuh.zero
          var i = 0
          while (i < gradHess.length) {
            gradNorm += gradHess(i) * gradHess(i)
            i += 2
          }
          gradNorm
        }.toArray

        require(gradBlock.size == gradNorms.length)
        (gradBlock, gradNorms)
      }

    gradBlocks.setName(s"GradientBlocks with Gradient-Norms (iteration $iteration)")
    gradBlocks.persist(boostConf.getStorageLevel)
    recoder.append(gradBlocks)


    val start = System.nanoTime
    logInfo(s"Iteration $iteration: start to compute the threshold of top gradients")

    val summary = gradBlocks.mapPartitionsWithIndex { case (partId, iter) =>
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


    val lowSample = 1 / boostConf.computeOtherReweight
    val seedOffset = boostConf.getSeed + iteration

    val baseIdBlocks = gradBlocks.mapPartitionsWithIndex { case (partId, iter) =>
      val rng = new Random(seedOffset + partId)
      val rngs = Array.tabulate(numBaseModels)(_ => new XORShiftRandom(rng.nextLong))

      val topBaseId = Array(int.negate(int.one))

      iter.map { case (gradBlock, gradNorms) =>
        require(gradBlock.size == gradNorms.length)

        val baseIdIter = gradNorms.map { gradNorm =>
          if (gradNorm >= threshold) {
            topBaseId
          } else {
            Iterator.range(0, numBaseModels).filter { i => rngs(i).nextDouble < lowSample }
              .map(int.fromInt).toArray
          }
        }
        ArrayBlock.build[T](baseIdIter.iterator)
      }
    }

    baseIdBlocks.setName(s"BaseIdBlocks (iteration $iteration)")
    baseIdBlocks.persist(boostConf.getStorageLevel)
    recoder.append(baseIdBlocks)


    val computeTreeIds = if (rawSize == 1) {
      baseIds: Array[T] => baseIds
    } else {
      baseIds: Array[T] =>
        baseIds.flatMap { i =>
          val offset = rawSize * int.toInt(i)
          Iterator.range(offset, offset + rawSize).map(int.fromInt)
        }
    }

    val weightScale = neh.fromDouble(boostConf.computeOtherReweight)

    binVecBlocks.zip(gradBlocks).zip(baseIdBlocks).mapPartitions { iter =>
      val topTreeId = Array.tabulate(numTrees)(int.fromInt)

      iter.flatMap { case ((binVecBlock, (gradBlock, _)), baseIdBlock) =>
        require(binVecBlock.size == gradBlock.size)
        require(binVecBlock.size == baseIdBlock.size)

        binVecBlock.iterator.zip(gradBlock.iterator).zip(baseIdBlock.iterator)
          .flatMap { case ((binVec, grad), baseIds) =>
            if (baseIds.length == 1 && int.lt(baseIds.head, int.zero)) {
              Iterator.single((binVec, topTreeId, grad))

            } else if (baseIds.nonEmpty) {
              val treeIds = computeTreeIds(baseIds)
              var i = 0
              while (i < grad.length) {
                grad(i) *= weightScale
                i += 1
              }
              Iterator.single((binVec, treeIds, grad))

            } else {
              Iterator.empty
            }
          }
      }
    }.setName(s"Gradients with TreeIds (iteration $iteration) (Gradient-based One-Side Sampled)")
  }


  def adaptTreeInputsForPartitionSampling[T, N, C, B, H](weightLabelBlocks: RDD[(CompactArray[H], ArrayBlock[H])],
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
                                                         ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[(KVVector[C, B], Array[T], Array[H])] = {
    val rawSize = boostConf.getRawSize
    val numBaseModels = boostConf.getBaseModelParallelism
    val subSample = boostConf.getSubSampleRate
    val seedOffset = boostConf.getSeed + iteration

    val gradBlocks = weightLabelBlocks.zip(rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val rng = new Random(seedOffset + partId)
        val selected = Iterator.range(0, numBaseModels).exists(_ => rng.nextDouble < subSample)

        if (selected) {
          iter.map { case ((weightBlock, labelBlock), rawBlock) =>
            computeGradBlock(weightBlock, labelBlock, rawBlock)
          }
        } else {
          Iterator.empty
        }
      }

    gradBlocks.setName(s"GradientBlocks (iteration $iteration)")
    gradBlocks.persist(boostConf.getStorageLevel)
    recoder.append(gradBlocks)

    val computeTreeIds = if (rawSize == 1) {
      baseId: Array[T] => baseId
    } else {
      baseId: Array[T] =>
        baseId.flatMap { i =>
          val offset = rawSize * int.toInt(i)
          Iterator.range(offset, offset + rawSize).map(int.fromInt)
        }
    }

    binVecBlocks.zip(gradBlocks).mapPartitionsWithIndex { case (partId, iter) =>
      val rng = new Random(seedOffset + partId)
      val baseIds = Iterator.range(0, numBaseModels).filter(_ => rng.nextDouble < subSample).map(int.fromInt).toArray

      if (baseIds.nonEmpty) {
        val treeIds = computeTreeIds(baseIds)

        iter.flatMap { case (binVecBlock, gradBlock) =>
          require(binVecBlock.size == gradBlock.size)
          binVecBlock.iterator
            .zip(gradBlock.iterator)
            .map { case (binVec, grad) => (binVec, treeIds, grad) }
        }

      } else {
        Iterator.empty
      }
    }.setName(s"Gradients with TreeIds (iteration $iteration) (Partition-Based Sampled)")
  }


  def adaptTreeInputsForBlockSampling[T, N, C, B, H](weightLabelBlocks: RDD[(CompactArray[H], ArrayBlock[H])],
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
                                                     ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[(KVVector[C, B], Array[T], Array[H])] = {
    val rawSize = boostConf.getRawSize
    val numBaseModels = boostConf.getBaseModelParallelism
    val seedOffset = boostConf.getSeed + iteration
    val subSample = boostConf.getSubSampleRate

    val gradBlocks = weightLabelBlocks.zip(rawBlocks).mapPartitionsWithIndex { case (partId, iter) =>
      val rng = new Random(seedOffset + partId)
      val rngs = Array.tabulate(numBaseModels)(_ => new XORShiftRandom(rng.nextLong))

      val emptyValue = (ArrayBlock.empty[H], net.emptyArray)

      iter.map { case ((weightBlock, labelBlock), rawBlock) =>
        val baseIds = Array.range(0, numBaseModels)
          .filter { i => rngs(i).nextDouble < subSample }.map(int.fromInt)

        if (baseIds.nonEmpty) {
          val gradBlock = computeGradBlock(weightBlock, labelBlock, rawBlock)
          (gradBlock, baseIds)
        } else {
          emptyValue
        }
      }
    }

    gradBlocks.setName(s"GradientBlocks with BaseModelIds (iteration $iteration)")
    gradBlocks.persist(boostConf.getStorageLevel)
    recoder.append(gradBlocks)


    val computeTreeIds = if (rawSize == 1) {
      baseIds: Array[T] => baseIds
    } else {
      baseId: Array[T] =>
        baseId.flatMap { i =>
          val offset = rawSize * int.toInt(i)
          Iterator.range(offset, offset + rawSize).map(int.fromInt)
        }
    }

    binVecBlocks.zip(gradBlocks).flatMap { case (binVecBlock, (gradBlock, baseIds)) =>
      if (baseIds.nonEmpty) {
        require(binVecBlock.size == gradBlock.size)
        val treeIds = computeTreeIds(baseIds)
        binVecBlock.iterator.zip(gradBlock.iterator)
          .map { case (binVec, grad) => (binVec, treeIds, grad) }

      } else {
        require(gradBlock.isEmpty)
        Iterator.empty
      }
    }.setName(s"Gradients with TreeIds (iteration $iteration) (Block-Based Sampled)")
  }


  def adaptTreeInputsForInstanceSampling[T, N, C, B, H](weightLabelBlocks: RDD[(CompactArray[H], ArrayBlock[H])],
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
                                                        ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[(KVVector[C, B], Array[T], Array[H])] = {
    val rawSize = boostConf.getRawSize
    val numBaseModels = boostConf.getBaseModelParallelism
    val seedOffset = boostConf.getSeed + iteration
    val subSample = boostConf.getSubSampleRate

    val gradBlocks = weightLabelBlocks.zip(rawBlocks)
      .mapPartitionsWithIndex { case (partId, iter) =>
        val rng = new Random(seedOffset + partId)
        val rngs = Array.tabulate(numBaseModels)(_ => new XORShiftRandom(rng.nextLong))

        val emptyValue = (neh.emptyArray, net.emptyArray)

        iter.map { case ((weightBlock, labelBlock), rawBlock) =>
          require(weightBlock.size == rawBlock.size)
          require(labelBlock.size == rawBlock.size)

          val seq = weightBlock.iterator.zip(labelBlock.iterator).zip(rawBlock.iterator)
            .map { case ((weight, label), rawSeq) =>
              val baseIds = Array.range(0, numBaseModels)
                .filter { i => rngs(i).nextDouble < subSample }.map(int.fromInt)

              if (baseIds.nonEmpty) {
                val grad = computeGrad(weight, label, rawSeq)
                (grad, baseIds)
              } else {
                emptyValue
              }
            }.toSeq

          val gradBlock = ArrayBlock.build[H](seq.iterator.map(_._1))
          val baseIdBlock = ArrayBlock.build[T](seq.iterator.map(_._2))

          (gradBlock, baseIdBlock)
        }
      }

    gradBlocks.setName(s"GradientBlocks with baseIdBlocks (iteration $iteration)")
    gradBlocks.persist(boostConf.getStorageLevel)
    recoder.append(gradBlocks)


    val computeTreeIds = if (rawSize == 1) {
      baseId: Array[T] => baseId
    } else {
      baseId: Array[T] =>
        baseId.flatMap { i =>
          val offset = rawSize * int.toInt(i)
          Iterator.range(offset, offset + rawSize).map(int.fromInt)
        }
    }

    binVecBlocks.zip(gradBlocks).flatMap { case (binVecBlock, (gradBlock, baseIdBlock)) =>
      require(binVecBlock.size == gradBlock.size)
      require(binVecBlock.size == baseIdBlock.size)

      binVecBlock.iterator.zip(baseIdBlock.iterator).zip(gradBlock.iterator)
        .flatMap { case ((binVec, baseIds), grad) =>
          if (baseIds.nonEmpty) {
            val treeIds = computeTreeIds(baseIds)
            Iterator.single(binVec, treeIds, grad)
          } else {
            Iterator.empty
          }
        }
    }.setName(s"Gradients with TreeIds (iteration $iteration) (Instance-Based Sampled)")
  }
}


