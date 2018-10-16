package org.apache.spark.ml.gbm

import scala.collection.{BitSet, mutable}
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel


/**
  * RDD-based API for gradient boosting machine
  */
class GBM extends Logging with Serializable {

  val boostConf = new BoostConfig

  /** maximum number of iterations */
  def setMaxIter(value: Int): this.type = {
    boostConf.setMaxIter(value)
    this
  }

  def getMaxIter: Int = boostConf.getMaxIter


  /** maximum tree depth */
  def setMaxDepth(value: Int): this.type = {
    boostConf.setMaxDepth(value)
    this
  }

  def getMaxDepth: Int = boostConf.getMaxDepth


  /** maximum number of tree leaves */
  def setMaxLeaves(value: Int): this.type = {
    boostConf.setMaxLeaves(value)
    this
  }

  def getMaxLeaves: Int = boostConf.getMaxLeaves


  /** minimum gain for each split */
  def setMinGain(value: Double): this.type = {
    boostConf.setMinGain(value)
    this
  }

  def getMinGain: Double = boostConf.getMinGain


  /** base score for global bias */
  def setBaseScore(value: Array[Double]): this.type = {
    boostConf.setBaseScore(value)
    this
  }

  def getBaseScore: Array[Double] = boostConf.getBaseScore


  /** minimum sum of hess for each node */
  def setMinNodeHess(value: Double): this.type = {
    boostConf.setMinNodeHess(value)
    this
  }

  def getMinNodeHess: Double = boostConf.getMinNodeHess


  /** learning rate */
  def setStepSize(value: Double): this.type = {
    boostConf.setStepSize(value)
    this
  }

  def getStepSize: Double = boostConf.getStepSize


  /** L1 regularization term on weights */
  def setRegAlpha(value: Double): this.type = {
    boostConf.setRegAlpha(value)
    this
  }

  def getRegAlpha: Double = boostConf.getRegAlpha


  /** L2 regularization term on weights */
  def setRegLambda(value: Double): this.type = {
    boostConf.setRegLambda(value)
    this
  }

  def getRegLambda: Double = boostConf.getRegLambda


  /** objective function */
  def setObjFunc(value: ObjFunc): this.type = {
    boostConf.setObjFunc(value)
    this
  }

  def getObjFunc: ObjFunc = boostConf.getObjFunc


  /** evaluation functions */
  def setEvalFunc(value: Array[EvalFunc]): this.type = {
    boostConf.setEvalFunc(value)
    this
  }

  def getEvalFunc: Array[EvalFunc] = boostConf.getEvalFunc


  /** callback functions */
  def setCallbackFunc(value: Array[CallbackFunc]): this.type = {
    boostConf.setCallbackFunc(value)
    this
  }

  def getCallbackFunc: Array[CallbackFunc] = boostConf.getCallbackFunc


  /** indices of categorical columns */
  def setCatCols(value: Set[Int]): this.type = {
    require(value.forall(_ >= 0))
    val builder = BitSet.newBuilder
    builder ++= value
    boostConf.setCatCols(builder.result)
    this
  }

  def getCatCols: Set[Int] = boostConf.getCatCols.toSet


  /** indices of ranking columns */
  def setRankCols(value: Set[Int]): this.type = {
    require(value.forall(_ >= 0))
    val builder = BitSet.newBuilder
    builder ++= value
    boostConf.setRankCols(builder.result)
    this
  }

  def getRankCols: Set[Int] = boostConf.getRankCols.toSet


  /** subsample ratio of the training instance */
  def setSubSampleRate(value: Double): this.type = {
    boostConf.setSubSampleRate(value)
    this
  }

  def getSubSampleRate: Double = boostConf.getSubSampleRate


  /** subsample ratio of columns when constructing each tree */
  def setColSampleRateByTree(value: Double): this.type = {
    boostConf.setColSampleRateByTree(value)
    this
  }

  def getColSampleRateByTree: Double = boostConf.getColSampleRateByTree


  /** subsample ratio of columns when constructing each level */
  def setColSampleRateByLevel(value: Double): this.type = {
    boostConf.setColSampleRateByLevel(value)
    this
  }

  def getColSampleRateByLevel: Double = boostConf.getColSampleRateByLevel


  /** checkpoint interval */
  def setCheckpointInterval(value: Int): this.type = {
    boostConf.setCheckpointInterval(value)
    this
  }

  def getCheckpointInterval: Int = boostConf.getCheckpointInterval


  /** storage level */
  def setStorageLevel(value: StorageLevel): this.type = {
    boostConf.setStorageLevel(value)
    this
  }

  def getStorageLevel: StorageLevel = boostConf.getStorageLevel


  /** depth for treeAggregate */
  def setAggregationDepth(value: Int): this.type = {
    boostConf.setAggregationDepth(value)
    this
  }

  def getAggregationDepth: Int = boostConf.getAggregationDepth


  /** random number seed */
  def setSeed(value: Long): this.type = {
    boostConf.setSeed(value)
    this
  }

  def getSeed: Long = boostConf.getSeed


  /** parallelism of histogram computation */
  def setReduceParallelism(value: Double): this.type = {
    boostConf.setReduceParallelism(value)
    this
  }

  def getReduceParallelism: Double = boostConf.getReduceParallelism


  /** parallelism of split searching */
  def setTrialParallelism(value: Double): this.type = {
    boostConf.setTrialParallelism(value)
    this
  }

  def getTrialParallelism: Double = boostConf.getTrialParallelism


  /** parallelism type */
  def setParallelismType(value: String): this.type = {
    boostConf.setParallelismType(value)
    this
  }

  def getParallelismType: String = boostConf.getParallelismType


  /** boosting type */
  def setBoostType(value: String): this.type = {
    boostConf.setBoostType(value)
    this
  }

  def getBoostType: String = boostConf.getBoostType


  /** dropout rate */
  def setDropRate(value: Double): this.type = {
    boostConf.setDropRate(value)
    this
  }

  def getDropRate: Double = boostConf.getDropRate


  /** probability of skipping drop */
  def setDropSkip(value: Double): this.type = {
    boostConf.setDropSkip(value)
    this
  }

  def getDropSkip: Double = boostConf.getDropSkip

  /** minimum number of dropped trees in each iteration */
  def setMinDrop(value: Int): this.type = {
    boostConf.setMinDrop(value)
    this
  }

  def getMinDrop: Int = boostConf.getMinDrop


  /** maximum number of dropped trees in each iteration */
  def setMaxDrop(value: Int): this.type = {
    boostConf.setMaxDrop(value)
    this
  }

  def getMaxDrop: Int = boostConf.getMaxDrop


  /** retain fraction of large gradient data in GOSS */
  def setTopRate(value: Double): this.type = {
    boostConf.setTopRate(value)
    this
  }

  def getTopRate: Double = boostConf.getTopRate


  /** retain fraction of small gradient data in GOSS */
  def setOtherRate(value: Double): this.type = {
    boostConf.setOtherRate(value)
    this
  }

  def getOtherRate: Double = boostConf.getOtherRate


  /** the maximum number of non-zero histogram bins to search split for categorical columns by brute force */
  def setMaxBruteBins(value: Int): this.type = {
    boostConf.setMaxBruteBins(value)
    this
  }

  def getMaxBruteBins: Int = boostConf.getMaxBruteBins

  /** Double precision to represent internal gradient, hessian and prediction */
  def setFloatType(value: String): this.type = {
    boostConf.setFloatType(value)
    this
  }

  def getFloatType: String = boostConf.getFloatType


  /** number of base models in one round */
  def setBaseModelParallelism(value: Int): this.type = {
    boostConf.setBaseModelParallelism(value)
    this
  }

  def getBaseModelParallelism: Int = boostConf.getBaseModelParallelism


  /** method of data sampling */
  def setSubSampleType(value: String): this.type = {
    boostConf.setSubSampleType(value)
    this
  }

  def getSubSampleType: String = boostConf.getSubSampleType


  /** method to compute histograms */
  def setHistogramComputationType(value: String): this.type = {
    boostConf.setHistogramComputationType(value)
    this
  }

  def getHistogramComputationType: String = boostConf.getHistogramComputationType


  /** size of block */
  def setBlockSize(value: Int): this.type = {
    boostConf.setBlockSize(value)
    this
  }

  def getBlockSize: Int = boostConf.getBlockSize


  /** initial model */
  private var initialModel: Option[GBMModel] = None

  def setInitialModel(value: Option[GBMModel]): this.type = {
    initialModel = value
    this
  }

  def getInitialModel: Option[GBMModel] = initialModel


  /** maximum number of bins for each column */
  private var maxBins: Int = 64

  def setMaxBins(value: Int): this.type = {
    require(value >= 4)
    maxBins = value
    this
  }

  def getMaxBins: Int = maxBins


  /** method to discretize numerical columns */
  private var numericalBinType: String = GBM.Width

  def setNumericalBinType(value: String): this.type = {
    require(value == GBM.Width || value == GBM.Depth)
    numericalBinType = value
    this
  }

  def getNumericalBinType: String = numericalBinType


  /** whether zero is viewed as missing value */
  private var zeroAsMissing: Boolean = false

  def setZeroAsMissing(value: Boolean): this.type = {
    zeroAsMissing = value
    this
  }

  def getZeroAsMissing: Boolean = zeroAsMissing


  /** training, dataset contains (weight, label, vec) */
  def fit(data: RDD[(Double, Array[Double], Vector)]): GBMModel = {
    fit(data, None)
  }


  /** training with validation, dataset contains (weight, label, vec) */
  def fit(data: RDD[(Double, Array[Double], Vector)],
          test: RDD[(Double, Array[Double], Vector)]): GBMModel = {
    fit(data, Some(test))
  }


  /** training with validation if any, dataset contains (weight, label, vec) */
  private[ml] def fit(data: RDD[(Double, Array[Double], Vector)],
                      test: Option[RDD[(Double, Array[Double], Vector)]]): GBMModel = {
    if (getBoostType == GBM.Dart) {
      require(getMaxDrop >= getMinDrop)
    } else if (getBoostType == GBM.Goss) {
      require(getTopRate + getOtherRate <= 1)
    }

    val sc = data.sparkContext

    val numCols = data.first._3.size
    require(numCols > 0)

    var labelAvg = Array.emptyDoubleArray
    var discretizer = null.asInstanceOf[Discretizer]

    if (initialModel.nonEmpty) {
      require(numCols == initialModel.get.discretizer.numCols)

      discretizer = initialModel.get.discretizer
      logWarning(s"Discretizer is already provided in the initial model, related params are ignored: " +
        s"maxBins,catCols,rankCols,numericalBinType,zeroAsMissing")

      val baseScore_ = initialModel.get.baseScore
      boostConf.setBaseScore(baseScore_)
      logWarning(s"BaseScore is already provided in the initial model, related param is overridden: " +
        s"${boostConf.getBaseScore.mkString(",")} -> ${baseScore_.mkString(",")}")

    } else {
      val t = Discretizer.fit2(data, numCols, boostConf.getCatCols, boostConf.getRankCols,
        maxBins, numericalBinType, zeroAsMissing, getAggregationDepth)
      discretizer = t._1
      labelAvg = t._2
    }

    if (boostConf.getBaseScore.isEmpty) {
      logInfo(s"Basescore is not provided, assign it to average label value " +
        s"${boostConf.getBaseScore.mkString(",")}")
      boostConf.setBaseScore(labelAvg)
    }

    val rawBase = boostConf.computeRawBaseScore
    logInfo(s"base score vector: ${boostConf.getBaseScore.mkString(",")}, " +
      s"raw base vector: ${rawBase.mkString(",")}")

    boostConf
      .setNumCols(numCols)
      .setRawSize(rawBase.length)

    GBM.boost(data, test, boostConf, discretizer, initialModel)
  }
}


private[gbm] object GBM extends Logging {

  val Data = "data"
  val Feature = "feature"

  val GBTree = "gbtree"
  val Dart = "dart"

  val Instance = "instance"
  val Block = "block"
  val Partition = "partition"
  val Goss = "goss"

  val Width = "width"
  val Depth = "depth"

  val SinglePrecision = "float"
  val DoublePrecision = "double"


  /**
    * train a GBM model, dataset contains (weight, label, vec)
    */
  def boost(data: RDD[(Double, Array[Double], Vector)],
            test: Option[RDD[(Double, Array[Double], Vector)]],
            boostConf: BoostConfig,
            discretizer: Discretizer,
            initialModel: Option[GBMModel]): GBMModel = {
    val sc = data.sparkContext
    Utils.registerKryoClasses(sc)

    logInfo(s"DataType of RealValue: ${boostConf.getFloatType.capitalize}")

    boostConf.getFloatType match {
      case SinglePrecision =>
        boost1[Double](data, test, boostConf, discretizer, initialModel)

      case DoublePrecision =>
        boost1[Double](data, test, boostConf, discretizer, initialModel)
    }
  }


  /**
    * train a GBM model, dataset contains (weight, label, vec)
    */
  def boost1[H](data: RDD[(Double, Array[Double], Vector)],
                test: Option[RDD[(Double, Array[Double], Vector)]],
                boostConf: BoostConfig,
                discretizer: Discretizer,
                initialModel: Option[GBMModel])
               (implicit ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): GBMModel = {

    val data2 = data.map {
      case (weight, label, vec) =>
        (neh.fromDouble(weight), neh.fromDouble(label), vec)
    }

    val test2 = test.map {
      _.map { case (weight, label, vec) =>
        (neh.fromDouble(weight), neh.fromDouble(label), vec)
      }
    }

    val columnIndexType = Utils.getTypeByRange(discretizer.numCols)
    logInfo(s"DataType of ColumnId: $columnIndexType")

    val binType = Utils.getTypeByRange(discretizer.numBins.max * 2 - 1)
    logInfo(s"DataType of Bin: $binType")

    (columnIndexType, binType) match {
      case ("Byte", "Byte") =>
        boost2[Byte, Byte, H](data2, test2, boostConf, discretizer, initialModel)

      case ("Byte", "Short") =>
        boost2[Byte, Short, H](data2, test2, boostConf, discretizer, initialModel)

      case ("Byte", "Int") =>
        boost2[Byte, Int, H](data2, test2, boostConf, discretizer, initialModel)

      case ("Short", "Byte") =>
        boost2[Short, Byte, H](data2, test2, boostConf, discretizer, initialModel)

      case ("Short", "Short") =>
        boost2[Short, Short, H](data2, test2, boostConf, discretizer, initialModel)

      case ("Short", "Int") =>
        boost2[Short, Int, H](data2, test2, boostConf, discretizer, initialModel)

      case ("Int", "Byte") =>
        boost2[Int, Byte, H](data2, test2, boostConf, discretizer, initialModel)

      case ("Int", "Short") =>
        boost2[Int, Short, H](data2, test2, boostConf, discretizer, initialModel)

      case ("Int", "Int") =>
        boost2[Int, Int, H](data2, test2, boostConf, discretizer, initialModel)
    }
  }


  /**
    * train a GBM model, dataset contains (weight, label, vec)
    */
  def boost2[C, B, H](data: RDD[(H, Array[H], Vector)],
                      test: Option[RDD[(H, Array[H], Vector)]],
                      boostConf: BoostConfig,
                      discretizer: Discretizer,
                      initialModel: Option[GBMModel])
                     (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                      cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                      ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): GBMModel = {

    val sc = data.sparkContext

    val recoder = new ResourceRecoder

    val bcDiscretizer = sc.broadcast(discretizer)
    recoder.append(bcDiscretizer)

    val trainBlocks = discretizeAndblockify[C, B, H](data, bcDiscretizer, boostConf.getBlockSize)

    val (trainweightLabelBlocks, trainBinVecBlocks) = trainBlocks

    trainweightLabelBlocks.setName("Train Weight+Label Blocks")
    trainweightLabelBlocks.persist(boostConf.getStorageLevel)
    recoder.append(trainweightLabelBlocks)

    trainBinVecBlocks.setName("Train BinVector Blocks")
    trainBinVecBlocks.persist(boostConf.getStorageLevel)
    recoder.append(trainBinVecBlocks)


    val testBlocks = test.map { rdd => discretizeAndblockify[C, B, H](rdd, bcDiscretizer, boostConf.getBlockSize) }

    testBlocks.foreach { case (testweightLabelBlocks, testBinVecBlocks) =>
      testweightLabelBlocks.setName("Test Weight+Label Blocks")
      testweightLabelBlocks.persist(boostConf.getStorageLevel)
      recoder.append(testweightLabelBlocks)

      testBinVecBlocks.setName("Test BinVector Blocks")
      testBinVecBlocks.persist(boostConf.getStorageLevel)
      recoder.append(testBinVecBlocks)
    }


    val model = boostConf.getParallelismType match {

      case Data => HorizontalGBM.boost[C, B, H](trainBlocks, testBlocks, boostConf, discretizer, initialModel)

      case Feature => VerticalGBM.boost[C, B, H](trainBlocks, testBlocks, boostConf, discretizer, initialModel)
    }

    recoder.clear()

    model
  }

  /**
    * discretize and blockify instances to weightAndLabel-blocks and binVec-blocks.
    */
  def discretizeAndblockify[C, B, H](data: RDD[(H, Array[H], Vector)],
                                     bcDiscretizer: Broadcast[Discretizer],
                                     blockSize: Int)
                                    (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                     cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                     ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): (RDD[(CompactArray[H], ArrayBlock[H])], RDD[KVMatrix[C, B]]) = {

    val weightLabelBlocks = data.mapPartitions {
      _.map(t => (t._1, t._2)).grouped(blockSize)
        .map { seq =>
          val weightBlock = CompactArray.build[H](seq.iterator.map(_._1))
          val labelBlock = ArrayBlock.build[H](seq.iterator.map(_._2))
          require(weightBlock.size == labelBlock.size)
          (weightBlock, labelBlock)
        }
    }

    val binVecBlocks = data.mapPartitions { iter =>
      val discretizer = bcDiscretizer.value
      iter.map(_._3).map(discretizer.transformToGBMVector[C, B])
        .grouped(blockSize)
        .map(KVMatrix.build[C, B])
    }

    (weightLabelBlocks, binVecBlocks)
  }


  def touchBlocksAndUpdateSizeInfo[H](weightLabelBlocks: RDD[(CompactArray[H], ArrayBlock[H])],
                                      boostConf: BoostConfig): Unit = {

    val array = weightLabelBlocks
      .mapPartitionsWithIndex { case (partId, iter) =>
        var numBlocks = 0L
        var numInstances = 0L
        iter.foreach { case (weightBlock, _) =>
          numBlocks += 1
          numInstances += weightBlock.size
        }
        Iterator.single((partId, numBlocks, numInstances))
      }.collect().sorted

    require(array.length == weightLabelBlocks.getNumPartitions)

    boostConf.setNumBlocksPerPartition(array.map(_._2))
    boostConf.setNumInstancesPerPartition(array.map(_._3))
    logInfo(s"${weightLabelBlocks.name}: ${boostConf.getNumInstances} instances, ${boostConf.getNumBlocks} blocks, " +
      s"numInstancesPerPartition ${boostConf.getNumInstancesPerPartition.mkString("[", ",", "]")}" +
      s"numBlocksPerPartition ${boostConf.getNumBlocksPerPartition.mkString("[", ",", "]")}")
  }


  def touchBlocks[H](weightLabelBlocks: RDD[(CompactArray[H], ArrayBlock[H])],
                     boostConf: BoostConfig): Unit = {
    val (numInstances, numBlocks) = weightLabelBlocks.map { case (weightBlock, _) =>
      (weightBlock.size.toLong, 1L)
    }.treeReduce(f = {
      case (t1, t2) => (t1._1 + t2._1, t1._2 + t2._2)
    }, depth = boostConf.getAggregationDepth)
    logInfo(s"${weightLabelBlocks.name}: $numInstances instances, $numBlocks blocks")
  }


  /**
    * append new tree to the model buffer
    *
    * @param weights   weights of trees
    * @param treeBuff  trees
    * @param trees     tree to be appended
    * @param dropped   indices of dropped trees
    * @param boostConf boosting configuration
    */
  def updateTreeBuffer[H](weights: mutable.ArrayBuffer[H],
                          treeBuff: mutable.ArrayBuffer[TreeModel],
                          trees: Array[TreeModel],
                          dropped: Set[Int],
                          boostConf: BoostConfig)
                         (implicit ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Unit = {
    import nuh._

    treeBuff.appendAll(trees)

    boostConf.getBoostType match {
      case GBTree =>
        weights.appendAll(Iterator.fill(trees.length)(neh.fromDouble(boostConf.getStepSize)))

      case Dart if dropped.isEmpty =>
        weights.appendAll(Iterator.fill(trees.length)(nuh.one))

      case Dart if dropped.nonEmpty =>
        require(dropped.size % boostConf.getRawSize == 0)
        // number of droped base models
        val k = dropped.size / boostConf.getRawSize
        val w = neh.fromDouble(1 / (k + boostConf.getStepSize))
        weights.appendAll(Iterator.fill(trees.length)(w))
        val scale = neh.fromDouble(k / (k + boostConf.getStepSize))

        val updateStrBuilder = mutable.ArrayBuilder.make[String]
        dropped.foreach { i =>
          val newWeight = weights(i) * scale
          updateStrBuilder += s"Tree $i: ${weights(i)} -> $newWeight"
          weights(i) = newWeight
        }

        logInfo(s"Weights updated : ${updateStrBuilder.result.mkString("(", ",", ")")}")
    }
  }


  /**
    * drop trees
    *
    * @param dropped   indices of dropped trees
    * @param boostConf boosting configuration
    * @param numTrees  number of trees
    * @param dartRng   random number generator
    */
  def dropTrees(dropped: mutable.Set[Int],
                boostConf: BoostConfig,
                numTrees: Int,
                dartRng: Random): Unit = {
    dropped.clear

    if (boostConf.getDropSkip < 1 &&
      dartRng.nextDouble < 1 - boostConf.getDropSkip) {

      require(numTrees % boostConf.getRawSize == 0)
      val numBaseModels = numTrees / boostConf.getRawSize

      var k = (numBaseModels * boostConf.getDropRate).ceil.toInt
      k = math.max(k, boostConf.getMinDrop)
      k = math.min(k, boostConf.getMaxDrop)
      k = math.min(k, numBaseModels)

      if (k > 0) {
        dartRng.shuffle(Seq.range(0, numBaseModels)).take(k)
          .flatMap { i => Iterator.range(boostConf.getRawSize * i, boostConf.getRawSize * (i + 1)) }
          .foreach(dropped.add)
      }
    }
  }


  /**
    * initialize prediction of instances, containing the final score and the scores of each tree.
    *
    * @param binVecBlocks bin vectors
    * @param trees        array of trees
    * @param weights      array of weights
    * @param boostConf    boosting configuration
    * @return RDD containing final score (weighted) and the scores of each tree (non-weighted, only for DART)
    */
  def initializeRawBlocks[C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                                   trees: Array[TreeModel],
                                   weights: Array[H],
                                   boostConf: BoostConfig)
                                  (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                   cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                   ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[ArrayBlock[H]] = {
    import nuh._

    require(trees.length == weights.length)
    require(trees.length % boostConf.getRawSize == 0)

    val rawBase = neh.fromDouble(boostConf.computeRawBaseScore)
    val rawSize = boostConf.getRawSize
    require(rawSize == rawBase.length)

    if (trees.nonEmpty) {
      boostConf.getBoostType match {
        case Dart =>
          binVecBlocks.map { binVecBlock =>
            val iter = binVecBlock.iterator.map { binVec =>
              val raw = rawBase.clone ++ trees.map(tree => neh.fromDouble(tree.predict(binVec.apply)))

              var j = 0
              while (j < trees.length) {
                val p = raw(rawSize + j)
                raw(j % rawSize) += p * weights(j)
                j += 1
              }
              raw
            }

            ArrayBlock.build[H](iter)
          }

        case _ =>
          binVecBlocks.map { binVecBlock =>
            val iter = binVecBlock.iterator.map { binVec =>
              val raw = rawBase.clone
              var j = 0
              while (j < trees.length) {
                val p = neh.fromDouble(trees(j).predict(binVec.apply))
                raw(j % rawSize) += p * weights(j)
                j += 1
              }
              raw
            }

            ArrayBlock.build[H](iter)
          }
      }

    } else {

      binVecBlocks.mapPartitions { iter =>
        val defaultRawBlock = ArrayBlock.fill[H](rawBase, boostConf.getBlockSize)
        iter.map { binVecBlock =>
          if (binVecBlock.size == defaultRawBlock.size) {
            defaultRawBlock
          } else {
            ArrayBlock.build[H](Iterator.range(0, binVecBlock.size).map(_ => rawBase))
          }
        }
      }
    }
  }


  /**
    * update prediction of instances, containing the final score and the predictions of each tree.
    *
    * @param binVecBlocks bin vectors
    * @param rawBlocks    previous raw predictions
    * @param newTrees     array of trees (new built)
    * @param weights      array of weights (total = old ++ new)
    * @param boostConf    boosting configuration
    * @param keepWeights  whether to keep the weights of previous trees
    * @return RDD containing final score and the predictions of each tree
    */
  def updateRawBlocks[C, B, H](binVecBlocks: RDD[KVMatrix[C, B]],
                               rawBlocks: RDD[ArrayBlock[H]],
                               newTrees: Array[TreeModel],
                               weights: Array[H],
                               boostConf: BoostConfig,
                               keepWeights: Boolean)
                              (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                               cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                               ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[ArrayBlock[H]] = {
    import nuh._

    require(newTrees.length % boostConf.getRawSize == 0)
    require(weights.length % boostConf.getRawSize == 0)

    val rawSize = boostConf.getRawSize
    val treeOffset = weights.length - newTrees.length

    boostConf.getBoostType match {
      case Dart =>
        val rawBase = neh.fromDouble(boostConf.computeRawBaseScore)
        require(rawSize == rawBase.length)

        binVecBlocks.zip(rawBlocks).map { case (binVecBlock, rawBlock) =>
          require(rawBlock.size == binVecBlock.size)

          val iter = binVecBlock.iterator
            .zip(rawBlock.iterator)
            .map { case (binVec, raw) =>
              require(raw.length == rawSize + treeOffset)
              val newRaw = raw ++ newTrees.map(tree => neh.fromDouble(tree.predict(binVec.apply)))

              if (keepWeights) {
                var j = 0
                while (j < newTrees.length) {
                  val p = newRaw(rawSize + treeOffset + j)
                  newRaw(j % rawSize) += p * weights(treeOffset + j)
                  j += 1
                }

              } else {
                Array.copy(rawBase, 0, newRaw, 0, rawSize)
                var j = 0
                while (j < weights.length) {
                  val p = newRaw(rawSize + j)
                  newRaw(j % rawSize) += p * weights(j)
                  j += 1
                }
              }

              newRaw
            }

          ArrayBlock.build[H](iter)
        }

      case _ =>
        binVecBlocks.zip(rawBlocks).map { case (binVecBlock, rawBlock) =>
          require(rawBlock.size == binVecBlock.size)

          val iter = binVecBlock.iterator
            .zip(rawBlock.iterator)
            .map { case (binVec, raw) =>
              require(raw.length == rawSize)
              var j = 0
              while (j < newTrees.length) {
                val p = neh.fromDouble(newTrees(j).predict(binVec.apply))
                raw(j % rawSize) += p * weights(treeOffset + j)
                j += 1
              }
              raw
            }

          ArrayBlock.build[H](iter)
        }
    }
  }


  /**
    * Evaluate current model and output the result
    *
    * @param weightLabelBlocks weights and labels
    * @param rawBlocks         prediction of instances, containing the final score and the scores of each tree
    * @param boostConf         boosting configuration containing the evaluation functions
    * @return Evaluation result with names as the keys and metrics as the values
    */
  def evaluate[H, C, B](weightLabelBlocks: RDD[(CompactArray[H], ArrayBlock[H])],
                        rawBlocks: RDD[ArrayBlock[H]],
                        boostConf: BoostConfig)
                       (implicit cc: ClassTag[C], inc: Integral[C],
                        cb: ClassTag[B], inb: Integral[B],
                        ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Map[String, Double] = {
    if (boostConf.getEvalFunc.isEmpty) {
      return Map.empty
    }

    val rawSize = boostConf.getRawSize

    val scores = weightLabelBlocks.zip(rawBlocks)
      .flatMap { case ((weightBlock, labelBlock), rawBlock) =>
        require(weightBlock.size == rawBlock.size)
        require(labelBlock.size == rawBlock.size)

        weightBlock.iterator.zip(labelBlock.iterator).zip(rawBlock.iterator)
          .map { case ((weight, label), rawSeq) =>
            val raw = neh.toDouble(rawSeq.take(rawSize))
            val score = boostConf.getObjFunc.transform(raw)
            (nuh.toDouble(weight), neh.toDouble(label), raw, score)
          }
      }

    val result = mutable.OpenHashMap.empty[String, Double]

    // persist if there are batch evaluators
    if (boostConf.getBatchEvalFunc.nonEmpty) {
      scores.setName(s"Evaluation Dataset (weight, label, raw, score)")
      scores.persist(boostConf.getStorageLevel)
    }

    if (boostConf.getIncEvalFunc.nonEmpty) {
      IncEvalFunc.compute(scores,
        boostConf.getIncEvalFunc, boostConf.getAggregationDepth)
        .foreach { case (name, value) => result.update(name, value) }
    }

    if (boostConf.getBatchEvalFunc.nonEmpty) {
      boostConf.getBatchEvalFunc
        .foreach { eval => result.update(eval.name, eval.compute(scores)) }
      scores.unpersist(blocking = false)
    }

    result.toMap
  }
}





