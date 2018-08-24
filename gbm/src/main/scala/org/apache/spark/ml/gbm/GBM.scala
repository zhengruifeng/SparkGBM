package org.apache.spark.ml.gbm

import scala.collection.{BitSet, mutable}
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.XORShiftRandom


/**
  * RDD-based API for gradient boosting machine
  */
class GBM extends Logging with Serializable {

  val boostConf = new BoostConfig

  /** maximum number of iterations */
  def setMaxIter(value: Int): this.type = {
    require(value >= 0)
    boostConf.setMaxIter(value)
    this
  }

  def getMaxIter: Int = boostConf.getMaxIter


  /** maximum tree depth */
  def setMaxDepth(value: Int): this.type = {
    require(value >= 1 && value <= 30)
    boostConf.setMaxDepth(value)
    this
  }

  def getMaxDepth: Int = boostConf.getMaxDepth


  /** maximum number of tree leaves */
  def setMaxLeaves(value: Int): this.type = {
    require(value >= 2)
    boostConf.setMaxLeaves(value)
    this
  }

  def getMaxLeaves: Int = boostConf.getMaxLeaves


  /** minimum gain for each split */
  def setMinGain(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    boostConf.setMinGain(value)
    this
  }

  def getMinGain: Double = boostConf.getMinGain


  /** base score for global bias */
  def setBaseScore(value: Array[Double]): this.type = {
    require(value.nonEmpty)
    require(value.forall(v => !v.isNaN && !v.isInfinity))
    boostConf.setBaseScore(value)
    this
  }

  def getBaseScore: Array[Double] = boostConf.getBaseScore


  /** minimum sum of hess for each node */
  def setMinNodeHess(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    boostConf.setMinNodeHess(value)
    this
  }

  def getMinNodeHess: Double = boostConf.getMinNodeHess


  /** learning rate */
  def setStepSize(value: Double): this.type = {
    require(value > 0 && !value.isNaN && !value.isInfinity)
    boostConf.setStepSize(value)
    this
  }

  def getStepSize: Double = boostConf.getStepSize


  /** L1 regularization term on weights */
  def setRegAlpha(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    boostConf.setRegAlpha(value)
    this
  }

  def getRegAlpha: Double = boostConf.getRegAlpha


  /** L2 regularization term on weights */
  def setRegLambda(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    boostConf.setRegLambda(value)
    this
  }

  def getRegLambda: Double = boostConf.getRegLambda


  /** objective function */
  def setObjFunc(value: ObjFunc): this.type = {
    require(value != null)
    boostConf.setObjFunc(value)
    this
  }

  def getObjFunc: ObjFunc = boostConf.getObjFunc


  /** evaluation functions */
  def setEvalFunc(value: Array[EvalFunc]): this.type = {
    require(value.map(_.name).distinct.length == value.length)
    boostConf.setEvalFunc(value)
    this
  }

  def getEvalFunc: Array[EvalFunc] = boostConf.getEvalFunc


  /** callback functions */
  def setCallbackFunc(value: Array[CallbackFunc]): this.type = {
    require(value.map(_.name).distinct.length == value.length)
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
  def setSubSample(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    boostConf.setSubSample(value)
    this
  }

  def getSubSample: Double = boostConf.getSubSample


  /** subsample ratio of columns when constructing each tree */
  def setColSampleByTree(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    boostConf.setColSampleByTree(value)
    this
  }

  def getColSampleByTree: Double = boostConf.getColSampleByTree


  /** subsample ratio of columns when constructing each level */
  def setColSampleByLevel(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    boostConf.setColSampleByLevel(value)
    this
  }

  def getColSampleByLevel: Double = boostConf.getColSampleByLevel


  /** checkpoint interval */
  def setCheckpointInterval(value: Int): this.type = {
    require(value == -1 || value > 0)
    boostConf.setCheckpointInterval(value)
    this
  }

  def getCheckpointInterval: Int = boostConf.getCheckpointInterval


  /** storage level */
  def setStorageLevel(value: StorageLevel): this.type = {
    require(value != StorageLevel.NONE)
    boostConf.setStorageLevel(value)
    this
  }

  def getStorageLevel: StorageLevel = boostConf.getStorageLevel


  /** depth for treeAggregate */
  def setAggregationDepth(value: Int): this.type = {
    require(value >= 2)
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


  /** parallelism of histogram subtraction and split searching */
  def setParallelism(value: Int): this.type = {
    require(value != 0)
    boostConf.setParallelism(value)
    this
  }

  def getParallelism: Int = boostConf.getParallelism


  /** boosting type */
  def setBoostType(value: String): this.type = {
    require(value == GBM.GBTree || value == GBM.Dart)
    boostConf.setBoostType(value)
    this
  }

  def getBoostType: String = boostConf.getBoostType


  /** dropout rate */
  def setDropRate(value: Double): this.type = {
    require(value >= 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    boostConf.setDropRate(value)
    this
  }

  def getDropRate: Double = boostConf.getDropRate


  /** probability of skipping drop */
  def setDropSkip(value: Double): this.type = {
    require(value >= 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    boostConf.setDropSkip(value)
    this
  }

  def getDropSkip: Double = boostConf.getDropSkip

  /** minimum number of dropped trees in each iteration */
  def setMinDrop(value: Int): this.type = {
    require(value >= 0)
    boostConf.setMinDrop(value)
    this
  }

  def getMinDrop: Int = boostConf.getMinDrop


  /** maximum number of dropped trees in each iteration */
  def setMaxDrop(value: Int): this.type = {
    require(value >= 0)
    boostConf.setMaxDrop(value)
    this
  }

  def getMaxDrop: Int = boostConf.getMaxDrop


  /** the maximum number of non-zero histogram bins to search split for categorical columns by brute force */
  def setMaxBruteBins(value: Int): this.type = {
    require(value >= 0)
    boostConf.setMaxBruteBins(value)
    this
  }

  def getMaxBruteBins: Int = boostConf.getMaxBruteBins

  /** Double precision to represent internal gradient, hessian and prediction */
  def setFloatType(value: String): this.type = {
    require(value == GBM.SinglePrecision || value == GBM.DoublePrecision)
    boostConf.setFloatType(value)
    this
  }

  def getFloatType: String = boostConf.getFloatType


  /** number of base models in one round */
  def setBaseModelParallelism(value: Int): this.type = {
    require(value > 0)
    boostConf.setBaseModelParallelism(value)
    this
  }

  def getBaseModelParallelism: Int = boostConf.getBaseModelParallelism


  /** whether to sample partitions instead of instances if possible */
  def setEnableSamplePartitions(value: Boolean): this.type = {
    boostConf.setEnableSamplePartitions(value)
    this
  }

  def getEnableSamplePartitions: Boolean = boostConf.getEnableSamplePartitions


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
    }

    val sc = data.sparkContext

    val numCols = data.first._3.size
    require(numCols > 0)

    val validation = test.nonEmpty

    val discretizer = if (initialModel.isDefined) {
      require(numCols == initialModel.get.discretizer.colDiscretizers.length)
      logWarning(s"Discretizer is already provided by the initial model, related params" +
        s" (maxBins,catCols,rankCols,numericalBinType) will be ignored")
      initialModel.get.discretizer

    } else {
      require(getCatCols.forall(v => v >= 0 && v < numCols))
      require(getRankCols.forall(v => v >= 0 && v < numCols))
      require((getCatCols & getRankCols).isEmpty)

      Discretizer.fit(data.map(_._3), numCols, boostConf.getCatCols, boostConf.getRankCols,
        maxBins, numericalBinType, zeroAsMissing, getAggregationDepth)
    }
    logInfo(s"bins: ${discretizer.numBins.mkString(",").take(200)}")
    logInfo(s"Maximum bins: ${discretizer.numBins.max}")
    logInfo(s"Average bins: ${discretizer.numBins.sum.toDouble / discretizer.numBins.length}")
    logInfo(s"Sparsity of train data: ${discretizer.sparsity}")


    if (boostConf.getBaseScore.isEmpty) {
      val (weight, sum) = data.map { case (weight, label, _) =>
        (weight, label.map(_ * weight))
      }.treeReduce(f = {
        case ((weight1, sum1), (weight2, sum2) =>
          require(sum1.length == sum2.length)
          Iterator.range(0, sum1.length).foreach(i => sum1(i) += sum2(i))
          (weight1 + weight2, sum1)
      }, depth = boostConf.getAggregationDepth)

      boostConf.setBaseScore(sum.map(_ / weight))
    }


    boostConf
      .setNumCols(numCols)
      .setRawSize(boostConf.computeRawBaseScore.length)
    logInfo(s"Raw base vector: ${boostConf.computeRawBaseScore.mkString(",")}")

    GBM.boost(data, test.getOrElse(sc.emptyRDD), boostConf, validation, discretizer, initialModel)
  }
}


private[gbm] object GBM extends Logging {

  val GBTree = "gbtree"
  val Dart = "dart"

  val Width = "width"
  val Depth = "depth"

  val SinglePrecision = "Double"
  val DoublePrecision = "double"


  /**
    * train a GBM model, dataset contains (weight, label, vec)
    */
  def boost(data: RDD[(Double, Array[Double], Vector)],
            test: RDD[(Double, Array[Double], Vector)],
            boostConf: BoostConfig,
            validation: Boolean,
            discretizer: Discretizer,
            initialModel: Option[GBMModel]): GBMModel = {

    logInfo(s"DataType of RealValue: ${boostConf.getFloatType.capitalize}")

    boostConf.getFloatType match {
      case SinglePrecision =>
        boostWithFloatType[Double](data, test, boostConf, validation, discretizer, initialModel)

      case DoublePrecision =>
        boostWithFloatType[Double](data, test, boostConf, validation, discretizer, initialModel)
    }
  }


  /**
    * train a GBM model, with the given type of bins and column indices, dataset contains (weight, label, vec)
    */
  def boostWithFloatType[H](data: RDD[(Double, Array[Double], Vector)],
                            test: RDD[(Double, Array[Double], Vector)],
                            boostConf: BoostConfig,
                            validation: Boolean,
                            discretizer: Discretizer,
                            initialModel: Option[GBMModel])
                           (implicit ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): GBMModel = {
    logInfo(s"DataType of RealValue: ${boostConf.getFloatType.capitalize}")

    val data2 = data.map { case (weight, label, vec) =>
      (neh.fromDouble(weight), label.map(neh.fromDouble), vec)
    }

    val test2 = test.map { case (weight, label, vec) =>
      (neh.fromDouble(weight), label.map(neh.fromDouble), vec)
    }


    val columnIndexType = Utils.getTypeByRange(discretizer.numCols)
    logInfo(s"DataType of ColumnId: $columnIndexType")

    val binType = Utils.getTypeByRange(discretizer.numBins.max * 2 - 1)
    logInfo(s"DataType of Bin: $binType")

    (columnIndexType, binType) match {
      case ("Byte", "Byte") =>
        boostWithBinType[Byte, Byte, H](data2, test2, boostConf, validation, discretizer, initialModel)

      case ("Byte", "Short") =>
        boostWithBinType[Byte, Short, H](data2, test2, boostConf, validation, discretizer, initialModel)

      case ("Byte", "Int") =>
        boostWithBinType[Byte, Int, H](data2, test2, boostConf, validation, discretizer, initialModel)

      case ("Short", "Byte") =>
        boostWithBinType[Short, Byte, H](data2, test2, boostConf, validation, discretizer, initialModel)

      case ("Short", "Short") =>
        boostWithBinType[Short, Short, H](data2, test2, boostConf, validation, discretizer, initialModel)

      case ("Short", "Int") =>
        boostWithBinType[Short, Int, H](data2, test2, boostConf, validation, discretizer, initialModel)

      case ("Int", "Byte") =>
        boostWithBinType[Int, Byte, H](data2, test2, boostConf, validation, discretizer, initialModel)

      case ("Int", "Short") =>
        boostWithBinType[Int, Short, H](data2, test2, boostConf, validation, discretizer, initialModel)

      case ("Int", "Int") =>
        boostWithBinType[Int, Int, H](data2, test2, boostConf, validation, discretizer, initialModel)
    }
  }


  /**
    * train a GBM model, with the given type of bins and column indices, dataset contains (weight, label, vec)
    */
  def boostWithBinType[C, B, H](data: RDD[(H, Array[H], Vector)],
                                test: RDD[(H, Array[H], Vector)],
                                boostConf: BoostConfig,
                                validation: Boolean,
                                discretizer: Discretizer,
                                initialModel: Option[GBMModel])
                               (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): GBMModel = {
    val binData = data.map { case (weight, label, vec) =>
      (weight, label, discretizer.transformToGBMVector[C, B](vec))
    }

    val binTest = test.map { case (weight, label, vec) =>
      (weight, label, discretizer.transformToGBMVector[C, B](vec))
    }

    boostImpl[C, B, H](binData, binTest, boostConf, validation, discretizer, initialModel)
  }


  /**
    * implementation of GBM, train a GBMModel, with given types
    *
    * @param data         training instances containing (weight, label, bins)
    * @param test         validation instances containing (weight, label, bins)
    * @param boostConf    boosting configuration
    * @param validation   whether to validate on test data
    * @param discretizer  discretizer to convert raw features into bins
    * @param initialModel inital model
    * @return the model
    */
  def boostImpl[C, B, H](data: RDD[(H, Array[H], KVVector[C, B])],
                         test: RDD[(H, Array[H], KVVector[C, B])],
                         boostConf: BoostConfig,
                         validation: Boolean,
                         discretizer: Discretizer,
                         initialModel: Option[GBMModel])
                        (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                         cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                         ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): GBMModel = {
    val spark = SparkSession.builder.getOrCreate
    val sc = spark.sparkContext
    Utils.registerKryoClasses(sc)

    val rawBase = boostConf.computeRawBaseScore

    data.persist(boostConf.getStorageLevel)
    logInfo(s"${data.count} instances in train data")

    if (validation) {
      test.persist(boostConf.getStorageLevel)
      logInfo(s"${test.count} instances in test data")
    }

    val weightBuff = mutable.ArrayBuffer.empty[H]
    val treeBuff = mutable.ArrayBuffer.empty[TreeModel]

    if (initialModel.isDefined) {
      weightBuff.appendAll(initialModel.get.weights.map(w => neh.fromDouble(w)))
      treeBuff.appendAll(initialModel.get.trees)
    }

    val trainRawScoresCheckpointer = new Checkpointer[(Array[H], Array[H])](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel)

    var trainRawScores = computeRawScores[C, B, H](data, treeBuff.toArray, weightBuff.toArray, boostConf)
    trainRawScoresCheckpointer.update(trainRawScores)

    val testRawScoresCheckpointer = new Checkpointer[(Array[H], Array[H])](sc,
      boostConf.getCheckpointInterval, boostConf.getStorageLevel)

    var testRawScores = sc.emptyRDD[(Array[H], Array[H])]
    if (validation && boostConf.getEvalFunc.nonEmpty) {
      testRawScores = computeRawScores[C, B, H](test, treeBuff.toArray, weightBuff.toArray, boostConf)
      testRawScoresCheckpointer.update(testRawScores)
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
      val numTrees = treeBuff.length
      val logPrefix = s"Iteration $iter: Tree $numTrees:"

      // drop out
      if (boostConf.getBoostType == Dart) {
        dropTrees(dropped, boostConf, numTrees, dartRng)
        if (dropped.nonEmpty) {
          logInfo(s"$logPrefix ${dropped.size} trees dropped")
        } else {
          logInfo(s"$logPrefix skip drop")
        }
      }


      // build trees
      logInfo(s"$logPrefix start")
      val start = System.nanoTime
      val trees = buildTrees[C, B, H](data, trainRawScores, weightBuff.toArray, boostConf, iter, dropped.toSet)
      logInfo(s"$logPrefix finish, duration: ${(System.nanoTime - start) / 1e9} sec")

      if (trees.forall(_.isEmpty)) {
        // fail to build a new tree
        logInfo(s"$logPrefix no more tree built, GBM training finished")
        finished = true

      } else {
        // update base model buffer
        updateTreeBuffer(weightBuff, treeBuff, trees, dropped.toSet, boostConf)

        // whether to keep the weights of previous trees
        val keepWeights = boostConf.getBoostType != Dart || dropped.isEmpty

        // update train data predictions
        trainRawScores = updateRawScores[C, B, H](data, trainRawScores, trees, weightBuff.toArray, boostConf, keepWeights)
        trainRawScoresCheckpointer.update(trainRawScores)


        if (boostConf.getEvalFunc.isEmpty) {
          // materialize predictions
          trainRawScores.count()
        }

        // evaluate on train data
        if (boostConf.getEvalFunc.nonEmpty) {
          val trainMetrics = evaluate(data, trainRawScores, boostConf)
          trainMetricsHistory.append(trainMetrics)
          logInfo(s"$logPrefix train metrics ${trainMetrics.mkString("(", ", ", ")")}")
        }

        if (validation && boostConf.getEvalFunc.nonEmpty) {
          // update test data predictions
          testRawScores = updateRawScores[C, B, H](test, testRawScores, trees, weightBuff.toArray, boostConf, keepWeights)
          testRawScoresCheckpointer.update(testRawScores)

          // evaluate on test data
          val testMetrics = evaluate(test, testRawScores, boostConf)
          testMetricsHistory.append(testMetrics)
          logInfo(s"$logPrefix test metrics ${testMetrics.mkString("(", ", ", ")")}")
        }

        // callback
        if (boostConf.getCallbackFunc.nonEmpty) {
          // using cloning to avoid model modification
          val snapshot = new GBMModel(boostConf.getObjFunc,
            new Discretizer(discretizer.colDiscretizers.clone(), discretizer.zeroAsMissing, discretizer.sparsity),
            rawBase.clone(), treeBuff.toArray.clone(), weightBuff.toArray.map(nuh.toDouble).clone())

          // callback can update boosting configuration
          boostConf.getCallbackFunc.foreach { callback =>
            if (callback.compute(spark, boostConf, snapshot,
              trainMetricsHistory.toArray.clone(), testMetricsHistory.toArray.clone())) {
              finished = true
              logInfo(s"$logPrefix callback ${callback.name} stop training")
            }
          }
        }
      }

      iter += 1
    }

    if (iter >= boostConf.getMaxIter) {
      logInfo(s"maxIter=${boostConf.getMaxIter} reached, GBM training finished")
    }

    data.unpersist(blocking = false)
    trainRawScoresCheckpointer.unpersistDataSet()
    trainRawScoresCheckpointer.deleteAllCheckpoints()

    if (validation) {
      test.unpersist(blocking = false)
      testRawScoresCheckpointer.unpersistDataSet()
      testRawScoresCheckpointer.deleteAllCheckpoints()
    }

    new GBMModel(boostConf.getObjFunc, discretizer, rawBase.clone(),
      treeBuff.toArray, weightBuff.toArray.map(nuh.toDouble))
  }


  /**
    * append new tree to the model buffer
    *
    * @param weights     weights of trees
    * @param treeBuff    trees
    * @param trees       tree to be appended
    * @param dropped     indices of dropped trees
    * @param boostConfig boosting configuration
    */
  def updateTreeBuffer[H](weights: mutable.ArrayBuffer[H],
                          treeBuff: mutable.ArrayBuffer[TreeModel],
                          trees: Array[TreeModel],
                          dropped: Set[Int],
                          boostConfig: BoostConfig)
                         (implicit ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Unit = {
    import nuh._

    treeBuff.appendAll(trees)

    boostConfig.getBoostType match {
      case GBTree =>
        weights.appendAll(Iterator.fill(trees.length)(neh.fromDouble(boostConfig.getStepSize)))

      case Dart if dropped.isEmpty =>
        weights.appendAll(Iterator.fill(trees.length)(nuh.one))

      case Dart if dropped.nonEmpty =>
        require(dropped.size % boostConfig.getRawSize == 0)
        val k = dropped.size / boostConfig.getRawSize
        val w = neh.fromDouble(1 / (k + boostConfig.getStepSize))
        weights.appendAll(Iterator.fill(trees.length)(w))
        val scale = neh.fromDouble(k / (k + boostConfig.getStepSize))
        dropped.foreach { i => weights(i) *= scale }
    }
  }


  /**
    * drop trees
    *
    * @param dropped     indices of dropped trees
    * @param boostConfig boosting configuration
    * @param numTrees    number of trees
    * @param dartRng     random number generator
    */
  def dropTrees(dropped: mutable.Set[Int],
                boostConfig: BoostConfig,
                numTrees: Int,
                dartRng: Random): Unit = {
    dropped.clear()

    if (boostConfig.getDropSkip < 1 &&
      dartRng.nextDouble < 1 - boostConfig.getDropSkip) {

      require(numTrees % boostConfig.getRawSize == 0)
      val numBaseModels = numTrees / boostConfig.getRawSize

      var k = (numBaseModels * boostConfig.getDropRate).ceil.toInt
      k = math.max(k, boostConfig.getMinDrop)
      k = math.min(k, boostConfig.getMaxDrop)
      k = math.min(k, numBaseModels)

      if (k > 0) {
        dartRng.shuffle(Seq.range(0, numBaseModels)).take(k)
          .flatMap { i => Iterator.range(boostConfig.getRawSize * i, boostConfig.getRawSize * (i + 1)) }
          .foreach(dropped.add)
      }
    }
  }


  def buildTrees[C, B, H](instances: RDD[(H, Array[H], KVVector[C, B])],
                          rawScores: RDD[(Array[H], Array[H])],
                          weights: Array[H],
                          boostConfig: BoostConfig,
                          iteration: Int,
                          dropped: Set[Int])
                         (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                          cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                          ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Array[TreeModel] = {
    val numTrees = boostConfig.getBaseModelParallelism * boostConfig.getRawSize
    logInfo(s"Iteration $iteration: Starting to create $numTrees trees")

    val treeIdType = Utils.getTypeByRange(numTrees)
    logInfo(s"DataType of TreeId: $treeIdType")

    val nodeIdType = Utils.getTypeByRange(1 << boostConfig.getMaxDepth)
    logInfo(s"DataType of NodeId: $nodeIdType")

    (treeIdType, nodeIdType) match {
      case ("Byte", "Byte") =>
        buildTreesImpl[Byte, Byte, C, B, H](instances, rawScores, weights, boostConfig, iteration, numTrees, dropped)

      case ("Byte", "Short") =>
        buildTreesImpl[Byte, Short, C, B, H](instances, rawScores, weights, boostConfig, iteration, numTrees, dropped)

      case ("Byte", "Int") =>
        buildTreesImpl[Byte, Int, C, B, H](instances, rawScores, weights, boostConfig, iteration, numTrees, dropped)

      case ("Short", "Byte") =>
        buildTreesImpl[Short, Byte, C, B, H](instances, rawScores, weights, boostConfig, iteration, numTrees, dropped)

      case ("Short", "Short") =>
        buildTreesImpl[Short, Short, C, B, H](instances, rawScores, weights, boostConfig, iteration, numTrees, dropped)

      case ("Short", "Int") =>
        buildTreesImpl[Short, Int, C, B, H](instances, rawScores, weights, boostConfig, iteration, numTrees, dropped)

      case ("Int", "Byte") =>
        buildTreesImpl[Int, Byte, C, B, H](instances, rawScores, weights, boostConfig, iteration, numTrees, dropped)

      case ("Int", "Short") =>
        buildTreesImpl[Int, Short, C, B, H](instances, rawScores, weights, boostConfig, iteration, numTrees, dropped)

      case ("Int", "Int") =>
        buildTreesImpl[Int, Int, C, B, H](instances, rawScores, weights, boostConfig, iteration, numTrees, dropped)
    }
  }


  /**
    * build new trees
    *
    * @param instances   instances containing (weight, label, bins)
    * @param rawScores   previous raw predictions
    * @param weights     weights of trees
    * @param boostConfig boosting configuration
    * @param iteration   current iteration
    * @param numTrees    current number of trees
    * @param dropped     indices of trees which are selected to drop during building of current tree
    * @return new trees
    */
  def buildTreesImpl[T, N, C, B, H](instances: RDD[(H, Array[H], KVVector[C, B])],
                                    rawScores: RDD[(Array[H], Array[H])],
                                    weights: Array[H],
                                    boostConfig: BoostConfig,
                                    iteration: Int,
                                    numTrees: Int,
                                    dropped: Set[Int])
                                   (implicit ct: ClassTag[T], int: Integral[T],
                                    cn: ClassTag[N], inn: Integral[N],
                                    cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                    cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B],
                                    ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Array[TreeModel] = {
    import nuh._

    val numBaseModels = numTrees / boostConfig.getRawSize
    require(numTrees % boostConfig.getRawSize == 0)

    val rawBase = boostConfig.computeRawBaseScore.map(neh.fromDouble)

    val computeRaw = boostConfig.getBoostType match {
      case GBTree =>
        (rawSum: Array[H], rawSeq: Array[H]) => rawSum

      case Dart if dropped.isEmpty =>
        (rawSum: Array[H], rawSeq: Array[H]) => rawSum

      case Dart if dropped.nonEmpty =>
        (rawSum: Array[H], rawSeq: Array[H]) =>
          val raw = rawBase.clone()
          Iterator.range(0, rawSeq.length)
            .filterNot(dropped.contains)
            .foreach { i => raw(i % boostConfig.getRawSize) += rawSeq(i) * weights(i) }
          raw
    }

    val computeGradHess =
      (weight: H, label: Array[H], rawSum: Array[H], rawSeq: Array[H]) => {
        val raw = computeRaw(rawSum, rawSeq).map(nuh.toDouble)
        val score = boostConfig.getObjFunc.transform(raw)
        val (grad, hess) = boostConfig.getObjFunc.compute(label.map(nuh.toDouble), score)
        require(grad.length == boostConfig.getRawSize && hess.length == boostConfig.getRawSize)
        val weightedGrad = grad.map(v => neh.fromDouble(v) * weight)
        val weightedHess = hess.map(v => neh.fromDouble(v) * weight)
        (weightedGrad, weightedHess)
      }

    val baseConfig = if (boostConfig.getColSampleByTree == 1) {
      new BaseConfig(iteration, numTrees, Array.empty)
    } else {
      val maximum = (Int.MaxValue * boostConfig.getColSampleByTree).toInt
      val selectors: Array[ColumSelector] =
        Array.range(0, numBaseModels).flatMap { i =>
          Iterator.fill(boostConfig.getRawSize)(
            HashSelector(maximum, boostConfig.getSeed.toInt + iteration + i))
        }
      new BaseConfig(iteration, numTrees, selectors)
    }

    logInfo(s"Column Selectors: ${Array.range(0, numTrees).map(baseConfig.getSelector).mkString(",")}")

    val gradients = if (boostConfig.getSubSample == 1) {
      val treeIds = Array.range(0, numTrees).map(int.fromInt)

      instances.zip(rawScores).map { case ((weight, label, _), (rawSum, rawSeq)) =>
        val (weightedGrad, weightedHess) = computeGradHess(weight, label, rawSum, rawSeq)
        val gradSeq = Array.range(0, numBaseModels).flatMap(_ => weightedGrad)
        val hessSeq = Array.range(0, numBaseModels).flatMap(_ => weightedHess)
        (treeIds, gradSeq, hessSeq)
      }

    } else {
      instances.zip(rawScores).mapPartitionsWithIndex { case (partId, iter) =>
        val sampleRNGs = Array.tabulate(numBaseModels)(i => new XORShiftRandom(boostConfig.getSeed + partId + i))

        iter.map { case ((weight, label, _), (rawSum, rawSeq)) =>
          val treeIds = Array.range(0, numBaseModels).flatMap { i =>
            if (sampleRNGs(i).nextDouble < boostConfig.getSubSample) {
              Iterator.range(boostConfig.getRawSize * i, boostConfig.getRawSize * (i + 1)).map(int.fromInt)
            } else {
              Iterator.empty
            }
          }

          if (treeIds.nonEmpty) {
            val (weightedGrad, weightedHess) = computeGradHess(weight, label, rawSum, rawSeq)
            val n = treeIds.length / boostConfig.getRawSize
            val gradSeq = Array.range(0, n).flatMap(_ => weightedGrad)
            val hessSeq = Array.range(0, n).flatMap(_ => weightedHess)
            (treeIds, gradSeq, hessSeq)

          } else {
            (Array.empty[T], Array.empty[H], Array.empty[H])
          }
        }
      }
    }

    gradients.persist(boostConfig.getStorageLevel)

    val data = instances.zip(gradients)
      .map { case ((_, _, bins), (treeIds, gradSeq, hessSeq)) => (bins, treeIds, gradSeq, hessSeq) }

    instances.zip(gradients).zip(rawScores).collect().foreach { case (((weight, label, bins), (treeIds, gradSeq, hessSeq)), (rawSum, rawSeq)) =>
      val raw = computeRaw(rawSum, rawSeq).map(nuh.toDouble)
      val str = s"Iter: $iteration, weight: $weight, label: ${label.mkString(",")}, raw: ${raw.mkString(",")}, bins: $bins, treeIds: ${treeIds.mkString(",")}, grad: ${gradSeq.mkString(",")}, hess: ${hessSeq.mkString(",")}"
      logInfo(str)
    }

    val trees = Tree.train[T, N, C, B, H](data, boostConfig, baseConfig)
    gradients.unpersist(false)

    trees
  }


  /**
    * compute prediction of instances, containing the final score and the scores of each tree.
    *
    * @param instances   instances containing (weight, label, bins)
    * @param trees       array of trees
    * @param weights     array of weights
    * @param boostConfig boosting configuration
    * @return RDD containing final score (weighted) and the scores of each tree (non-weighted)
    */
  def computeRawScores[C, B, H](instances: RDD[(H, Array[H], KVVector[C, B])],
                                trees: Array[TreeModel],
                                weights: Array[H],
                                boostConfig: BoostConfig)
                               (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                cb: ClassTag[B], inb: Integral[B],
                                ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[(Array[H], Array[H])] = {
    import nuh._

    require(trees.length == weights.length)
    require(trees.length % boostConfig.getRawSize == 0)

    val rawBase = boostConfig.computeRawBaseScore.map(neh.fromDouble)
    require(boostConfig.getRawSize == rawBase.length)

    instances.map { case (_, _, bins) =>
      val rawSum = rawBase.clone()
      val rawSeq = trees.map { tree => neh.fromDouble(tree.predict(bins.apply)) }
      Iterator.range(0, rawSeq.length)
        .foreach { i => rawSum(i % boostConfig.getRawSize) += rawSeq(i) * weights(i) }

      (rawSum, rawSeq)
    }
  }


  /**
    * update prediction of instances, containing the final score and the predictions of each tree.
    *
    * @param instances   instances containing (weight, label, bins)
    * @param rawScores   previous predictions
    * @param forest      array of trees (new built)
    * @param weights     array of weights (total = old + new)
    * @param boostConfig boosting configuration
    * @param keepWeights whether to keep the weights of previous trees
    * @return RDD containing final score and the predictions of each tree
    */
  def updateRawScores[C, B, H](instances: RDD[(H, Array[H], KVVector[C, B])],
                               rawScores: RDD[(Array[H], Array[H])],
                               forest: Array[TreeModel],
                               weights: Array[H],
                               boostConfig: BoostConfig,
                               keepWeights: Boolean)
                              (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                               cb: ClassTag[B], inb: Integral[B],
                               ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): RDD[(Array[H], Array[H])] = {
    import nuh._

    require(forest.length % boostConfig.getRawSize == 0)

    if (keepWeights) {
      instances.zip(rawScores).map { case ((_, _, bins), (rawSum, rawSeq)) =>
        val newRawSeq = rawSeq ++ forest.map { tree => neh.fromDouble(tree.predict(bins.apply)) }
        require(rawSum.length == boostConfig.getRawSize)
        require(newRawSeq.length == weights.length)

        Iterator.range(rawSeq.length, newRawSeq.length)
          .foreach { i => rawSum(i % boostConfig.getRawSize) += newRawSeq(i) * weights(i) }

        (rawSum, newRawSeq)
      }

    } else {
      val rawBase = boostConfig.computeRawBaseScore.map(neh.fromDouble)
      require(boostConfig.getRawSize == rawBase.length)

      instances.zip(rawScores).map { case ((_, _, bins), (rawSum, rawSeq)) =>
        val newRawSeq = rawSeq ++ forest.map { tree => neh.fromDouble(tree.predict(bins.apply)) }
        require(rawSum.length == boostConfig.getRawSize)
        require(newRawSeq.length == weights.length)

        Iterator.range(0, boostConfig.getRawSize)
          .foreach { i => rawSum(i) = rawBase(i) }
        Iterator.range(0, newRawSeq.length)
          .foreach { i => rawSum(i % boostConfig.getRawSize) += newRawSeq(i) * weights(i) }

        (rawSum, newRawSeq)
      }
    }
  }


  /**
    * Evaluate current model and output the result
    *
    * @param instances   instances containing (weight, label, bins)
    * @param rawScores   prediction of instances, containing the final score and the scores of each tree
    * @param boostConfig boosting configuration containing the evaluation functions
    * @return Evaluation result with names as the keys and metrics as the values
    */
  def evaluate[H, C, B](instances: RDD[(H, Array[H], KVVector[C, B])],
                        rawScores: RDD[(Array[H], Array[H])],
                        boostConfig: BoostConfig)
                       (implicit cc: ClassTag[C], inc: Integral[C],
                        cb: ClassTag[B], inb: Integral[B],
                        ch: ClassTag[H], nuh: Numeric[H], neh: NumericExt[H]): Map[String, Double] = {
    if (boostConfig.getEvalFunc.isEmpty) {
      return Map.empty
    }

    val scores = instances.zip(rawScores)
      .map { case ((weight, label, _), (rawSum, _)) =>
        val raw = rawSum.map(nuh.toDouble)
        val score = boostConfig.getObjFunc.transform(raw)
        (nuh.toDouble(weight), label.map(nuh.toDouble), raw, score)
      }

    val result = mutable.OpenHashMap.empty[String, Double]

    // persist if there are batch evaluators
    if (boostConfig.getBatchEvalFunc.nonEmpty) {
      scores.persist(boostConfig.getStorageLevel)
    }

    if (boostConfig.getIncEvalFunc.nonEmpty) {
      IncEvalFunc.compute(scores,
        boostConfig.getIncEvalFunc, boostConfig.getAggregationDepth)
        .foreach { case (name, value) => result.update(name, value) }
    }

    if (boostConfig.getBatchEvalFunc.nonEmpty) {
      boostConfig.getBatchEvalFunc
        .foreach { eval => result.update(eval.name, eval.compute(scores)) }
      scores.unpersist(blocking = false)
    }

    result.toMap
  }
}

