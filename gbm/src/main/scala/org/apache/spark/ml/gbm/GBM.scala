package org.apache.spark.ml.gbm

import scala.collection.{BitSet, mutable}
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.storage.StorageLevel


/**
  * RDD-based API for gradient boosting machine
  */
class GBM extends Logging with Serializable {

  /** maximum number of iterations */
  private var maxIter: Int = 20

  def setMaxIter(value: Int): this.type = {
    require(value >= 0)
    maxIter = value
    this
  }

  def getMaxIter: Int = maxIter


  /** maximum tree depth */
  private var maxDepth: Int = 5

  def setMaxDepth(value: Int): this.type = {
    require(value >= 1)
    maxDepth = value
    this
  }

  def getMaxDepth: Int = maxDepth


  /** maximum number of tree leaves */
  private var maxLeaves: Int = 1000

  def setMaxLeaves(value: Int): this.type = {
    require(value >= 2)
    maxLeaves = value
    this
  }

  def getMaxLeaves: Int = maxLeaves


  /** minimum gain for each split */
  private var minGain: Double = 0.0

  def setMinGain(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    minGain = value
    this
  }

  def getMinGain: Double = minGain


  /** maximum number of bins for each column */
  private var maxBins: Int = 128

  def setMaxBins(value: Int): this.type = {
    require(value >= 4)
    maxBins = value
    this
  }

  def getMaxBins: Int = maxBins


  /** base score for global bias */
  private var baseScore: Double = 0.0

  def setBaseScore(value: Double): this.type = {
    require(!value.isNaN && !value.isInfinity)
    baseScore = value
    this
  }

  def getBaseScore: Double = baseScore


  /** minimum sum of hess for each node */
  private var minNodeHess: Double = 1.0

  def setMinNodeHess(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    minNodeHess = value
    this
  }

  def getMinNodeHess: Double = minNodeHess


  /** learning rate */
  private var stepSize: Double = 0.1

  def setStepSize(value: Double): this.type = {
    require(value > 0 && !value.isNaN && !value.isInfinity)
    stepSize = value
    this
  }

  def getStepSize: Double = stepSize


  /** L1 regularization term on weights */
  private var regAlpha: Double = 0.0

  def setRegAlpha(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    regAlpha = value
    this
  }

  def getRegAlpha: Double = regAlpha


  /** L2 regularization term on weights */
  private var regLambda: Double = 1.0

  def setRegLambda(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    regLambda = value
    this
  }

  def getRegLambda: Double = regLambda


  /** objective function */
  private var objectiveFunc: ObjFunc = new SquareObj

  def setObjectiveFunc(value: ObjFunc): this.type = {
    require(value != null)
    objectiveFunc = value
    this
  }

  def getObjectiveFunc: ObjFunc = objectiveFunc


  /** evaluation functions */
  private var evaluateFunc: Array[EvalFunc] = Array(new MAEEval, new MSEEval, new RMSEEval)

  def setEvaluateFunc(value: Array[EvalFunc]): this.type = {
    require(value.forall(e => e.isInstanceOf[IncrementalEvalFunc] || e.isInstanceOf[BatchEvalFunc]))
    require(value.map(_.name).distinct.length == value.length)
    evaluateFunc = value
    this
  }

  def getEvaluateFunc: Array[EvalFunc] = evaluateFunc


  /** callback functions */
  private var callbackFunc: Array[CallbackFunc] = Array(new EarlyStop)

  def setCallbackFunc(value: Array[CallbackFunc]): this.type = {
    require(value.map(_.name).distinct.length == value.length)
    callbackFunc = value
    this
  }

  def getCallbackFunc: Array[CallbackFunc] = callbackFunc


  /** indices of categorical columns */
  private var catCols: BitSet = BitSet.empty

  def setCatCols(value: Set[Int]): this.type = {
    require(value.forall(_ >= 0))
    val builder = BitSet.newBuilder
    builder ++= value
    catCols = builder.result()
    this
  }

  def getCatCols: Set[Int] = catCols.iterator.toSet


  /** indices of ranking columns */
  private var rankCols: BitSet = BitSet.empty

  def setRankCols(value: Set[Int]): this.type = {
    require(value.forall(_ >= 0))
    val builder = BitSet.newBuilder
    builder ++= value
    rankCols = builder.result()
    this
  }

  def getRankCols: Set[Int] = rankCols.iterator.toSet


  /** subsample ratio of the training instance */
  private var subSample: Double = 1.0

  def setSubSample(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    subSample = value
    this
  }

  def getSubSample: Double = subSample


  /** subsample ratio of columns when constructing each tree */
  private var colSampleByTree: Double = 1.0

  def setColSampleByTree(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    colSampleByTree = value
    this
  }

  def getColSampleByTree: Double = colSampleByTree


  /** subsample ratio of columns when constructing each level */
  private var colSampleByLevel: Double = 1.0

  def setColSampleByLevel(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    colSampleByLevel = value
    this
  }

  def getColSampleByLevel: Double = colSampleByLevel


  /** checkpoint interval */
  private var checkpointInterval: Int = 10

  def setCheckpointInterval(value: Int): this.type = {
    require(value == -1 || value > 0)
    checkpointInterval = value
    this
  }

  def getCheckpointInterval: Int = checkpointInterval


  /** storage level */
  private var storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  def setStorageLevel(value: StorageLevel): this.type = {
    require(value != StorageLevel.NONE)
    storageLevel = value
    this
  }

  def getStorageLevel: StorageLevel = storageLevel


  /** depth for treeAggregate */
  private var aggregationDepth: Int = 2

  def setAggregationDepth(value: Int): this.type = {
    require(value >= 2)
    aggregationDepth = value
    this
  }

  def getAggregationDepth: Int = aggregationDepth


  /** random number seed */
  private var seed: Long = -1L

  def setSeed(value: Long): this.type = {
    seed = value
    this
  }

  def getSeed: Long = seed


  /** parallelism of histogram subtraction and split searching */
  private var parallelism = -1

  def setParallelism(value: Int): this.type = {
    require(value != 0)
    parallelism = value
    this
  }

  def getParallelism: Int = parallelism


  /** whether to sample partitions instead of instances */
  private var enableSamplePartitions = false

  def setEnableSamplePartitions(value: Boolean): this.type = {
    enableSamplePartitions = value
    this
  }

  def getEnableSamplePartitions = enableSamplePartitions


  /** boosting type */
  private var boostType: String = GBM.GBTree

  def setBoostType(value: String): this.type = {
    require(value == GBM.GBTree || value == GBM.Dart)
    boostType = value
    this
  }

  def getBoostType: String = boostType


  /** dropout rate */
  private var dropRate: Double = 0.0

  def setDropRate(value: Double): this.type = {
    require(value >= 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    dropRate = value
    this
  }

  def getDropRate: Double = dropRate


  /** probability of skipping drop */
  private var dropSkip: Double = 0.5

  def setDropSkip(value: Double): this.type = {
    require(value >= 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    dropSkip = value
    this
  }

  def getDropSkip: Double = dropSkip

  /** minimum number of dropped trees in each iteration */
  private var minDrop: Int = 0

  def setMinDrop(value: Int): this.type = {
    require(value >= 0)
    minDrop = value
    this
  }

  def getMinDrop: Int = minDrop


  /** maximum number of dropped trees in each iteration */
  private var maxDrop: Int = 50

  def setMaxDrop(value: Int): this.type = {
    require(value >= 0)
    maxDrop = value
    this
  }

  def getMaxDrop: Int = maxDrop


  /** initial model */
  private var initialModel: Option[GBMModel] = None

  def setInitialModel(value: Option[GBMModel]): this.type = {
    initialModel = value
    this
  }

  def getInitialModel: Option[GBMModel] = initialModel


  /** the maximum number of non-zero histogram bins to search split for categorical columns by brute force */
  private var maxBruteBins: Int = 10

  def setMaxBruteBins(value: Int): this.type = {
    require(value >= 0)
    maxBruteBins = value
    this
  }

  def getMaxBruteBins: Int = maxBruteBins

  /** method to discretize numerical columns */
  private var numericalBinType: String = GBM.Width

  def setNumericalBinType(value: String): this.type = {
    require(value == GBM.Width || value == GBM.Depth)
    numericalBinType = value
    this
  }

  def getNumericalBinType: String = numericalBinType


  /** float precision to represent internal gradient, hessian and prediction */
  private var floatType: String = GBM.SinglePrecision

  def setFloatType(value: String): this.type = {
    require(value == GBM.SinglePrecision || value == GBM.DoublePrecision)
    floatType = value
    this
  }

  def getFloatType: String = floatType


  /** whether zero is viewed as missing value */
  private var zeroAsMissing: Boolean = false

  def setZeroAsMissing(value: Boolean): this.type = {
    zeroAsMissing = value
    this
  }

  def getZeroAsMissing: Boolean = zeroAsMissing


  /** training, dataset contains (weight, label, vec) */
  def fit(data: RDD[(Double, Double, Vector)]): GBMModel = {
    fit(data, None)
  }


  /** training with validation, dataset contains (weight, label, vec) */
  def fit(data: RDD[(Double, Double, Vector)],
          test: RDD[(Double, Double, Vector)]): GBMModel = {
    fit(data, Some(test))
  }


  /** training with validation if any, dataset contains (weight, label, vec) */
  private[ml] def fit(data: RDD[(Double, Double, Vector)],
                      test: Option[RDD[(Double, Double, Vector)]]): GBMModel = {
    if (boostType == GBM.Dart) {
      require(maxDrop >= minDrop)
    }

    val sc = data.sparkContext

    val numCols = data.first._3.size
    require(numCols > 0)

    val validation = test.isDefined

    val discretizer = if (initialModel.isDefined) {
      require(numCols == initialModel.get.discretizer.colDiscretizers.length)
      logWarning(s"Discretizer is already provided by the initial model, related params" +
        s" (maxBins,catCols,rankCols,numericalBinType) will be ignored")
      initialModel.get.discretizer

    } else {
      catCols.iterator.forall(v => v >= 0 && v < numCols)
      rankCols.iterator.forall(v => v >= 0 && v < numCols)

      require((catCols & rankCols).iterator.isEmpty)

      Discretizer.fit(data.map(_._3), numCols, catCols, rankCols, maxBins,
        numericalBinType, zeroAsMissing, aggregationDepth)
    }
    logInfo(s"Average number of bins: ${discretizer.numBins.sum.toDouble / discretizer.numBins.length}")
    logInfo(s"Sparsity of train data: ${discretizer.sparsity}")

    val boostConfig = new BoostConfig
    boostConfig.setMaxIter(maxIter)
      .setMaxDepth(maxDepth)
      .setMaxLeaves(maxLeaves)
      .setNumCols(numCols)
      .setMinGain(minGain)
      .setMinNodeHess(minNodeHess)
      .setStepSize(stepSize)
      .setRegAlpha(regAlpha)
      .setRegLambda(regLambda)
      .setObjectiveFunc(objectiveFunc)
      .setEvaluateFunc(evaluateFunc)
      .setCallbackFunc(callbackFunc)
      .setCatCols(catCols)
      .setRankCols(rankCols)
      .setSubSample(subSample)
      .setColSampleByTree(colSampleByTree)
      .setColSampleByLevel(colSampleByLevel)
      .setCheckpointInterval(checkpointInterval)
      .setStorageLevel(storageLevel)
      .setBoostType(boostType)
      .setDropRate(dropRate)
      .setDropSkip(dropSkip)
      .setMinDrop(minDrop)
      .setMaxDrop(maxDrop)
      .setAggregationDepth(aggregationDepth)
      .setMaxBruteBins(maxBruteBins)
      .setFloatType(floatType)
      .setParallelism(parallelism)
      .setEnableSamplePartitions(enableSamplePartitions)
      .setSeed(seed)


    if (initialModel.isEmpty) {
      boostConfig.setBaseScore(baseScore)

    } else {

      if (initialModel.get.baseScore != baseScore) {
        logWarning(s"baseScore is already provided by the initial model, related param (baseScore) will be ignored")
      }
      boostConfig.setBaseScore(initialModel.get.baseScore)
    }

    GBM.boost(data, test.getOrElse(sc.emptyRDD), boostConfig, validation, discretizer, initialModel)
  }
}


private[gbm] object GBM extends Logging {

  val GBTree = "gbtree"
  val Dart = "dart"

  val Width = "width"
  val Depth = "depth"

  val SinglePrecision = "float"
  val DoublePrecision = "double"


  /**
    * train a GBM model, dataset contains (weight, label, vec)
    */
  def boost(data: RDD[(Double, Double, Vector)],
            test: RDD[(Double, Double, Vector)],
            boostConfig: BoostConfig,
            validation: Boolean,
            discretizer: Discretizer,
            initialModel: Option[GBMModel]): GBMModel = {

    val maxBinIndex = discretizer.numBins.max - 1

    if (maxBinIndex <= Byte.MaxValue) {
      logInfo("Data representation of bins: Vector[Byte]")
      boostWithBinType[Byte](data, test, boostConfig, validation, discretizer, initialModel)

    } else if (maxBinIndex <= Short.MaxValue) {
      logInfo("Data representation of bins: Vector[Short]")
      boostWithBinType[Short](data, test, boostConfig, validation, discretizer, initialModel)

    } else {
      logInfo("Data representation of bins: Vector[Int]")
      boostWithBinType[Int](data, test, boostConfig, validation, discretizer, initialModel)
    }
  }


  /**
    * train a GBM model, with the given type of bin, dataset contains (weight, label, vec)
    */
  def boostWithBinType[B: Integral : ClassTag](data: RDD[(Double, Double, Vector)],
                                               test: RDD[(Double, Double, Vector)],
                                               boostConfig: BoostConfig,
                                               validation: Boolean,
                                               discretizer: Discretizer,
                                               initialModel: Option[GBMModel]): GBMModel = {
    val binData = discretizer.transform[B](data)
    val binTest = discretizer.transform[B](test)

    boostConfig.getFloatType match {
      case SinglePrecision =>
        logInfo("Data representation of gradient: Float")
        boostImpl[Float, B](binData, binTest, boostConfig, validation, discretizer, initialModel)

      case DoublePrecision =>
        logInfo("Data representation of gradient: Double")
        boostImpl[Double, B](binData, binTest, boostConfig, validation, discretizer, initialModel)
    }
  }


  /**
    * implementation of GBM, train a GBMModel, with given bin type and float precision
    *
    * @param data         training instances containing (weight, label, bins)
    * @param test         validation instances containing (weight, label, bins)
    * @param boostConfig  boosting configuration
    * @param validation   whether to validate on test data
    * @param discretizer  discretizer to convert raw features into bins
    * @param initialModel inital model
    * @tparam H type of gradient, hessian and tree predictions
    * @tparam B type of bin
    * @return the model
    */
  def boostImpl[H: Numeric : ClassTag : FromDouble, B: Integral : ClassTag](data: RDD[(Double, Double, BinVector[B])],
                                                                            test: RDD[(Double, Double, BinVector[B])],
                                                                            boostConfig: BoostConfig,
                                                                            validation: Boolean,
                                                                            discretizer: Discretizer,
                                                                            initialModel: Option[GBMModel]): GBMModel = {
    val spark = SparkSession.builder().getOrCreate()
    val sc = spark.sparkContext
    Utils.registerKryoClasses(sc)

    data.persist(boostConfig.getStorageLevel)
    logInfo(s"${data.count} instances in train data")

    if (validation) {
      test.persist(boostConfig.getStorageLevel)
      logInfo(s"${test.count} instances in test data")
    }

    val weights = mutable.ArrayBuffer.empty[Double]
    val trees = mutable.ArrayBuffer.empty[TreeModel]

    if (initialModel.isDefined) {
      weights.appendAll(initialModel.get.weights)
      trees.appendAll(initialModel.get.trees)
    }

    val trainPredsCheckpointer = new Checkpointer[Array[H]](sc,
      boostConfig.getCheckpointInterval, boostConfig.getStorageLevel)

    var trainPreds = computePrediction[H, B](data, trees.zip(weights).toArray, boostConfig.getBaseScore)
    trainPredsCheckpointer.update(trainPreds)

    val testPredsCheckpointer = new Checkpointer[Array[H]](sc,
      boostConfig.getCheckpointInterval, boostConfig.getStorageLevel)

    var testPreds = sc.emptyRDD[Array[H]]
    if (validation && boostConfig.getEvaluateFunc.nonEmpty) {
      testPreds = computePrediction[H, B](test, trees.zip(weights).toArray, boostConfig.getBaseScore)
      testPredsCheckpointer.update(testPreds)
    }

    // metrics history recoder
    val trainMetricsHistory = mutable.ArrayBuffer.empty[Map[String, Double]]
    val testMetricsHistory = mutable.ArrayBuffer.empty[Map[String, Double]]

    // random number generator for drop out
    val dartRng = new Random(boostConfig.getSeed)
    val dropped = mutable.Set.empty[Int]

    // random number generator for column sampling
    val colSampleRng = new Random(boostConfig.getSeed)

    var iter = 0
    var finished = false

    while (!finished && iter < boostConfig.getMaxIter) {
      // if initial model is provided, round will not equal to iter
      val numTrees = trees.length
      val logPrefix = s"Iter $iter: Tree $numTrees:"

      // drop out
      if (boostConfig.getBoostType == Dart) {
        dropTrees(dropped, boostConfig, numTrees, dartRng)
        if (dropped.nonEmpty) {
          logInfo(s"$logPrefix ${dropped.size} trees dropped")
        } else {
          logInfo(s"$logPrefix skip drop")
        }
      }

      // build tree
      logInfo(s"$logPrefix start")
      val start = System.nanoTime
      val tree = buildTree(data, trainPreds, weights.toArray, boostConfig, iter,
        numTrees, dropped.toSet, colSampleRng)
      logInfo(s"$logPrefix finish, duration: ${(System.nanoTime - start) / 1e9} sec")

      if (tree.isEmpty) {
        // fail to build a new tree
        logInfo(s"$logPrefix no more tree built, GBM training finished")
        finished = true

      } else {
        // update base model buffer
        updateTreeBuffer(weights, trees, tree.get, dropped.toSet, boostConfig)

        // whether to keep the weights of previous trees
        val keepWeights = boostConfig.getBoostType != Dart || dropped.isEmpty

        // update train data predictions
        trainPreds = updatePrediction(data, trainPreds, weights.toArray, tree.get,
          boostConfig.getBaseScore, keepWeights)
        trainPredsCheckpointer.update(trainPreds)

        if (boostConfig.getEvaluateFunc.isEmpty) {
          // materialize predictions
          trainPreds.count()
        }

        // evaluate on train data
        if (boostConfig.getEvaluateFunc.nonEmpty) {
          val trainMetrics = evaluate(data, trainPreds, boostConfig)
          trainMetricsHistory.append(trainMetrics)
          logInfo(s"$logPrefix train metrics ${trainMetrics.mkString("(", ", ", ")")}")
        }

        if (validation && boostConfig.getEvaluateFunc.nonEmpty) {
          // update test data predictions
          testPreds = updatePrediction(test, testPreds, weights.toArray, tree.get,
            boostConfig.getBaseScore, keepWeights)
          testPredsCheckpointer.update(testPreds)

          // evaluate on test data
          val testMetrics = evaluate(test, testPreds, boostConfig)
          testMetricsHistory.append(testMetrics)
          logInfo(s"$logPrefix test metrics ${testMetrics.mkString("(", ", ", ")")}")
        }

        // callback
        if (boostConfig.getCallbackFunc.nonEmpty) {
          // using cloning to avoid model modification
          val snapshot = new GBMModel(
            new Discretizer(discretizer.colDiscretizers.clone(), discretizer.zeroAsMissing, discretizer.sparsity),
            boostConfig.getBaseScore, trees.toArray.clone(), weights.toArray.clone())

          // callback can update boosting configuration
          boostConfig.getCallbackFunc.foreach { callback =>
            if (callback.compute(spark, boostConfig, snapshot,
              trainMetricsHistory.toArray.clone(), testMetricsHistory.toArray.clone())) {
              finished = true
              logInfo(s"$logPrefix callback ${callback.name} stop training")
            }
          }
        }
      }

      iter += 1
    }
    if (iter >= boostConfig.getMaxIter) {
      logInfo(s"maxIter=${boostConfig.getMaxIter} reached, GBM training finished")
    }

    data.unpersist(blocking = false)
    trainPredsCheckpointer.unpersistDataSet()
    trainPredsCheckpointer.deleteAllCheckpoints()

    if (validation) {
      test.unpersist(blocking = false)
      testPredsCheckpointer.unpersistDataSet()
      testPredsCheckpointer.deleteAllCheckpoints()
    }

    new GBMModel(discretizer, boostConfig.getBaseScore, trees.toArray, weights.toArray)
  }


  /**
    * append new tree to the model buffer
    *
    * @param weights     weights of trees
    * @param trees       trees
    * @param tree        tree to be appended
    * @param dropped     indices of dropped trees
    * @param boostConfig boosting configuration
    */
  def updateTreeBuffer(weights: mutable.ArrayBuffer[Double],
                       trees: mutable.ArrayBuffer[TreeModel],
                       tree: TreeModel,
                       dropped: Set[Int],
                       boostConfig: BoostConfig): Unit = {

    trees.append(tree)

    boostConfig.getBoostType match {
      case GBTree =>
        weights.append(boostConfig.getStepSize)

      case Dart if dropped.isEmpty =>
        weights.append(1.0)

      case Dart if dropped.nonEmpty =>
        val k = dropped.size
        weights.append(1 / (k + boostConfig.getStepSize))
        val scale = k / (k + boostConfig.getStepSize)
        dropped.foreach { i =>
          weights(i) *= scale
        }
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
      var k = (numTrees * boostConfig.getDropRate).ceil.toInt
      k = math.max(k, boostConfig.getMinDrop)
      k = math.min(k, boostConfig.getMaxDrop)
      k = math.min(k, numTrees)

      if (k > 0) {
        dartRng.shuffle(Seq.range(0, numTrees))
          .take(k).foreach(dropped.add)
      }
    }
  }


  /**
    * build a new tree
    *
    * @param instances    instances containing (weight, label, bins)
    * @param preds        previous predictions
    * @param weights      weights of trees
    * @param boostConfig  boosting configuration
    * @param iteration    current iteration
    * @param numTrees     current number of trees
    * @param dropped      indices of columns which are selected to drop during building of current tree
    * @param colSampleRng random number generator for column sampling
    * @tparam H
    * @tparam B
    * @return a new tree if possible
    */
  def buildTree[H: Numeric : ClassTag : FromDouble, B: Integral : ClassTag](instances: RDD[(Double, Double, BinVector[B])],
                                                                            preds: RDD[Array[H]],
                                                                            weights: Array[Double],
                                                                            boostConfig: BoostConfig,
                                                                            iteration: Int,
                                                                            numTrees: Int,
                                                                            dropped: Set[Int],
                                                                            colSampleRng: Random): Option[TreeModel] = {
    val numH = implicitly[Numeric[H]]
    val toH = implicitly[FromDouble[H]]

    val sc = instances.sparkContext

    var handlePersistence = false

    val zipped = instances.zip(preds)

    val rowSampled = if (boostConfig.getSubSample == 1) {
      zipped

    } else if (boostConfig.getEnableSamplePartitions &&
      (instances.getNumPartitions * boostConfig.getSubSample).ceil < instances.getNumPartitions) {
      import RDDFunctions._
      zipped.samplePartitions(boostConfig.getSubSample, boostConfig.getSeed + numTrees)

    } else {

      handlePersistence = true
      zipped.sample(false, boostConfig.getSubSample, boostConfig.getSeed + numTrees)
    }


    // selected columns
    val cols = if (boostConfig.getColSampleByTree == 1) {
      Array.range(0, boostConfig.getNumCols)

    } else {

      handlePersistence = true
      val numCols = (boostConfig.getNumCols * boostConfig.getColSampleByTree).ceil.toInt
      colSampleRng.shuffle(Seq.range(0, boostConfig.getNumCols))
        .take(numCols).toArray.sorted
    }

    // column sampling function
    val slice = if (boostConfig.getColSampleByTree == 1) {
      (bins: BinVector[B]) => bins
    } else {
      (bins: BinVector[B]) => bins.slice(cols)
    }

    // indices of categorical columns in the selected column subset
    val catCols = {
      val builder = BitSet.newBuilder
      var i = 0
      while (i < cols.length) {
        if (boostConfig.isCat(cols(i))) {
          builder += i
        }
        i += 1
      }
      builder.result()
    }


    val colSampled = boostConfig.getBoostType match {
      case GBTree =>
        rowSampled.map { case ((weight, label, bins), pred) =>
          val (grad, hess) = boostConfig.getObjectiveFunc.compute(label, numH.toDouble(pred.head))
          (toH.fromDouble(grad * weight), toH.fromDouble(hess * weight), slice(bins))
        }

      case Dart if dropped.isEmpty =>
        rowSampled.map { case ((weight, label, bins), pred) =>
          val (grad, hess) = boostConfig.getObjectiveFunc.compute(label, numH.toDouble(pred.head))
          (toH.fromDouble(grad * weight), toH.fromDouble(hess * weight), slice(bins))
        }

      case Dart if dropped.nonEmpty =>
        rowSampled.map { case ((weight, label, bins), pred) =>
          var score = boostConfig.getBaseScore

          var i = 0
          while (i < weights.length) {
            if (!dropped.contains(i)) {
              score += numH.toDouble(pred(i + 1)) * weights(i)
            }
            i += 1
          }

          val (grad, hess) = boostConfig.getObjectiveFunc.compute(label, score)
          (toH.fromDouble(grad * weight), toH.fromDouble(hess * weight), slice(bins))
        }
    }


    if (handlePersistence) {
      colSampled.persist(boostConfig.getStorageLevel)
    }

    val treeConfig = new TreeConfig(iteration, numTrees, catCols, cols)
    val tree = Tree.train[H, B](colSampled, boostConfig, treeConfig)

    if (handlePersistence) {
      colSampled.unpersist(blocking = false)
    }

    tree
  }


  /**
    * compute prediction of instances, containing the final score and the scores of each tree.
    *
    * @param instances instances containing (weight, label, bins)
    * @param trees     array of trees with weights
    * @param baseScore global bias
    * @tparam H
    * @tparam B
    * @return RDD containing final score and the scores of each tree
    */
  def computePrediction[H: Numeric : ClassTag : FromDouble, B: Integral : ClassTag](instances: RDD[(Double, Double, BinVector[B])],
                                                                                    trees: Array[(TreeModel, Double)],
                                                                                    baseScore: Double): RDD[Array[H]] = {
    val numH = implicitly[Numeric[H]]
    val toH = implicitly[FromDouble[H]]

    instances.map { case (_, _, bins) =>
      val pred = Array.fill(trees.length + 1)(numH.zero)

      pred(0) = toH.fromDouble(baseScore)

      var i = 0
      while (i < trees.length) {
        val (tree, w) = trees(i)
        val p = toH.fromDouble(tree.predict(bins))
        pred(i + 1) = p
        pred(0) = numH.plus(pred(0), numH.times(p, toH.fromDouble(w)))
        i += 1
      }

      pred
    }
  }


  /**
    * update prediction of instances, containing the final score and the predictions of each tree.
    *
    * @param instances   instances containing (weight, label, bins)
    * @param preds       previous predictions
    * @param weights     weights of each tree
    * @param tree        the last tree model
    * @param baseScore   global bias
    * @param keepWeights whether to keep the weights of previous trees
    * @tparam H
    * @tparam B
    * @return RDD containing final score and the predictions of each tree
    */
  def updatePrediction[H: Numeric : ClassTag : FromDouble, B: Integral : ClassTag](instances: RDD[(Double, Double, BinVector[B])],
                                                                                   preds: RDD[Array[H]],
                                                                                   weights: Array[Double],
                                                                                   tree: TreeModel,
                                                                                   baseScore: Double,
                                                                                   keepWeights: Boolean): RDD[Array[H]] = {
    val numH = implicitly[Numeric[H]]
    val toH = implicitly[FromDouble[H]]

    if (keepWeights) {
      instances.zip(preds).map { case ((_, _, bins), pred) =>
        val p = toH.fromDouble(tree.predict[B](bins))
        val newPred = pred :+ p
        require(newPred.length == weights.length + 1)

        newPred(0) = numH.plus(newPred(0), numH.times(p, toH.fromDouble(weights.last)))
        newPred
      }

    } else {

      instances.zip(preds).map { case ((_, _, bins), pred) =>
        val p = toH.fromDouble(tree.predict[B](bins))
        val newPred = pred :+ p
        require(newPred.length == weights.length + 1)
        newPred(0) = toH.fromDouble(baseScore)

        var i = 0
        while (i < weights.length) {
          newPred(0) = numH.plus(newPred(0), numH.times(newPred(i + 1), toH.fromDouble(weights(i))))
          i += 1
        }

        newPred
      }
    }
  }


  /**
    * Evaluate current model and output the result
    *
    * @param instances   instances containing (weight, label, bins)
    * @param preds       prediction of instances, containing the final score and the scores of each tree
    * @param boostConfig boosting configuration containing the evaluation functions
    * @tparam H
    * @tparam B
    * @return Evaluation result with names as the keys and metrics as the values
    */
  def evaluate[H: Numeric : ClassTag, B: Integral : ClassTag](instances: RDD[(Double, Double, BinVector[B])],
                                                              preds: RDD[Array[H]],
                                                              boostConfig: BoostConfig): Map[String, Double] = {

    if (boostConfig.getEvaluateFunc.isEmpty) {
      return Map.empty
    }

    val numH = implicitly[Numeric[H]]

    val scores = instances.zip(preds).map {
      case ((weight, label, _), pred) =>
        (weight, label, numH.toDouble(pred.head))
    }

    val result = mutable.OpenHashMap.empty[String, Double]

    // persist if there are batch evaluators
    if (boostConfig.getBatchEvaluateFunc.nonEmpty) {
      scores.persist(boostConfig.getStorageLevel)
    }

    if (boostConfig.getIncrementalEvaluateFunc.nonEmpty) {
      IncrementalEvalFunc.compute(scores,
        boostConfig.getIncrementalEvaluateFunc, boostConfig.getAggregationDepth)
        .foreach {
          case (name, value) =>
            result.update(name, value)
        }
    }

    if (boostConfig.getBatchEvaluateFunc.nonEmpty) {
      boostConfig.getBatchEvaluateFunc.foreach { eval =>
        val value = eval.compute(scores)
        result.update(eval.name, value)
      }
      scores.unpersist(blocking = false)
    }

    result.toMap
  }
}


/**
  * Model of GBM
  *
  * @param discretizer discretizer to convert raw features into bins
  * @param baseScore   global bias
  * @param trees       array of trees
  * @param weights     weights of trees
  */
class GBMModel(val discretizer: Discretizer,
               val baseScore: Double,
               val trees: Array[TreeModel],
               val weights: Array[Double]) extends Serializable {
  require(trees.length == weights.length)
  require(!baseScore.isNaN && !baseScore.isInfinity)
  require(weights.forall(w => !w.isNaN || !w.isInfinity))

  /** feature importance of whole trees */
  lazy val importance: Vector = computeImportance(numTrees)

  def numCols: Int = discretizer.colDiscretizers.length

  def numTrees: Int = trees.length

  def numLeaves: Array[Long] = trees.map(_.numLeaves)

  def numNodes: Array[Long] = trees.map(_.numNodes)

  def depths: Array[Int] = trees.map(_.depth)

  /** feature importance of the first trees */
  def computeImportance(firstTrees: Int): Vector = {
    require(firstTrees >= -1 && firstTrees <= numTrees)

    var n = firstTrees
    if (n == 0) {
      return Vectors.sparse(n, Seq.empty)
    }

    if (n == -1) {
      n = numTrees
    }

    val gains = Array.ofDim[Double](numCols)

    var i = 0
    while (i < n) {
      trees(i).computeImportance.foreach {
        case (index, gain) =>
          gains(index) += gain * weights(i)
      }
      i += 1
    }

    val sum = gains.sum

    i = 0
    while (i < n) {
      gains(i) /= sum
      i += 1
    }

    Vectors.dense(gains).compressed
  }


  def predict(features: Vector): Double = {
    predict(features, numTrees)
  }


  def predict(features: Vector,
              firstTrees: Int): Double = {
    require(features.size == numCols)
    require(firstTrees >= -1 && firstTrees <= numTrees)

    var n = firstTrees
    if (n == -1) {
      n = numTrees
    }

    var score = baseScore
    var i = 0
    while (i < n) {
      score += trees(i).predict(features, discretizer) * weights(i)
      i += 1
    }
    score
  }


  def leaf(features: Vector): Vector = {
    leaf(features, false)
  }


  def leaf(features: Vector,
           oneHot: Boolean): Vector = {
    leaf(features, oneHot, numTrees)
  }


  /** leaf transformation with first trees
    * if oneHot is enable, transform input into a sparse one-hot encoded vector */
  def leaf(features: Vector,
           oneHot: Boolean,
           firstTrees: Int): Vector = {
    require(features.size == numCols)
    require(firstTrees >= -1 && firstTrees <= numTrees)

    var n = firstTrees
    if (n == -1) {
      n = numTrees
    }

    if (oneHot) {

      val indices = Array.ofDim[Int](n)

      var step = 0
      var i = 0
      while (i < n) {
        val index = trees(i).index(features, discretizer)
        indices(i) = step + index.toInt
        step += numLeaves(i).toInt
        i += 1
      }

      val values = Array.fill(n)(1.0)

      Vectors.sparse(step, indices, values)

    } else {

      val indices = Array.ofDim[Double](n)
      var i = 0
      while (i < n) {
        indices(i) = trees(i).index(features, discretizer).toDouble
        i += 1
      }

      Vectors.dense(indices).compressed
    }
  }

  def save(path: String): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    GBMModel.save(spark, this, path)
  }
}


object GBMModel {

  /** save GBMModel to a path */
  private[ml] def save(spark: SparkSession,
                       model: GBMModel,
                       path: String): Unit = {
    val names = Array("discretizerCol", "discretizerExtra", "weights", "trees", "extra")
    val dataframes = toDF(spark, model)
    Utils.saveDataFrames(dataframes, names, path)
  }


  /** load GBMModel from a path */
  def load(path: String): GBMModel = {
    val spark = SparkSession.builder().getOrCreate()
    val names = Array("discretizerCol", "discretizerExtra", "weights", "trees", "extra")
    val dataframes = Utils.loadDataFrames(spark, names, path)
    fromDF(dataframes)
  }


  /** helper function to convert GBMModel to dataframes */
  private[gbm] def toDF(spark: SparkSession,
                        model: GBMModel): Array[DataFrame] = {

    val Array(disColDF, disExtraDF) = Discretizer.toDF(spark, model.discretizer)

    val weightsDatum = model.weights.zipWithIndex
    val weightsDF = spark.createDataFrame(weightsDatum)
      .toDF("weight", "treeIndex")

    val treesDatum = model.trees.zipWithIndex.flatMap {
      case (tree, index) =>
        val (nodeData, _) = NodeData.createData(tree.root, 0)
        nodeData.map((_, index))
    }
    val treesDF = spark.createDataFrame(treesDatum)
      .toDF("node", "treeIndex")

    val extraDF = spark.createDataFrame(
      Seq(("baseScore", model.baseScore.toString)))
      .toDF("key", "value")

    Array(disColDF, disExtraDF, weightsDF, treesDF, extraDF)
  }


  /** helper function to convert dataframes back to GBMModel */
  private[gbm] def fromDF(dataframes: Array[DataFrame]): GBMModel = {
    val Array(disColDF, disExtraDF, weightsDF, treesDF, extraDF) = dataframes

    val spark = disColDF.sparkSession
    import spark.implicits._

    val discretizer = Discretizer.fromDF(Array(disColDF, disExtraDF))

    val (indices, weights) =
      weightsDF.select("treeIndex", "weight").rdd
        .map { row =>
          val index = row.getInt(0)
          val weight = row.getDouble(1)
          (index, weight)
        }.collect().sortBy(_._1).unzip

    require(indices.length == indices.distinct.length)
    require(indices.length == indices.max + 1)

    val (indices2, trees) =
      treesDF.select("treeIndex", "node").as[(Int, NodeData)].rdd
        .groupByKey().map { case (index, nodes) =>
        val root = NodeData.createNode(nodes.toArray)
        (index, new TreeModel(root))
      }.collect().sortBy(_._1).unzip

    require(indices.length == indices2.length)
    require(indices2.length == indices2.distinct.length)
    require(indices2.length == indices2.max + 1)

    var baseScore = Double.NaN
    extraDF.select("key", "value").collect()
      .foreach { row =>
        val key = row.getString(0)
        val value = row.getString(1)

        key match {
          case "baseScore" =>
            baseScore = value.toDouble
        }
      }
    require(!baseScore.isNaN && !baseScore.isInfinity)

    new GBMModel(discretizer, baseScore, trees, weights)
  }
}
