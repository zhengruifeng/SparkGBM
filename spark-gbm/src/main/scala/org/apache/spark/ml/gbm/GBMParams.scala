package org.apache.spark.ml.gbm

import scala.util.Try

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.storage.StorageLevel


private[ml] trait GBMParams extends PredictorParams with HasWeightCol with HasMaxIter
  with HasStepSize with HasCheckpointInterval with HasAggregationDepth with HasSeed {

  setDefault(maxIter -> 20)

  setDefault(stepSize -> 0.1)

  setDefault(checkpointInterval -> 10)


  /**
    * Parallelism type.
    * (default = data)
    *
    * @group param
    */
  final val parallelismType: Param[String] =
    new Param[String](this, "parallelismType", "Parallelism type.",
      ParamValidators.inArray[String](Array("data", "feature")))

  def getParallelismType: String = $(parallelismType)

  setDefault(parallelismType -> "data")


  /**
    * whether to update prediction and gradient after each level.
    * (default = false)
    *
    * @group param
    */
  final val greedierSearch: BooleanParam =
    new BooleanParam(this, "greedierSearch", "Whether to update " +
      "prediction and gradient after each level.")

  def getGreedierSearch: Boolean = $(greedierSearch)

  setDefault(greedierSearch -> false)


  /**
    * Leaf index column name.
    * (default = leafCol)
    *
    * @group param
    */
  final val leafCol: Param[String] =
    new Param[String](this, "leafCol", "Leaf index column name.")

  def getLeafCol: String = $(leafCol)

  setDefault(leafCol -> "")

  /**
    * Maximum depth of the tree (>= 1).
    * E.g., depth 1 means 1 internal node + 2 leaf nodes.
    * (default = 5)
    *
    * @group param
    */
  final val maxDepth: IntParam =
    new IntParam(this, "maxDepth", "Maximum depth of the tree. (>= 1)" +
      " E.g., depth 1 means 1 internal node + 2 leaf nodes.",
      ParamValidators.gtEq(1))

  def getMaxDepth: Int = $(maxDepth)

  setDefault(maxDepth -> 5)

  /**
    * Maximum number of bins used for discretizing continuous features and for choosing how to split
    * on features at each node.  More bins give higher granularity.
    * Must be >= 4 and >= number of categories in any categorical feature.
    * (default = 64)
    *
    * @group param
    */
  final val maxBins: IntParam =
    new IntParam(this, "maxBins", "Max number of bins for" +
      " discretizing continuous features.  Must be >=4 and >= number of categories for any" +
      " categorical feature.", ParamValidators.gtEq(4))

  def getMaxBins: Int = $(maxBins)

  setDefault(maxBins -> 64)

  /**
    * Maximum number of leaves of the tree (>= 2).
    * (default = 1000)
    *
    * @group param
    */
  final val maxLeaves: IntParam =
    new IntParam(this, "maxLeaves", "Maximum number of leaves of the tree. (>= 2).",
      ParamValidators.gtEq(2))

  def getMaxLeaves: Int = $(maxLeaves)

  setDefault(maxLeaves -> 1000)

  /**
    * Minimum gain required to make a further partition on a leaf node of the tree.
    * (default = 0.0)
    *
    * @group param
    */
  final val minGain: DoubleParam =
    new DoubleParam(this, "minGain", "Minimum gain required to make a further partition " +
      "on a leaf node of the tree.",
      ParamValidators.gtEq(0.0))

  def getMinGain: Double = $(minGain)

  setDefault(minGain -> 0.0)


  /**
    * Global bias, the initial prediction score of all instances.
    * (default = 0.0)
    *
    * @group param
    */
  final val baseScore: DoubleArrayParam =
    new DoubleArrayParam(this, "baseScore", "Global bias, the initial prediction score of all instances.",
      (baseScore: Array[Double]) => baseScore.forall(v => !v.isNaN && !v.isInfinity))

  def getBaseScore: Array[Double] = $(baseScore)

  setDefault(baseScore -> Array.emptyDoubleArray)


  /**
    * Minimum sum of instance hessian needed in a node.
    * (default = 1.0)
    *
    * @group param
    */
  final val minNodeHess: DoubleParam =
    new DoubleParam(this, "minNodeHess", "Minimum sum of instance hessian needed in a node.",
      ParamValidators.gtEq(0.0))

  def getMinNodeHess: Double = $(minNodeHess)

  setDefault(minNodeHess -> 1.0)


  /**
    * L1 regularization term on weights, increase this value will make model more conservative.
    * (default = 0.0)
    *
    * @group param
    */
  final val regAlpha: DoubleParam =
    new DoubleParam(this, "regAlpha", "L1 regularization term on weights, increase this " +
      "value will make model more conservative.",
      ParamValidators.gtEq(0.0))

  def getRegAlpha: Double = $(regAlpha)

  setDefault(regAlpha -> 0.0)


  /**
    * L2 regularization term on weights, increase this value will make model more conservative.
    * (default = 1.0)
    *
    * @group param
    */
  final val regLambda: DoubleParam =
    new DoubleParam(this, "regLambda", "L2 regularization term on weights, increase this " +
      "value will make model more conservative.",
      ParamValidators.gtEq(0.0))

  def getRegLambda: Double = $(regLambda)

  setDefault(regLambda -> 1.0)


  /**
    * Indices of categorical features.
    *
    * @group param
    */
  final val catCols: IntArrayParam =
    new IntArrayParam(this, "catCols", "Indices of categorical features",
      (cols: Array[Int]) => cols.forall(_ >= 0) && cols.length == cols.distinct.length)

  def getCatCols: Array[Int] = $(catCols)

  setDefault(catCols -> Array.emptyIntArray)


  /**
    * Indices of ranking features.
    *
    * @group param
    */
  final val rankCols: IntArrayParam =
    new IntArrayParam(this, "rankCols", "Indices of ranking features",
      (cols: Array[Int]) => cols.forall(_ >= 0) && cols.length == cols.distinct.length)

  def getRankCols: Array[Int] = $(rankCols)

  setDefault(rankCols -> Array.emptyIntArray)


  /**
    * Subsample ratio of the training instance.
    * (default = 1.0)
    *
    * @group param
    */
  final val subSampleRate: DoubleParam =
    new DoubleParam(this, "subSampleRate", "Subsample ratio of the training instance.",
      ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))

  def getSubSampleRate: Double = $(subSampleRate)

  setDefault(subSampleRate -> 1.0)


  /**
    * Subsample ratio of the training instance when constructing each level.
    * (default = 1.0)
    *
    * @group param
    */
  final val subSampleRateByLevel: DoubleParam =
    new DoubleParam(this, "subSampleRateByLevel", "Subsample ratio of the training" +
      " instance when constructing each level.",
      ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))

  def getSubSampleRateByLevel: Double = $(subSampleRateByLevel)

  setDefault(subSampleRateByLevel -> 1.0)


  /**
    * Subsample ratio of columns when constructing each tree.
    * (default = 1.0)
    *
    * @group param
    */
  final val colSampleRateByTree: DoubleParam =
    new DoubleParam(this, "colSampleRateByTree", "Subsample ratio of columns " +
      "when constructing each tree.",
      ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))

  def getColSampleRateByTree: Double = $(colSampleRateByTree)

  setDefault(colSampleRateByTree -> 1.0)


  /**
    * Subsample ratio of columns when constructing each level.
    * (default = 1.0)
    *
    * @group param
    */
  final val colSampleRateByLevel: DoubleParam =
    new DoubleParam(this, "colSampleRateByLevel", "Subsample ratio of columns when " +
      "constructing each tree.",
      ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))

  def getColSampleRateByLevel: Double = $(colSampleRateByLevel)

  setDefault(colSampleRateByLevel -> 1.0)


  /**
    * StorageLevel for intermediate datasets.
    * (Default: "MEMORY_AND_DISK")
    *
    * @group expertParam
    */
  final val storageLevel1: Param[String] =
    new Param[String](this, "storageLevel1",
      "StorageLevel for intermediate datasets. Cannot be 'NONE'.",
      (s: String) => Try(StorageLevel.fromString(s)).isSuccess && s != "NONE")

  def getStorageLevel1: String = $(storageLevel1)

  setDefault(storageLevel1 -> "MEMORY_AND_DISK")


  /**
    * StorageLevel for intermediate datasets.
    * (Default: "MEMORY_AND_DISK_SER")
    *
    * @group expertParam
    */
  final val storageLevel2: Param[String] =
    new Param[String](this, "storageLevel2",
      "StorageLevel for intermediate datasets. Cannot be 'NONE'.",
      (s: String) => Try(StorageLevel.fromString(s)).isSuccess && s != "NONE")

  def getStorageLevel2: String = $(storageLevel2)

  setDefault(storageLevel2 -> "MEMORY_AND_DISK_SER")


  /**
    * StorageLevel for intermediate datasets.
    * (Default: "DISK_ONLY")
    *
    * @group expertParam
    */
  final val storageLevel3: Param[String] =
    new Param[String](this, "storageLevel3",
      "StorageLevel for intermediate datasets. Cannot be 'NONE'.",
      (s: String) => Try(StorageLevel.fromString(s)).isSuccess && s != "NONE")

  def getStorageLevel3: String = $(storageLevel3)

  setDefault(storageLevel3 -> "DISK_ONLY")


  /**
    * Boosting type.
    * (Default: "gbtree")
    *
    * @group expertParam
    */
  final val boostType: Param[String] =
    new Param[String](this, "boostType", "boosting type",
      ParamValidators.inArray[String](Array("gbtree", "dart")))

  def getBoostType: String = $(boostType)

  setDefault(boostType -> "gbtree")


  /**
    * Dropout rate in each iteration.
    * (default = 0.0)
    *
    * @group param
    */
  final val dropRate: DoubleParam =
    new DoubleParam(this, "dropRate", "Dropout rate in each iteration.",
      ParamValidators.inRange(0.0, 1.0))

  def getDropRate: Double = $(dropRate)

  setDefault(dropRate -> 0.0)


  /**
    * Probability of skipping drop.
    * (default = 0.5)
    *
    * @group param
    */
  final val dropSkip: DoubleParam =
    new DoubleParam(this, "dropSkip", "Probability of skipping drop.",
      ParamValidators.inRange(0.0, 1.0))

  def getDropSkip: Double = $(dropSkip)

  setDefault(dropSkip -> 0.5)


  /**
    * Minimum number of dropped trees in each iteration.
    * (default = 0)
    *
    * @group param
    */
  final val minDrop: IntParam =
    new IntParam(this, "minDrop", "Minimum number of dropped trees in each iteration.",
      ParamValidators.gtEq(0))

  def getMinDrop: Double = $(minDrop)

  setDefault(minDrop -> 0)


  /**
    * Maximum number of dropped trees in each iteration.
    * (default = 50)
    *
    * @group param
    */
  final val maxDrop: IntParam =
    new IntParam(this, "maxDrop", "Maximum number of dropped trees in each iteration.",
      ParamValidators.gtEq(0))

  def getMaxDrop: Double = $(maxDrop)

  setDefault(maxDrop -> 50)


  /**
    * Retain fraction of large gradient data in GOSS.
    * (default = 0.2)
    *
    * @group param
    */
  final val topRate: DoubleParam =
    new DoubleParam(this, "topRate", "Retain fraction of large gradient data in GOSS.",
      ParamValidators.inRange(0.0, 1.0, false, false))

  def getTopRate: Double = $(topRate)

  setDefault(topRate -> 0.2)


  /**
    * Retain fraction of small gradient data in GOSS.
    * (default = 0.1)
    *
    * @group param
    */
  final val otherRate: DoubleParam =
    new DoubleParam(this, "otherRate", "Retain fraction of small gradient data in GOSS.",
      ParamValidators.inRange(0.0, 1.0, false, false))

  def getOtherRate: Double = $(otherRate)

  setDefault(otherRate -> 0.1)


  /**
    * Path of initial model, set empty string to disable initial model.
    * (default = "")
    *
    * @group param
    */
  final val initialModelPath: Param[String] =
    new Param[String](this, "initialModelPath", "Path of initial model, set empty " +
      "string to disable initial model.")

  def getInitialModelPath: String = $(initialModelPath)

  setDefault(initialModelPath -> "")


  /**
    * Whether to encode the leaf indices in one-hot format.
    * (default = false)
    *
    * @group param
    */
  final val enableOneHot: BooleanParam =
    new BooleanParam(this, "enableOneHot", "Whether to encode the leaf indices " +
      "in one-hot format.")

  def getEnableOneHot: Boolean = $(enableOneHot)

  setDefault(enableOneHot -> false)


  /**
    * The number of first trees for prediction, leaf transformation, feature importance computation.
    * Use all trees if set -1.
    * (default = -1)
    *
    * @group param
    */
  final val firstTrees: IntParam =
    new IntParam(this, "firstTrees", "The number of first trees for prediction, " +
      "leaf transformation, feature importance computation. Use all trees if set -1.",
      ParamValidators.gtEq(-1))

  def getFirstTrees: Int = $(firstTrees)

  setDefault(firstTrees -> -1)


  /**
    * The maximum number of non-zero histogram bins to search split for categorical columns by brute force.
    * (default = 10)
    *
    * @group param
    */
  final val maxBruteBins: IntParam = new IntParam(this, "maxBruteBins", "The maximum number " +
    "of non-zero histogram bins to search split for categorical columns by brute force.",
    ParamValidators.inRange(0, 64))

  def getMaxBruteBins: Int = $(maxBruteBins)

  setDefault(maxBruteBins -> 10)


  /**
    * Number of iterations to stop training if one metric on validation data doesn't improve.
    * Set -1 to disable early stopping.
    * (default = 10)
    *
    * @group param
    */
  final val earlyStopIters: IntParam = new IntParam(this, "earlyStopIters", "Number of " +
    "iterations to stop training if one metric on validation data doesn't improve, set -1 to disable early stopping.",
    (value: Int) => value == -1 || value >= 1)

  def getEarlyStopIters: Int = $(earlyStopIters)

  setDefault(earlyStopIters -> -1)


  /**
    * Path for model checkpoint, set empty string to disable checkpoint.
    * (default = "")
    *
    * @group param
    */
  final val modelCheckpointPath: Param[String] =
    new Param[String](this, "modelCheckpointPath", "Path for model checkpoint, set empty " +
      "string to disable checkpoint.")

  def getModelCheckpointPath: String = $(modelCheckpointPath)

  setDefault(modelCheckpointPath -> "")


  /**
    * Model checkpoint interval, set -1 to disable checkpoint.
    * (default = -1)
    *
    * @group param
    */
  final val modelCheckpointInterval: IntParam =
    new IntParam(this, "modelCheckpointInterval", "Model checkpoint interval, set -1 " +
      "to disable checkpoint.",
      (value: Int) => value == -1 || value >= 1)

  def getModelCheckpointInterval: Int = $(modelCheckpointInterval)

  setDefault(modelCheckpointInterval -> -1)


  /**
    * Method to discretize numerical columns, set "width" for interval-equal discretization, "depth" for
    * quantile based discretization.
    * (default = "width")
    *
    * @group param
    */
  final val numericalBinType: Param[String] =
    new Param[String](this, "numericalBinType", "Method to discretize numerical columns, " +
      "set width for interval-equal discretization, depth for quantile based discretization.",
      ParamValidators.inArray[String](Array("width", "depth")))

  def getNumericalBinType: String = $(numericalBinType)

  setDefault(numericalBinType -> "width")


  /**
    * Float precision to represent internal gradient, hessian and prediction.
    * (default = "float")
    *
    * @group param
    */
  final val floatType: Param[String] =
    new Param[String](this, "floatType", "Float precision to represent internal " +
      "gradient, hessian and prediction.",
      ParamValidators.inArray[String](Array("float", "double")))

  def getFloatType: String = $(floatType)

  setDefault(floatType -> "float")


  /**
    * Whether zero is viewed as missing value.
    * (default = false)
    *
    * @group param
    */
  final val zeroAsMissing: BooleanParam =
    new BooleanParam(this, "zeroAsMissing", "Whether zero is viewed as missing value.")

  def getZeroAsMissing: Boolean = $(zeroAsMissing)

  setDefault(zeroAsMissing -> false)


  /**
    * Parallelism of histogram computation. If negative, means times of defaultParallelism of Spark.
    * (default = -1)
    *
    * @group param
    */
  final val reduceParallelism: DoubleParam =
    new DoubleParam(this, "reduceParallelism", "Parallelism of histogram computation. " +
      "If negative, means times of defaultParallelism of Spark.",
      (value: Double) => value != 0 && !value.isNaN && !value.isInfinity)

  def getReduceParallelism: Double = $(reduceParallelism)

  setDefault(reduceParallelism -> -1.0)


  /**
    * Method of data sampling.
    * (default = "block")
    *
    * @group param
    */
  final val subSampleType: Param[String] =
    new Param[String](this, "subSampleType", "Method of data sampling.",
      ParamValidators.inArray[String](Array("row", "block", "partition", "goss")))

  def getSubSampleType: String = $(subSampleType)

  setDefault(subSampleType -> "block")


  /**
    * Method to compute histograms.
    * (default = "subtract")
    *
    * @group param
    */
  final val histogramComputationType: Param[String] =
    new Param[String](this, "histogramComputationType", "Method to compute histograms.",
      ParamValidators.inArray[String](Array("basic", "subtract", "vote")))

  def getHistogramComputationType: String = $(histogramComputationType)

  setDefault(histogramComputationType -> "subtract")


  /**
    * Number of base models in one round.
    * (default = 1)
    *
    * @group param
    */
  final val baseModelParallelism: IntParam =
    new IntParam(this, "baseModelParallelism", "Number of base models in one round.",
      ParamValidators.gt(0))

  def getBaseModelParallelism: Int = $(baseModelParallelism)

  setDefault(baseModelParallelism -> 1)


  /**
    * Size of block.
    * (default = 4096)
    *
    * @group param
    */
  final val blockSize: IntParam =
    new IntParam(this, "blockSize", "Size of block.",
      ParamValidators.gt(0))

  def getBlockSize: Int = $(blockSize)

  setDefault(blockSize -> 4096)


  /**
    * Method to compute feature importance.
    * (default = "numsplits")
    *
    * @group param
    */
  final val importanceType: Param[String] =
    new Param[String](this, "importanceType", "Method to compute feature importance.",
      ParamValidators.inArray[String](Array("avggain", "sumgain", "numsplits")))

  def getImportanceType: String = $(importanceType)

  setDefault(importanceType -> "numsplits")
}

