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
    new DoubleParam(this, "minGain", "Minimum gain required to make a further partition on a leaf node of the tree.",
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
    new DoubleParam(this, "regAlpha", "L1 regularization term on weights, increase this value will make model more conservative.",
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
    new DoubleParam(this, "regLambda", "L2 regularization term on weights, increase this value will make model more conservative.",
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
  final val subSample: DoubleParam =
    new DoubleParam(this, "subSample", "Subsample ratio of the training instance.",
      ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))

  def getSubSample: Double = $(subSample)

  setDefault(subSample -> 1.0)


  /**
    * Subsample ratio of columns when constructing each tree.
    * (default = 1.0)
    *
    * @group param
    */
  final val colSampleByTree: DoubleParam =
    new DoubleParam(this, "colSampleByTree", "Subsample ratio of columns when constructing each tree.",
      ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))

  def getColSampleByTree: Double = $(colSampleByTree)

  setDefault(colSampleByTree -> 1.0)


  /**
    * Subsample ratio of columns when constructing each level.
    * (default = 1.0)
    *
    * @group param
    */
  final val colSampleByLevel: DoubleParam =
    new DoubleParam(this, "colSampleByLevel", "Subsample ratio of columns when constructing each tree.",
      ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))

  def getColSampleByLevel: Double = $(colSampleByLevel)

  setDefault(colSampleByLevel -> 1.0)


  /**
    * StorageLevel for intermediate datasets.
    * (Default: "MEMORY_AND_DISK")
    *
    * @group expertParam
    */
  final val storageLevel: Param[String] =
    new Param[String](this, "storageLevel",
      "StorageLevel for intermediate datasets. Cannot be 'NONE'.",
      (s: String) => Try(StorageLevel.fromString(s)).isSuccess && s != "NONE")

  def getStorageLevel: String = $(storageLevel)

  setDefault(storageLevel -> "MEMORY_AND_DISK")


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
    * Path of initial model, set empty string to disable initial model.
    * (default = "")
    *
    * @group param
    */
  final val initialModelPath: Param[String] =
    new Param[String](this, "initialModelPath", "Path of initial model, set empty string to disable initial model.")

  def getInitialModelPath: String = $(initialModelPath)

  setDefault(initialModelPath -> "")


  /**
    * Whether to encode the leaf indices in one-hot format.
    * (default = false)
    *
    * @group param
    */
  final val enableOneHot: BooleanParam =
    new BooleanParam(this, "enableOneHot", "Whether to encode the leaf indices in one-hot format.")

  def getEnableOneHot: Boolean = $(enableOneHot)

  setDefault(enableOneHot -> false)


  /**
    * The number of first trees for prediction, leaf transformation, feature importance computation. Use all trees if set -1.
    * (default = -1)
    *
    * @group param
    */
  final val firstTrees: IntParam =
    new IntParam(this, "firstTrees", "The number of first trees for prediction, leaf transformation, feature importance computation. Use all trees if set -1.",
      ParamValidators.gtEq(-1))

  def getFirstTrees: Int = $(firstTrees)

  setDefault(firstTrees -> -1)


  /**
    * The maximum number of non-zero histogram bins to search split for categorical columns by brute force.
    * (default = 10)
    *
    * @group param
    */
  final val maxBruteBins: IntParam = new IntParam(this, "maxBruteBins", "The maximum number of non-zero histogram bins to search split for categorical columns by brute force.",
    ParamValidators.gtEq(0))

  def getMaxBruteBins: Int = $(maxBruteBins)

  setDefault(maxBruteBins -> 10)


  /**
    * Number of iterations to stop training if one metric on validation data doesn't improve, set -1 to disable early stopping.
    * (default = 10)
    *
    * @group param
    */
  final val earlyStopIters: IntParam = new IntParam(this, "earlyStopIters", "Number of iterations to stop training if one metric on validation data doesn't improve, set -1 to disable early stopping.",
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
    new Param[String](this, "modelCheckpointPath", "Path for model checkpoint, set empty string to disable checkpoint.")

  def getModelCheckpointPath: String = $(modelCheckpointPath)

  setDefault(modelCheckpointPath -> "")


  /**
    * Model checkpoint interval, set -1 to disable checkpoint.
    * (default = -1)
    *
    * @group param
    */
  final val modelCheckpointInterval: IntParam =
    new IntParam(this, "modelCheckpointInterval", "Model checkpoint interval, set -1 to disable checkpoint.",
      (value: Int) => value == -1 || value >= 1)

  def getModelCheckpointInterval: Int = $(modelCheckpointInterval)

  setDefault(modelCheckpointInterval -> -1)


  /**
    * Method to discretize numerical columns, set "width" for interval-equal discretization, "depth" for quantile based discretization.
    * (default = "width")
    *
    * @group param
    */
  final val numericalBinType: Param[String] =
    new Param[String](this, "numericalBinType", "Method to discretize numerical columns, set width for interval-equal discretization, depth for quantile based discretization.",
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
    new Param[String](this, "floatType", "Float precision to represent internal gradient, hessian and prediction.",
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
    * Parallelism of histogram subtraction and split searching. If negative, means times of defaultParallelism of Spark.
    * (default = -1)
    *
    * @group param
    */
  final val parallelism: IntParam =
    new IntParam(this, "parallelism", "Parallelism of histogram subtraction and split searching. If negative, means times of defaultParallelism of Spark.",
      (value: Int) => value != 0)

  def getParallelism: Int = $(parallelism)

  setDefault(parallelism -> -1)


  /**
    * Whether to sample partitions instead of instances.
    * (default = false)
    *
    * @group param
    */
  final val enableSamplePartitions: BooleanParam =
    new BooleanParam(this, "enableSamplePartitions", "Whether to sample partitions instead of instances.")

  def getEnableSamplePartitions: Boolean = $(enableSamplePartitions)

  setDefault(enableSamplePartitions -> false)


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
}

