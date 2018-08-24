package org.apache.spark.ml.gbm

import scala.collection.BitSet

import org.apache.spark.internal.Logging
import org.apache.spark.storage.StorageLevel


class BoostConfig extends Logging with Serializable {

  /** maximum number of iterations */
  private var maxIter: Int = 20

  private[gbm] def setMaxIter(value: Int): this.type = {
    require(value >= 0)
    maxIter = value
    this
  }

  def updateMaxIter(value: Int): this.type = {
    logInfo(s"maxIter was changed from $maxIter to $value")
    setMaxIter(value)
  }

  def getMaxIter: Int = maxIter


  /** maximum tree depth */
  private var maxDepth: Int = 5

  private[gbm] def setMaxDepth(value: Int): this.type = {
    require(value >= 1 && value <= 30)
    maxDepth = value
    this
  }

  def updateMaxDepth(value: Int): this.type = {
    logInfo(s"maxDepth was changed from $maxDepth to $value")
    setMaxDepth(value)
  }

  def getMaxDepth: Int = maxDepth


  /** maximum number of tree leaves */
  private var maxLeaves: Int = 1000

  private[gbm] def setMaxLeaves(value: Int): this.type = {
    require(value >= 2)
    maxLeaves = value
    this
  }

  def updateMaxLeaves(value: Int): this.type = {
    logInfo(s"maxLeaves was changed from $maxLeaves to $value")
    setMaxLeaves(value)
  }

  def getMaxLeaves: Int = maxLeaves


  /** minimum gain for each split */
  private var minGain: Double = 0.0

  private[gbm] def setMinGain(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    minGain = value
    this
  }

  def updateMinGain(value: Double): this.type = {
    logInfo(s"minGain was changed from $minGain to $value")
    setMinGain(value)
  }

  def getMinGain: Double = minGain


  /** minimum sum of hess for each node */
  private var minNodeHess: Double = 1.0

  private[gbm] def setMinNodeHess(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    minNodeHess = value
    this
  }

  def updateMinNodeHess(value: Double): this.type = {
    logInfo(s"minNodeHess was changed from $minNodeHess to $value")
    setMinNodeHess(value)
  }

  def getMinNodeHess: Double = minNodeHess


  /** learning rate */
  private var stepSize: Double = 0.1

  private[gbm] def setStepSize(value: Double): this.type = {
    require(value > 0 && !value.isNaN && !value.isInfinity)
    stepSize = value
    this
  }

  def updateStepSize(value: Double): this.type = {
    logInfo(s"stepSize was changed from $stepSize to $value")
    setStepSize(value)
  }

  def getStepSize: Double = stepSize


  /** L1 regularization term on weights */
  private var regAlpha: Double = 0.0

  private[gbm] def setRegAlpha(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    regAlpha = value
    this
  }

  def updateRegAlpha(value: Double): this.type = {
    logInfo(s"regAlpha was changed from $regAlpha to $value")
    setRegAlpha(value)
  }

  def getRegAlpha: Double = regAlpha


  /** L2 regularization term on weights */
  private var regLambda: Double = 1.0

  private[gbm] def setRegLambda(value: Double): this.type = {
    require(value >= 0 && !value.isNaN && !value.isInfinity)
    regLambda = value
    this
  }

  def updateRegLambda(value: Double): this.type = {
    logInfo(s"regLambda was changed from $regLambda to $value")
    setRegLambda(value)
  }

  def getRegLambda: Double = regLambda


  /** subsample ratio of the training instance */
  private var subSample: Double = 1.0

  private[gbm] def setSubSample(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    subSample = value
    this
  }

  def updateSubSample(value: Double): this.type = {
    logInfo(s"subSample was changed from $subSample to $value")
    setSubSample(value)
  }

  def getSubSample: Double = subSample


  /** subsample ratio of columns when constructing each tree */
  private var colSampleByTree: Double = 1.0

  private[gbm] def setColSampleByTree(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    colSampleByTree = value
    this
  }

  def updateColSampleByTree(value: Double): this.type = {
    logInfo(s"colSampleByTree was changed from $colSampleByTree to $value")
    setColSampleByTree(value)
  }

  def getColSampleByTree: Double = colSampleByTree


  /** subsample ratio of columns when constructing each level */
  private var colSampleByLevel: Double = 1.0

  private[gbm] def setColSampleByLevel(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    colSampleByLevel = value
    this
  }

  def updateColSampleByLevel(value: Double): this.type = {
    logInfo(s"colSampleByLevel was changed from $colSampleByLevel to $value")
    setColSampleByLevel(value)
  }

  def getColSampleByLevel: Double = colSampleByLevel


  /** the maximum number of non-zero histogram bins to search split for categorical columns by brute force */
  private var maxBruteBins: Int = 10

  private[gbm] def setMaxBruteBins(value: Int): this.type = {
    require(value >= 0)
    maxBruteBins = value
    this
  }

  def updateMaxBruteBins(value: Int): this.type = {
    logInfo(s"maxBruteBins was changed from $maxBruteBins to $value")
    setMaxBruteBins(value)
  }

  def getMaxBruteBins: Int = maxBruteBins


  /** checkpoint interval */
  private var checkpointInterval: Int = 10

  private[gbm] def setCheckpointInterval(value: Int): this.type = {
    require(value == -1 || value > 0)
    checkpointInterval = value
    this
  }

  def updateCheckpointInterval(value: Int): this.type = {
    logInfo(s"checkpointInterval was changed from $checkpointInterval to $value")
    setCheckpointInterval(value)
  }

  def getCheckpointInterval: Int = checkpointInterval


  /** storage level */
  private var storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  private[gbm] def setStorageLevel(value: StorageLevel): this.type = {
    require(value != StorageLevel.NONE)
    storageLevel = value
    this
  }

  def updateStorageLevel(value: StorageLevel): this.type = {
    logInfo(s"storageLevel was changed from $storageLevel to $value")
    setStorageLevel(value)
  }

  def getStorageLevel: StorageLevel = storageLevel


  /** boosting type */
  private var boostType: String = "gbtree"

  private[gbm] def setBoostType(value: String): this.type = {
    require(value == "gbtree" || value == "dart")
    boostType = value
    this
  }

  def updateBoostType(value: String): this.type = {
    logInfo(s"boostType was changed from $boostType to $value")
    setBoostType(value)
  }

  def getBoostType: String = boostType


  /** dropout rate */
  private var dropRate: Double = 0.0

  private[gbm] def setDropRate(value: Double): this.type = {
    require(value >= 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    dropRate = value
    this
  }

  def updateDropRate(value: Double): this.type = {
    logInfo(s"dropRate was changed from $dropRate to $value")
    setDropRate(value)
  }

  def getDropRate: Double = dropRate


  /** probability of skipping drop */
  private var dropSkip: Double = 0.5

  private[gbm] def setDropSkip(value: Double): this.type = {
    require(value >= 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    dropSkip = value
    this
  }

  def updateDropSkip(value: Double): this.type = {
    logInfo(s"dropSkip was changed from $dropSkip to $value")
    setDropSkip(value)
  }

  def getDropSkip: Double = dropSkip


  /** minimum number of dropped trees in each iteration */
  private var minDrop: Int = 0

  private[gbm] def setMinDrop(value: Int): this.type = {
    require(value >= 0)
    minDrop = value
    this
  }

  def updateMinDrop(value: Int): this.type = {
    logInfo(s"minDrop was changed from $minDrop to $value")
    setMinDrop(value)
  }

  def getMinDrop: Int = minDrop


  /** maximum number of dropped trees in each iteration */
  private var maxDrop: Int = 50

  private[gbm] def setMaxDrop(value: Int): this.type = {
    require(value >= 0)
    maxDrop = value
    this
  }

  def updateMaxDrop(value: Int): this.type = {
    logInfo(s"maxDrop was changed from $maxDrop to $value")
    setMaxDrop(value)
  }

  def getMaxDrop: Int = maxDrop


  /** depth for treeAggregate */
  private var aggregationDepth: Int = 2

  private[gbm] def setAggregationDepth(value: Int): this.type = {
    require(value >= 2)
    aggregationDepth = value
    this
  }

  def updateAggregationDepth(value: Int): this.type = {
    logInfo(s"aggregationDepth was changed from $aggregationDepth to $value")
    setAggregationDepth(value)
  }

  def getAggregationDepth: Int = aggregationDepth


  /** whether to sample partitions instead of instances if possible */
  private var enableSamplePartitions: Boolean = false

  private[gbm] def setEnableSamplePartitions(value: Boolean): this.type = {
    enableSamplePartitions = value
    this
  }

  def updateEnableSamplePartitions(value: Boolean): this.type = {
    logInfo(s"enableSamplePartitions was changed from $enableSamplePartitions to $value")
    setEnableSamplePartitions(value)
  }

  def getEnableSamplePartitions: Boolean = enableSamplePartitions


  /** parallelism of histogram computation and leaves splitting */
  private var parallelism: Int = -1

  private[gbm] def setParallelism(value: Int): this.type = {
    require(value != 0)
    parallelism = value
    this
  }

  def updateParallelism(value: Long): this.type = {
    logInfo(s"parallelism was changed from $parallelism to $value")
    setSeed(value)
  }

  def getParallelism: Int = parallelism

  private[gbm] def getRealParallelism(value: Int): Int = {
    require(value > 0)
    if (parallelism > 0) {
      parallelism
    } else {
      parallelism.abs * value
    }
  }


  /** random number seed */
  private var seed: Long = -1L

  private[gbm] def setSeed(value: Long): this.type = {
    seed = value
    this
  }

  def updateSeed(value: Long): this.type = {
    logInfo(s"seed was changed from $seed to $value")
    setSeed(value)
  }

  def getSeed: Long = seed


  /** scalar objective function */
  private var objFunc: ObjFunc = new SquareObj

  private[gbm] def setObjFunc(value: ObjFunc): this.type = {
    require(value != null)
    objFunc = value
    this
  }

  def updateObjFunc(value: ObjFunc): this.type = {
    logInfo(s"scalarObjFunc was changed from ${objFunc.name} to ${value.name}")
    setObjFunc(value)
  }

  def getObjFunc: ObjFunc = objFunc


  /** scalar incremental evaluation functions */
  private var evalFunc: Array[EvalFunc] = Array.empty

  private[gbm] def setEvalFunc(value: Array[EvalFunc]): this.type = {
    require(value != null)
    evalFunc = value
    this
  }

  def updateEvalFunc(value: Array[EvalFunc]): this.type = {
    logInfo(s"scalarIncEvalFunc was changed from ${evalFunc.map(_.name)} to ${value.map(_.name)}")
    setEvalFunc(value)
  }

  def getEvalFunc: Array[EvalFunc] = evalFunc

  def getBatchEvalFunc: Array[EvalFunc] = {
    evalFunc.flatMap {
      case _: IncEvalFunc => Iterator.empty
      case e => Iterator.single(e)
    }
  }

  def getIncEvalFunc: Array[IncEvalFunc] = {
    evalFunc.flatMap {
      case e: IncEvalFunc => Iterator.single(e)
      case _ => Iterator.empty
    }
  }


  /** callback functions */
  private var callbackFunc: Array[CallbackFunc] = Array.empty

  private[gbm] def setCallbackFunc(value: Array[CallbackFunc]): this.type = {
    require(value.map(_.name).distinct.length == value.length)
    callbackFunc = value
    this
  }

  def updateCallbackFunc(value: Array[CallbackFunc]): this.type = {
    logInfo(s"callbackFunc was changed from ${callbackFunc.map(_.name).mkString("(", ",", ")")} to ${value.map(_.name).mkString("(", ",", ")")}")
    setCallbackFunc(value)
  }

  def getCallbackFunc: Array[CallbackFunc] = callbackFunc


  /** base score for global scalar bias */
  private var baseScore: Array[Double] = Array.empty

  private[gbm] def setBaseScore(value: Array[Double]): this.type = {
    require(value.nonEmpty)
    require(value.forall(v => !v.isNaN && !v.isInfinity))
    baseScore = value
    this
  }

  def getBaseScore: Array[Double] = baseScore

  def computeRawBaseScore: Array[Double] = objFunc.inverseTransform(baseScore)


  /** length of raw prediction vectors */
  private var rawSize: Int = 0

  private[gbm] def setRawSize(value: Int): this.type = {
    require(value > 0)
    rawSize = value
    this
  }

  def getRawSize: Int = rawSize


  /** number of base models in one round */
  private var baseModelParallelism: Int = 1

  private[gbm] def setBaseModelParallelism(value: Int): this.type = {
    require(value > 0)
    baseModelParallelism = value
    this
  }

  def updateBaseModelParallelism(value: Int): this.type = {
    logInfo(s"numParallelBaseModels was changed from $baseModelParallelism to $value")
    setBaseModelParallelism(value)
  }

  def getBaseModelParallelism: Int = baseModelParallelism


  /** Double precision */
  private var floatType: String = "Double"

  private[gbm] def setFloatType(value: String): this.type = {
    require(value == "Double" || value == "double")
    floatType = value
    this
  }

  def getFloatType: String = floatType


  /** number of columns */
  private var numCols: Int = -1

  private[gbm] def setNumCols(value: Int): this.type = {
    require(value > 0)
    numCols = value
    this
  }

  def getNumCols: Int = numCols


  /** indices of categorical columns */
  private var catCols: BitSet = BitSet.empty

  private[gbm] def setCatCols(value: BitSet): this.type = {
    catCols = value
    this
  }

  def getCatCols: BitSet = catCols


  /** indices of ranking columns */
  private var rankCols: BitSet = BitSet.empty

  private[gbm] def setRankCols(value: BitSet): this.type = {
    rankCols = value
    this
  }

  def getRankCols: BitSet = rankCols


  private[gbm] def isNum(colId: Int): Boolean = !isCat(colId) && !isRank(colId)

  private[gbm] def isCat(colId: Int): Boolean = catCols.contains(colId)

  private[gbm] def isRank(colId: Int): Boolean = rankCols.contains(colId)

  private[gbm] def isSeq(colId: Int): Boolean = !isCat(colId)
}


private[gbm] class BaseConfig(val iteration: Int,
                              val numTrees: Int,
                              val colSelectors: Array[ColumSelector]) extends Serializable {

  def getSelector(index: Int): ColumSelector = {
    if (colSelectors.nonEmpty) {
      colSelectors(index)
    } else {
      TotalSelector()
    }
  }
}




