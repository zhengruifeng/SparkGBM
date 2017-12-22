package org.apache.spark.ml.gbm

import org.apache.spark.internal.Logging
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.collection.BitSet


class BoostConfig extends Logging with Serializable {

  private var maxIter: Int = _
  private var maxDepth: Int = _
  private var maxLeaves: Int = _
  private var minGain: Double = _
  private var minNodeHess: Double = _
  private var stepSize: Double = _
  private var regAlpha: Double = _
  private var regLambda: Double = _
  private var subSample: Double = _
  private var colSampleByTree: Double = _
  private var colSampleByLevel: Double = _
  private var maxBruteBins: Int = _
  private var checkpointInterval: Int = _
  private var storageLevel: StorageLevel = _
  private var boostType: String = _
  private var dropRate: Double = _
  private var dropSkip: Double = _
  private var minDrop: Int = _
  private var maxDrop: Int = _
  private var aggregationDepth: Int = _
  private var parallelism: Int = _
  private var seed: Long = _

  private var obj: ObjFunc = _
  private var increEvals: Array[IncrementalEvalFunc] = _
  private var batchEvals: Array[BatchEvalFunc] = _
  private var callbacks: Array[CallbackFunc] = _

  private var numCols: Int = _
  private var floatType: String = _
  private var baseScore: Double = _
  private var catCols: BitSet = _
  private var rankCols: BitSet = _
  private var handleSparsity: Boolean = _

  /** maximum number of iterations */
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
  private[gbm] def setMaxDepth(value: Int): this.type = {
    require(value >= 1)
    maxDepth = value
    this
  }

  def updateMaxDepth(value: Int): this.type = {
    logInfo(s"maxDepth was changed from $maxDepth to $value")
    setMaxDepth(value)
  }

  def getMaxDepth: Int = maxDepth


  /** maximum number of tree leaves */
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


  /** checkpoint interval */
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


  /** depth for treeAggregate */
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


  /** parallelism of histogram computation and leaves splitting */
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
  private[gbm] def setSeed(value: Long): this.type = {
    seed = value
    this
  }

  def updateSeed(value: Long): this.type = {
    logInfo(s"seed was changed from $seed to $value")
    setSeed(value)
  }

  def getSeed: Long = seed


  /** boosting type */
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


  /** the maximum number of non-zero histogram bins to search split for categorical columns by brute force */
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


  /** objective function */
  private[gbm] def setObjectiveFunc(value: ObjFunc): this.type = {
    require(value != null)
    obj = value
    this
  }

  def updateObjectiveFunc(value: ObjFunc): this.type = {
    logInfo(s"obj was changed from ${obj.name} to ${value.name}")
    setObjectiveFunc(value)
  }

  def getObjectiveFunc: ObjFunc = obj


  /** evaluation functions */
  private[gbm] def setEvaluateFunc(value: Array[EvalFunc]): this.type = {
    require(value.forall(e => e.isInstanceOf[IncrementalEvalFunc] || e.isInstanceOf[BatchEvalFunc]))
    require(value.map(_.name).distinct.length == value.length)
    increEvals = value.flatMap {
      case eval: IncrementalEvalFunc =>
        Some(eval)
      case _ => None
    }
    batchEvals = value.flatMap {
      case eval: BatchEvalFunc =>
        Some(eval)
      case _ => None
    }

    this
  }

  def updateEvaluateFunc(value: Array[EvalFunc]): this.type = {
    logInfo(s"eval was changed from ${getEvaluateFunc.map(_.name).mkString("(", ",", ")")} to ${value.map(_.name).mkString("(", ",", ")")}")
    setEvaluateFunc(value)
  }

  def getEvaluateFunc: Array[EvalFunc] = increEvals ++ batchEvals

  def getIncrementalEvaluateFunc: Array[IncrementalEvalFunc] = increEvals

  def getBatchEvaluateFunc: Array[BatchEvalFunc] = batchEvals


  /** callback functions */
  private[gbm] def setCallbackFunc(value: Array[CallbackFunc]): this.type = {
    require(value.map(_.name).distinct.length == value.length)
    callbacks = value
    this
  }

  def updateCallbackFunc(value: Array[CallbackFunc]): this.type = {
    logInfo(s"obj was changed from ${callbacks.map(_.name).mkString("(", ",", ")")} to ${value.map(_.name).mkString("(", ",", ")")}")
    setCallbackFunc(value)
  }

  def getCallbackFunc: Array[CallbackFunc] = callbacks


  /** number of columns */
  private[gbm] def setNumCols(value: Int): this.type = {
    require(value > 0)
    numCols = value
    this
  }

  def getNumCols: Int = numCols


  /** float precision */
  private[gbm] def setFloatType(value: String): this.type = {
    require(value == "float" || value == "double")
    floatType = value
    this
  }

  def getFloatType: String = floatType


  /** base score for global bias */
  private[gbm] def setBaseScore(value: Double): this.type = {
    require(!value.isNaN && !value.isInfinity)
    baseScore = value
    this
  }

  def getBaseScore: Double = baseScore


  /** indices of categorical columns */
  private[gbm] def setCatCols(value: BitSet): this.type = {
    catCols = value
    this
  }

  def getCatCols: BitSet = catCols


  /** indices of ranking columns */
  private[gbm] def setRankCols(value: BitSet): this.type = {
    rankCols = value
    this
  }

  def getRankCols: BitSet = rankCols


  /** whether to store the bins in a sparse fashion */
  private[gbm] def setHandleSparsity(value: Boolean): this.type = {
    handleSparsity = value
    this
  }

  def getHandleSparsity: Boolean = handleSparsity


  private[gbm] def isNum(colIndex: Int): Boolean = !isCat(colIndex) && !isRank(colIndex)

  private[gbm] def isCat(colIndex: Int): Boolean = catCols.get(colIndex)

  private[gbm] def isRank(colIndex: Int): Boolean = rankCols.get(colIndex)
}


private[gbm] class TreeConfig(val iteration: Int,
                              val treeIndex: Int,
                              val catCols: BitSet,
                              val columns: Array[Int]) extends Serializable {

  def isSeq(colIndex: Int): Boolean = !catCols.get(colIndex)

  def numCols: Int = columns.length
}




