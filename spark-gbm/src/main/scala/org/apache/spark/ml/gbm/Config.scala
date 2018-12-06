package org.apache.spark.ml.gbm

import scala.collection.BitSet
import scala.reflect.ClassTag

import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm.func._
import org.apache.spark.ml.gbm.util.Selector
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel


class BoostConfig extends Logging with Serializable {


  @transient private var spark: SparkSession = null

  private[gbm] def setSparkSession(value: SparkSession): this.type = {
    spark = value
    this
  }

  private[gbm] def getSparkSession: SparkSession = spark


  var realParallelism = -1

  private[gbm] def updateParallelismInfo(): Unit = {
    realParallelism = if (reduceParallelism > 0) {
      reduceParallelism.ceil.toInt
    } else {
      (spark.sparkContext.defaultParallelism * reduceParallelism.abs).ceil.toInt
    }

    require(realParallelism > 0)
  }

  private[gbm] def getRealParallelism: Int = realParallelism


  /** parallelism type */
  private var parallelismType: String = "data"

  private[gbm] def setParallelismType(value: String): this.type = {
    require(Array("data", "feature").contains(value))
    parallelismType = value
    this
  }

  def getParallelismType: String = parallelismType


  /** whether to update prediction and gradient after each level */
  private var greedierSearch: Boolean = false

  private[gbm] def setGreedierSearch(value: Boolean): this.type = {
    greedierSearch = value
    this
  }

  def getGreedierSearch: Boolean = greedierSearch


  /** boosting type */
  private var boostType: String = "gbtree"

  private[gbm] def setBoostType(value: String): this.type = {
    require(Array("gbtree", "dart").contains(value))
    boostType = value
    this
  }

  def getBoostType: String = boostType


  /** size of block */
  private var blockSize: Int = 4096

  private[gbm] def setBlockSize(value: Int): this.type = {
    require(value > 0)
    blockSize = value
    this
  }

  def getBlockSize: Int = blockSize


  /** top K in voting parallelism */
  private var topK: Int = 20

  private[gbm] def setTopK(value: Int): this.type = {
    require(value > 0)
    topK = value
    this
  }

  def updateTopK(value: Int): this.type = {
    logInfo(s"topK was changed from $topK to $value")
    setTopK(value)
  }

  def getTopK: Int = topK


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


  /** subsample ratio of the training instance when constructing each tree */
  private var subSampleRate: Double = 1.0

  private[gbm] def setSubSampleRate(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    subSampleRate = value
    this
  }

  def updateSubSampleRate(value: Double): this.type = {
    logInfo(s"subSampleRate was changed from $subSampleRate to $value")
    setSubSampleRate(value)
  }

  def getSubSampleRate: Double = subSampleRate


  /** subsample ratio of the training instance when constructing each level */
  private var subSampleRateByLevel: Double = 1.0

  private[gbm] def setSubSampleRateByLevel(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    subSampleRateByLevel = value
    this
  }

  def updateSubSampleRateByLevel(value: Double): this.type = {
    logInfo(s"subSampleRateByLevel was changed from $subSampleRateByLevel to $value")
    setSubSampleRateByLevel(value)
  }

  def getSubSampleRateByLevel: Double = subSampleRateByLevel


  /** retain fraction of large gradient data */
  private var topRate: Double = 0.2

  private[gbm] def setTopRate(value: Double): this.type = {
    require(value > 0 && value < 1 && !value.isNaN && !value.isInfinity)
    topRate = value
    this
  }

  def updateTopRate(value: Double): this.type = {
    logInfo(s"topRate was changed from $topRate to $value")
    setTopRate(value)
  }

  def getTopRate: Double = topRate


  /** retain fraction of small gradient data */
  private var otherRate: Double = 0.2

  def setOtherRate(value: Double): this.type = {
    require(value > 0 && value < 1 && !value.isNaN && !value.isInfinity)
    otherRate = value
    this
  }

  def updateOtherRate(value: Double): this.type = {
    logInfo(s"otherRate was changed from $otherRate to $value")
    setOtherRate(value)
  }

  def getOtherRate: Double = otherRate


  private[gbm] def getOtherReweight: Double = (1 - getTopRate) / getOtherRate

  private[gbm] def getOtherSampleRate: Double = getOtherRate / (1 - getTopRate)


  /** subsample ratio of columns when constructing each tree */
  private var colSampleRateByTree: Double = 1.0

  private[gbm] def setColSampleRateByTree(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    colSampleRateByTree = value
    this
  }

  def updateColSampleRateByTree(value: Double): this.type = {
    logInfo(s"colSampleRateByTree was changed from $colSampleRateByTree to $value")
    setColSampleRateByTree(value)
  }

  def getColSampleRateByTree: Double = colSampleRateByTree


  /** subsample ratio of columns when constructing each level */
  private var colSampleRateByLevel: Double = 1.0

  private[gbm] def setColSampleRateByLevel(value: Double): this.type = {
    require(value > 0 && value <= 1 && !value.isNaN && !value.isInfinity)
    colSampleRateByLevel = value
    this
  }

  def updateColSampleRateByLevel(value: Double): this.type = {
    logInfo(s"colSampleRateByLevel was changed from $colSampleRateByLevel to $value")
    setColSampleRateByLevel(value)
  }

  def getColSampleRateByLevel: Double = colSampleRateByLevel


  /** the maximum number of non-zero histogram bins to search split for categorical columns by brute force */
  private var maxBruteBins: Int = 10

  private[gbm] def setMaxBruteBins(value: Int): this.type = {
    require(value >= 0 && value <= 64)
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


  /** storage level 1 */
  private var storageLevel1: StorageLevel = StorageLevel.MEMORY_AND_DISK

  private[gbm] def setStorageLevel1(value: StorageLevel): this.type = {
    require(value != StorageLevel.NONE)
    storageLevel1 = value
    this
  }

  def getStorageLevel1: StorageLevel = storageLevel1


  /** storage level 2 */
  private var storageLevel2: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER

  private[gbm] def setStorageLevel2(value: StorageLevel): this.type = {
    require(value != StorageLevel.NONE)
    storageLevel2 = value
    this
  }

  def getStorageLevel2: StorageLevel = storageLevel2


  /** storage level 3 */
  private var storageLevel3: StorageLevel = StorageLevel.DISK_ONLY

  private[gbm] def setStorageLevel3(value: StorageLevel): this.type = {
    require(value != StorageLevel.NONE)
    storageLevel3 = value
    this
  }

  def getStorageLevel3: StorageLevel = storageLevel3


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


  /** method of data sampling */
  private var subSampleType: String = "block"

  private[gbm] def setSubSampleType(value: String): this.type = {
    require(Array("row", "block", "partition", "goss").contains(value))
    subSampleType = value
    this
  }

  def updateSubSampleType(value: String): this.type = {
    logInfo(s"subsampleMethod was changed from $subSampleType to $value")
    setSubSampleType(value)
    this
  }

  def getSubSampleType: String = subSampleType


  /** method to compute histograms */
  private var histogramComputationType: String = "subtract"

  private[gbm] def setHistogramComputationType(value: String): this.type = {
    require(Array("basic", "subtract", "vote").contains(value))
    histogramComputationType = value
    this
  }

  def updateHistogramComputationType(value: String): this.type = {
    logInfo(s"histogramComputationMethod was changed from $histogramComputationType to $value")
    setHistogramComputationType(value)
    this
  }

  def getHistogramComputationType: String = histogramComputationType


  /** parallelism of histogram computation */
  private var reduceParallelism: Double = -1.0

  private[gbm] def setReduceParallelism(value: Double): this.type = {
    require(value != 0 && !value.isNaN && !value.isInfinity)
    reduceParallelism = value
    this
  }

  def getReduceParallelism: Double = reduceParallelism


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
    require(value != null && value.map(_.name).distinct.length == value.length)
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
    require(value != null && value.map(_.name).distinct.length == value.length)
    callbackFunc = value
    this
  }

  def updateCallbackFunc(value: Array[CallbackFunc]): this.type = {
    logInfo(s"callbackFunc was changed from ${callbackFunc.map(_.name).mkString("(", ",", ")")} " +
      s"to ${value.map(_.name).mkString("(", ",", ")")}")
    setCallbackFunc(value)
  }

  def getCallbackFunc: Array[CallbackFunc] = callbackFunc


  /** base score for global bias */
  private var baseScore: Array[Double] = Array.empty

  private[gbm] def setBaseScore(value: Array[Double]): this.type = {
    require(value.forall(v => !v.isNaN && !v.isInfinity))
    baseScore = value
    this
  }

  def getBaseScore: Array[Double] = baseScore.clone()


  /** base raw prediction for global bias */
  private var rawBaseScore: Array[Double] = Array.empty

  private[gbm] def getRawBaseScore: Array[Double] = rawBaseScore.clone()


  /** length of raw prediction vectors */
  private var rawSize: Int = 0

  def getRawSize: Int = rawSize


  /** helper function to convert baseIds to treeIds */
  private[gbm] def computeTreeIds[T]()
                                    (implicit ct: ClassTag[T], int: Integral[T]): Array[T] => Array[T] = {
    if (rawSize == 1) {
      baseIds: Array[T] => baseIds
    } else {
      baseIds: Array[T] =>
        baseIds.flatMap { i =>
          val offset = rawSize * int.toInt(i)
          Iterator.range(offset, offset + rawSize).map(int.fromInt)
        }
    }
  }


  def updateRawBaseScoreInfo(): Unit = {
    require(baseScore.nonEmpty)
    rawBaseScore = objFunc.inverseTransform(baseScore)
    rawSize = rawBaseScore.length
  }


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


  /** number of trees in one round */
  def getNumTrees: Int = getBaseModelParallelism * getRawSize


  /** numerical precision */
  private var floatType: String = "float"

  private[gbm] def setFloatType(value: String): this.type = {
    require(value == "float" || value == "double")
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


  private var numBlocksPerPart: Array[Long] = Array.emptyLongArray

  private[gbm] def setNumBlocksPerPart(value: Array[Long]): this.type = {
    require(value.nonEmpty)
    numBlocksPerPart = value
    this
  }

  private[gbm] def getNumBlocksPerPart: Array[Long] = numBlocksPerPart

  private[gbm] def getNumBlocks: Long = numBlocksPerPart.sum

  private[gbm] def getBlockOffsetPerPart: Array[Long] = {
    if (numBlocksPerPart.nonEmpty) {
      numBlocksPerPart.take(numBlocksPerPart.length - 1).scanLeft(0L)(_ + _)
    } else {
      Array.emptyLongArray
    }
  }

  private[gbm] def getBlockOffset(partId: Int): Long = {
    var offset = 0L
    var i = 0
    while (i < partId) {
      offset += numBlocksPerPart(i)
      i += 1
    }
    offset
  }


  private[gbm] def updateVPartInfo(): Unit = {
    require(numCols > 0)
    require(realParallelism > 0)

    numVLayers = (realParallelism.toDouble / numCols).ceil.toInt
    require(numVLayers >= 1)

    val n = realParallelism / numVLayers

    colIdsPerBaseVPart = if (n == numCols) {
      Array.tabulate(numCols)(Array(_))
    } else {
      Array.tabulate(n)(i => Iterator.range(0, numCols).filter(_ % n == i).toArray)
    }
  }


  private[gbm] var numVLayers: Int = -1

  private[gbm] def getNumVLayers: Int = numVLayers

  private[gbm] var colIdsPerBaseVPart: Array[Array[Int]] = Array.empty

  private[gbm] def getColIdsPerBaseVPart: Array[Array[Int]] = colIdsPerBaseVPart

  private[gbm] def getColIdsPerVPart: Array[Array[Int]] = {
    require(numVLayers >= 1)
    if (numVLayers == 1) {
      colIdsPerBaseVPart
    } else {
      Array.range(0, numVLayers).flatMap(_ => colIdsPerBaseVPart)
    }
  }

  private[gbm] def getNumBaseVParts: Int = colIdsPerBaseVPart.length

  private[gbm] def getNumVParts: Int = numVLayers * colIdsPerBaseVPart.length

  private[gbm] def getVBlockOffset(vPartId: Int): Long = {
    require(numVLayers >= 1)
    if (numVLayers == 1) {
      0L
    } else {
      (vPartId / colIdsPerBaseVPart.length).toLong
    }
  }

  private[gbm] def getBaseVCols[C]()
                                  (implicit cc: ClassTag[C], inc: Integral[C]): Array[Array[C]] = {
    colIdsPerBaseVPart.map(_.map(inc.fromInt))
  }

  private[gbm] def getVCols[C]()
                              (implicit cc: ClassTag[C], inc: Integral[C]): Array[Array[C]] = {
    getColIdsPerVPart.map(_.map(inc.fromInt))
  }

  private[gbm] def getVCols[C](vPartId: Int)
                              (implicit cc: ClassTag[C], inc: Integral[C]): Array[C] = {
    colIdsPerBaseVPart(vPartId % colIdsPerBaseVPart.length).map(inc.fromInt)
  }
}


private[gbm] class BaseConfig(val iteration: Int,
                              val colSelector: Selector) extends Serializable


private[gbm] object BaseConfig extends Serializable {


  def create(boostConf: BoostConfig,
             iteration: Int,
             seed: Long): BaseConfig = {

    val colSelector = Selector.create(boostConf.getColSampleRateByTree, boostConf.getNumCols,
      boostConf.getBaseModelParallelism, boostConf.getRawSize, seed)


    new BaseConfig(iteration, colSelector)
  }


  /**
    * The default `BaseConfig` passed in `Tree` contains selector for tree-wise column sampling.
    * Call this function to merge level-wise column sampling if needed.
    */
  def mergeColSamplingByLevel(boostConf: BoostConfig,
                              baseConf: BaseConfig,
                              depth: Int): BaseConfig = {

    if (boostConf.getColSampleRateByLevel == 1) {
      baseConf
    } else {
      val levelSelector = Selector.create(boostConf.getColSampleRateByLevel, boostConf.getNumCols,
        boostConf.getBaseModelParallelism, boostConf.getRawSize, boostConf.getSeed * baseConf.iteration + depth)
      val unionSelector = Selector.union(baseConf.colSelector, levelSelector)
      new BaseConfig(baseConf.iteration, unionSelector)
    }
  }
}



