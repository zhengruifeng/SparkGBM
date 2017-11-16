package org.apache.spark.ml.gbm

import org.apache.spark.storage.StorageLevel

class BoostConfig(var maxIter: Int,
                  var maxDepth: Int,
                  var maxLeaves: Int,
                  var minGain: Double,
                  var minNodeHess: Double,
                  var stepSize: Double,
                  var regAlpha: Double,
                  var regLambda: Double,
                  var subSample: Double,
                  var colSampleByTree: Double,
                  var colSampleByLevel: Double,
                  var maxBruteBins: Int,
                  var checkpointInterval: Int,
                  var storageLevel: StorageLevel,
                  var boostType: String,
                  var dropRate: Double,
                  var dropSkip: Double,
                  var minDrop: Int,
                  var maxDrop: Int,
                  var aggregationDepth: Int,
                  var seed: Long,
                  var obj: ObjFunc,
                  var increEvals: Array[IncrementalEvalFunc],
                  var batchEvals: Array[BatchEvalFunc],
                  var callbacks: Array[CallbackFunc],

                  val numCols: Int,
                  val baseScore: Double,
                  val catCols: Set[Int],
                  val rankCols: Set[Int],
                  val initialModel: Option[GBMModel]) extends Serializable {

  def isNum(colIndex: Int): Boolean = !isCat(colIndex) && !isRank(colIndex)

  def isCat(colIndex: Int): Boolean = catCols.contains(colIndex)

  def isRank(colIndex: Int): Boolean = rankCols.contains(colIndex)
}


private[gbm] class TreeConfig(val iteration: Int,
                              val treeIndex: Int,
                              val catCols: Set[Int],
                              val colsMap: Array[Int]) extends Serializable {

  def isSeq(colIndex: Int): Boolean = !catCols.contains(colIndex)

  def numCols: Int = colsMap.length
}




