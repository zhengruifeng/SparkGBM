package org.apache.spark.ml.gbm

import org.apache.spark.storage.StorageLevel

private[gbm] class BoostConfig(val maxIter: Int,
                               val maxDepth: Int,
                               val maxLeaves: Int,
                               val numCols: Int,

                               val baseScore: Double,
                               val minGain: Double,
                               val minNodeHess: Double,

                               val stepSize: Double,
                               val regAlpha: Double,
                               val regLambda: Double,

                               val obj: ObjFunc,
                               val increEvals: Array[IncrementalEvalFunc],
                               val batchEvals: Array[BatchEvalFunc],
                               val callbacks: Array[CallbackFunc],

                               val catCols: Set[Int],
                               val rankCols: Set[Int],

                               val subSample: Double,
                               val colSampleByTree: Double,
                               val colSampleByLevel: Double,

                               val checkpointInterval: Int,
                               val storageLevel: StorageLevel,

                               val boostType: String,
                               val dropRate: Double,
                               val dropSkip: Double,
                               val minDrop: Int,
                               val maxDrop: Int,

                               val aggregationDepth: Int,
                               val initialModel: Option[GBMModel],

                               val maxBruteBins: Int,

                               val seed: Long) extends Serializable {

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




