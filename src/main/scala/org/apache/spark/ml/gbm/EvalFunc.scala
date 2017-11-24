package org.apache.spark.ml.gbm

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD

/**
  * trait for evaluation function to compute the metric
  */
private[ml] trait EvalFunc extends Logging with Serializable {

  def isLargerBetter: Boolean

  def name: String
}


/**
  * trait for batch evaluation function
  */
trait BatchEvalFunc extends EvalFunc {

  /**
    * compute the metric in a batch fashion, Must not persist/unpersist the input data
    *
    * @param data scored dataset containing (weight, label, score)
    * @return the metric value
    */
  def compute(data: RDD[(Double, Double, Double)]): Double
}

/**
  * trait for incremental (aggregation) evaluation function
  */
trait IncrementalEvalFunc extends EvalFunc {

  /** update the internal statistics aggregator */
  def update(weight: Double, label: Double, score: Double): Unit

  /** merge the internal aggregators */
  def merge(other: IncrementalEvalFunc): Unit

  /** compute the final metric value */
  def getValue: Double

  /** initial the internal statistics */
  def init: Unit
}


private[gbm] object IncrementalEvalFunc {

  def compute(data: RDD[(Double, Double, Double)],
              evals: Array[IncrementalEvalFunc],
              depth: Int): Array[Double] = {

    evals.foreach(_.init)

    data.treeAggregate[Array[IncrementalEvalFunc]](evals)(
      seqOp = {
        case (evals, (weight, label, score)) =>
          evals.foreach(eval => eval.update(weight, label, score))
          evals
      }, combOp = {
        case (evals1, evals2) =>
          require(evals1.length == evals2.length)
          evals1.zip(evals2).foreach {
            case (eval1, eval2) =>
              eval1.merge(eval2)
          }
          evals1
      }, depth = depth).map(_.getValue)
  }
}


/**
  * trait for simple element-wise evaluation function
  */
trait SimpleEvalFunc extends IncrementalEvalFunc {

  /** compute the metric on each instance */
  def compute(label: Double, score: Double): Double

  var count = 0.0
  var sum = 0.0

  override def update(weight: Double, label: Double, score: Double): Unit = {
    count += weight
    sum += weight * compute(label, score)
  }

  override def merge(other: IncrementalEvalFunc): Unit = {
    val o = other.asInstanceOf[SimpleEvalFunc]
    count += o.count
    sum += o.sum
  }

  override def getValue: Double = {
    sum / count
  }

  override def init: Unit = {
    count = 0.0
    sum = 0.0
  }
}


/**
  * Mean square error
  */
class MSEEval extends SimpleEvalFunc {
  override def compute(label: Double, score: Double): Double = {
    (label - score) * (label - score)
  }

  override def isLargerBetter: Boolean = false

  override def name = "MSE"
}


/**
  * Root mean square error
  */
class RMSEEval extends SimpleEvalFunc {
  override def compute(label: Double, score: Double): Double = {
    (label - score) * (label - score)
  }

  override def getValue: Double = {
    math.sqrt(sum / count)
  }

  override def isLargerBetter: Boolean = false

  override def name = "RMSE"
}


/**
  * Mean absolute error
  */
class MAEEval extends SimpleEvalFunc {
  override def compute(label: Double, score: Double): Double = {
    (label - score).abs
  }

  override def isLargerBetter: Boolean = false

  override def name = "MAE"
}


/**
  * Log-loss or Cross Entropy
  */
class LogLossEval extends SimpleEvalFunc {

  override def compute(label: Double, score: Double): Double = {
    require(label >= 0 && label <= 1)

    /** probability of positive class */
    val ppos = 1.0 / (1.0 + math.exp(-score))

    val pneg = 1.0 - ppos

    val eps = 1e-16

    if (ppos < eps) {
      -label * math.log(eps) - (1.0 - label) * math.log(1.0 - eps)
    } else if (pneg < eps) {
      -label * math.log(1.0 - eps) - (1.0 - label) * math.log(eps)
    } else {
      -label * math.log(ppos) - (1.0 - label) * math.log(pneg)
    }
  }

  override def isLargerBetter: Boolean = false

  override def name = "LogLoss"
}


/**
  * Classification error
  *
  * @param threshold threshold in binary classification prediction
  */
class ErrorEval(val threshold: Double) extends SimpleEvalFunc {
  require(threshold >= 0 && threshold <= 1)

  override def compute(label: Double, score: Double): Double = {
    require(label == 0 || label == 1)

    if (label == 0 && score <= threshold) {
      0.0
    } else if (label == 1 && score > threshold) {
      0.0
    } else {
      1.0
    }
  }

  override def isLargerBetter = false

  override def name = s"Error-$threshold"
}


/**
  * Area under curve
  *
  * @param numBins number of bins for approximate computation
  */
class AUCEval(val numBins: Int) extends IncrementalEvalFunc {
  require(numBins >= 16)

  def this() = this(1 << 16)

  val histPos: Array[Double] = Array.ofDim[Double](numBins)
  val histNeg: Array[Double] = Array.ofDim[Double](numBins)

  override def update(weight: Double, label: Double, score: Double): Unit = {
    require(label == 0 || label == 1)

    /** probability of positive class */
    val ppos = 1.0 / (1.0 + math.exp(-score))

    val index = math.min((ppos * numBins).floor.toInt, numBins - 1)

    if (label == 1) {
      histPos(index) += weight
    } else {
      histNeg(index) += weight
    }
  }

  override def merge(other: IncrementalEvalFunc): Unit = {
    val o = other.asInstanceOf[AUCEval]
    require(numBins == o.numBins)
    var i = 0
    while (i < numBins) {
      histPos(i) += o.histPos(i)
      histNeg(i) += o.histNeg(i)
      i += 1
    }
  }

  override def getValue: Double = {
    val numPos = histPos.sum
    val numNeg = histNeg.sum

    var i = 0
    while (i < numBins) {
      histPos(i) /= numPos
      histNeg(i) /= numNeg
      i += 1
    }

    var truePos = 0.0
    var falsePos = 0.0

    var auc = 0.0
    i = numBins - 1
    while (i >= 0) {
      /** trapezoidal area between point (falsePos, truePos) <-> point (falsePos + histPos(i), truePos + histPos(i)) on ROC */
      auc += histNeg(i) * (truePos + histPos(i) / 2)
      truePos += histPos(i)
      falsePos += histNeg(i)
      i -= 1
    }
    auc += (1.0 - falsePos) * (1.0 + truePos) / 2
    auc
  }

  override def init: Unit = {
    var i = 0
    while (i < numBins) {
      histPos(i) = 0.0
      histNeg(i) = 0.0
      i += 1
    }
  }

  override def isLargerBetter: Boolean = true

  override def name: String = "AUC"
}

