package org.apache.spark.ml.gbm.func

/**
  * objective function, compute the gradient and hessian of each instance
  */
trait ObjFunc extends Serializable {

  /**
    * compute the gradient and hessian with respect to current score
    *
    * @param label label
    * @param score current score
    * @return gradient and hessian
    */
  def compute(label: Array[Double],
              score: Array[Double]): (Array[Double], Array[Double])

  def transform(raw: Array[Double]): Array[Double] = raw

  def inverseTransform(score: Array[Double]): Array[Double] = score

  def name: String
}


trait ScalarObjFunc extends ObjFunc {

  def compute(label: Double, score: Double): (Double, Double)

  final override def compute(label: Array[Double],
                             score: Array[Double]): (Array[Double], Array[Double]) = {
    require(label.length == 1 && score.length == 1)
    val (g, h) = compute(label.head, score.head)
    (Array(g), Array(h))
  }

  def transform(raw: Double): Double = raw

  override def transform(raw: Array[Double]): Array[Double] = {
    require(raw.length == 1)
    Array(transform(raw.head))
  }

  def inverseTransform(score: Double): Double = score

  override def inverseTransform(score: Array[Double]): Array[Double] = {
    require(score.length == 1)
    Array(inverseTransform(score.head))
  }
}


/**
  * Square loss
  */
class SquareObj extends ScalarObjFunc {

  override def compute(label: Double,
                       score: Double): (Double, Double) = (score - label, 1.0)

  def name = "Square"
}


/**
  * Log-loss or Cross Entropy
  */
class LogisticObj extends ScalarObjFunc {

  override def compute(label: Double,
                       score: Double): (Double, Double) = {
    require(label >= 0 && label <= 1)
    val grad = score - label
    val hess = math.max(score * (1 - score), 1e-16)
    (grad, hess)
  }


  /** Logistic transform, probability of class 1 */
  override def transform(raw: Double): Double = 1.0 / (1.0 + math.exp(-raw))

  /** Logit transformation */
  override def inverseTransform(score: Double): Double = -math.log(1.0 / score - 1.0)

  override def name = "Logistic"
}


class SoftmaxObj(val numClasses: Int) extends ObjFunc {
  require(numClasses > 1)

  /**
    * compute the gradient and hessian with respect to current score
    *
    * @param label label
    * @param score current score
    * @return gradient and hessian
    */
  override def compute(label: Array[Double],
                       score: Array[Double]): (Array[Double], Array[Double]) = {
    require(label.length == numClasses)
    require(score.length == numClasses)

    val grad = Array.ofDim[Double](numClasses)
    val hess = Array.ofDim[Double](numClasses)

    var i = 0
    while (i < grad.length) {
      grad(i) = score(i) - label(i)
      hess(i) = math.max(2.0 * score(i) * (1.0 - score(i)), 1e-16F)
      i += 1
    }

    (grad, hess)
  }

  override def transform(raw: Array[Double]): Array[Double] = {
    require(raw.length == numClasses)

    val score = Array.ofDim[Double](numClasses)

    var expSum = 0.0
    var i = 0
    while (i < numClasses) {
      score(i) = math.exp(raw(i))
      expSum += score(i)
      i += 1
    }

    i = 0
    while (i < numClasses) {
      score(i) /= expSum
      i += 1
    }

    score
  }

  override def inverseTransform(score: Array[Double]): Array[Double] = {
    require(score.length == numClasses)

    val raw = Array.ofDim[Double](numClasses)

    var sum = 0.0
    var i = 0
    while (i < numClasses) {
      raw(i) = score(i)
      sum += raw(i)
      i += 1
    }

    val base = math.log(numClasses / sum)

    i = 0
    while (i < numClasses) {
      raw(i) = base + math.log(score(i))
      i += 1
    }

    raw
  }

  override def name = "Softmax"
}


