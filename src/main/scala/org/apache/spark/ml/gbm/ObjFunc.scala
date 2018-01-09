package org.apache.spark.ml.gbm

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
  def compute(label: Double, score: Double): (Double, Double)

  def name: String
}


/**
  * Square loss
  */
class SquareObj extends ObjFunc {
  override def compute(label: Double, score: Double): (Double, Double) = {
    (score - label, 1.0)
  }

  def name = "Square"
}


/**
  * Pseudo-Huber loss
  *
  * @param delta the param of Huber loss
  */
class HuberObj(val delta: Double) extends ObjFunc {
  require(delta > 1)

  def this() = this(1.35)

  override def compute(label: Double, score: Double): (Double, Double) = {
    val a = score - label

    val r = a / delta

    val e = 1 + r * r

    val s = math.sqrt(e)

    val grad = a / s

    val hess = math.max(1 / (e * s), 1e-16)

    (grad, hess)
  }

  override def name = s"Huber-$delta"
}


/**
  * Log-loss or Cross Entropy
  */
class LogisticObj extends ObjFunc {
  override def compute(label: Double, score: Double): (Double, Double) = {
    require(label >= 0 && label <= 1)

    /** probability of class 1 */
    val ppos = 1.0 / (1.0 + math.exp(-score))

    val grad = ppos - label

    val hess = math.max(ppos * (1 - ppos), 1e-16)

    (grad, hess)
  }

  override def name = "Logistic"
}


