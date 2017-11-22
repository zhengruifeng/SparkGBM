package org.apache.spark.ml.gbm

private trait FromDouble[H] extends Serializable {
  def fromDouble(value: Double): H
}

private object DoubleFromDouble extends FromDouble[Double] {
  override def fromDouble(value: Double): Double = value
}

private object FloatFromDouble extends FromDouble[Float] {
  override def fromDouble(value: Double): Float = value.toFloat
}

private object DecimalFromDouble extends FromDouble[BigDecimal] {
  override def fromDouble(value: Double): BigDecimal = BigDecimal(value)
}

private[gbm] object FromDouble {

  implicit final val doubleFromDouble: FromDouble[Double] = DoubleFromDouble

  implicit final val floatFromDouble: FromDouble[Float] = FloatFromDouble

  implicit final val decimalFromDouble: FromDouble[BigDecimal] = DecimalFromDouble
}
