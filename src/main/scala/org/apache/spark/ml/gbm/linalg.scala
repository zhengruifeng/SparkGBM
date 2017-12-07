package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.{specialized => spec}


private[gbm] trait BinVector[@spec(Byte, Short, Int) V] extends Serializable {

  def len: Int

  def apply(index: Int): V

  def slice(sorted: Array[Int]): BinVector[V]

  def totalIter: Iterator[(Int, V)]

  def activeIter: Iterator[(Int, V)]
}

private[gbm] object BinVector {
  def dense[V: Integral : ClassTag](values: Array[V]): BinVector[V] = {
    new DenseBinVector[V](values)
  }

  def sparse[V: Integral : ClassTag](size: Int,
                                     indices: Array[Int],
                                     values: Array[V]): BinVector[V] = {
    val lastIndex = indices.lastOption.getOrElse(0)
    if (lastIndex <= Byte.MaxValue) {
      new SparseBinVector[Byte, V](size, indices.map(_.toByte), values)
    } else if (lastIndex <= Short.MaxValue) {
      new SparseBinVector[Short, V](size, indices.map(_.toShort), values)
    } else {
      new SparseBinVector[Int, V](size, indices, values)
    }
  }
}


private class DenseBinVector[@spec(Byte, Short, Int) V: Integral : ClassTag](val values: Array[V]) extends BinVector[V] {

  override def len = values.length

  override def apply(index: Int) = values(index)

  override def slice(sorted: Array[Int]) =
    BinVector.dense(sorted.map(values))

  override def totalIter =
    Iterator.range(0, values.length).map(i => (i, values(i)))

  override def activeIter = {
    val intV = implicitly[Integral[V]]
    totalIter.filter(t => !intV.equiv(t._2, intV.zero))
  }
}


private class SparseBinVector[@spec(Byte, Short, Int) K: Integral : ClassTag, @spec(Byte, Short, Int) V: Integral : ClassTag](val len: Int,
                                                                                                                              val indices: Array[K],
                                                                                                                              val values: Array[V]) extends BinVector[V] {

  require(indices.length == values.length)
  require(len >= 0)

  private def binarySearch = Utils.makeBinarySearch[K]

  override def apply(index: Int) = {
    val intK = implicitly[Integral[K]]
    val j = binarySearch(indices, intK.fromInt(index))
    if (j >= 0) {
      values(j)
    } else {
      val intV = implicitly[Integral[V]]
      intV.zero
    }
  }

  override def slice(sorted: Array[Int]) = {
    val intK = implicitly[Integral[K]]
    val indexBuff = mutable.ArrayBuffer[Int]()
    val valueBuff = mutable.ArrayBuffer[V]()
    var i = 0
    var j = 0
    while (i < sorted.length && j < indices.length) {
      val k = intK.toInt(indices(j))
      if (sorted(i) == k) {
        indexBuff.append(i)
        valueBuff.append(values(j))
        i += 1
        j += 1
      } else if (sorted(i) > k) {
        j += 1
      } else {
        i += 1
      }
    }

    BinVector.sparse[V](sorted.length, indexBuff.toArray, valueBuff.toArray)
  }

  override def totalIter = new Iterator[(Int, V)]() {
    private val intK = implicitly[Integral[K]]
    private val intV = implicitly[Integral[V]]

    private var i = 0
    private var j = 0

    override def hasNext = i < len

    override def next = {
      val v = if (j == indices.length) {
        intV.zero
      } else {
        val k = intK.toInt(indices(j))
        if (i == k) {
          j += 1
          values(j - 1)
        } else {
          intV.zero
        }
      }
      i += 1
      (i - 1, v)
    }
  }

  override def activeIter = {
    val intK = implicitly[Integral[K]]
    val intV = implicitly[Integral[V]]
    Iterator.range(0, indices.length)
      .map(i => (intK.toInt(indices(i)), values(i)))
      .filter(t => !intV.equiv(t._2, intV.zero))
  }
}


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

