package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.{ClassTag, classTag}
import scala.{specialized => spec}

private[gbm] trait GBMVector[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V] extends Serializable {

  def len: Int

  def apply(index: Int): V

  def slice(sorted: Array[Int]): GBMVector[K, V]

  def toArray: Array[V]

  def totalIter: Iterator[(K, V)]

  def activeIter: Iterator[(K, V)]

  def nnz: Int = activeIter.size

  def toDense: GBMVector[K, V]

  def toSparse: GBMVector[K, V]

  def compress()(implicit kk: ClassTag[K], kv: ClassTag[V]): GBMVector[K, V] = {
    val kSize = GBMVector.getTypeSize[K]
    val vSize = GBMVector.getTypeSize[V]
    if (vSize * len + 8 <= (kSize + vSize) * nnz + 20) {
      toDense
    } else {
      toSparse
    }
  }
}

private[gbm] object GBMVector {

  def dense[@spec(Byte, Short, Int) K: Integral : ClassTag, @spec(Byte, Short, Int, Long, Float, Double) V: Numeric : ClassTag](values: Array[V]): GBMVector[K, V] = {
    new DenseGBMVector[K, V](values)
  }

  def sparse[@spec(Byte, Short, Int) K: Integral : ClassTag, @spec(Byte, Short, Int, Long, Float, Double) V: Numeric : ClassTag](size: Int,
                                                                                                                                 indices: Array[K],
                                                                                                                                 values: Array[V]): GBMVector[K, V] = {
    new SparseGBMVector[K, V](size, indices, values)
  }

  def getTypeSize[T: ClassTag]: Int = {
    classTag[T] match {
      case ClassTag.Byte => 1
      case ClassTag.Short => 2
      case ClassTag.Int => 4
      case ClassTag.Long => 8
      case ClassTag.Float => 4
      case ClassTag.Double => 8
    }
  }
}


private class DenseGBMVector[@spec(Byte, Short, Int) K: Integral : ClassTag, @spec(Byte, Short, Int, Long, Float, Double) V: Numeric : ClassTag](val values: Array[V]) extends GBMVector[K, V] {

  override def len: Int = values.length

  override def apply(index: Int) = values(index)

  override def slice(sorted: Array[Int]): GBMVector[K, V] =
    GBMVector.dense(sorted.map(values))

  override def toArray: Array[V] =
    totalIter.map(_._2).toArray

  override def totalIter: Iterator[(K, V)] = {
    val intK = implicitly[Integral[K]]
    values.iterator.zipWithIndex.map { case (v, i) => (intK.fromInt(i), v) }
  }

  override def activeIter: Iterator[(K, V)] = {
    val numV = implicitly[Numeric[V]]
    totalIter.filter(t => t._2 != numV.zero)
  }

  override def toDense: GBMVector[K, V] = this

  override def toSparse: GBMVector[K, V] = {
    val indexBuilder = mutable.ArrayBuilder.make[K]
    val valueBuilder = mutable.ArrayBuilder.make[V]
    activeIter.foreach { case (i, v) =>
      indexBuilder += i
      valueBuilder += v
    }
    GBMVector.sparse[K, V](len, indexBuilder.result(), valueBuilder.result())
  }
}


private class SparseGBMVector[@spec(Byte, Short, Int) K: Integral : ClassTag, @spec(Byte, Short, Int, Long, Float, Double) V: Numeric : ClassTag](val len: Int,
                                                                                                                                                  val indices: Array[K],
                                                                                                                                                  val values: Array[V]) extends GBMVector[K, V] {

  require(indices.length == values.length)
  require(len >= 0)

  private def binarySearch: (Array[K], K) => Int =
    Utils.makeBinarySearch[K]

  override def apply(index: Int): V = {
    val intK = implicitly[Integral[K]]
    val j = binarySearch(indices, intK.fromInt(index))
    if (j >= 0) {
      values(j)
    } else {
      val numV = implicitly[Numeric[V]]
      numV.zero
    }
  }

  override def slice(sorted: Array[Int]): GBMVector[K, V] = {
    val intK = implicitly[Integral[K]]
    val indexBuilder = mutable.ArrayBuilder.make[K]
    val valueBuilder = mutable.ArrayBuilder.make[V]

    var i = 0
    var j = 0
    while (i < sorted.length && j < indices.length) {
      val k = intK.toInt(indices(j))
      if (sorted(i) == k) {
        indexBuilder += intK.fromInt(i)
        valueBuilder += values(j)
        i += 1
        j += 1
      } else if (sorted(i) > k) {
        j += 1
      } else {
        i += 1
      }
    }

    GBMVector.sparse[K, V](sorted.length, indexBuilder.result(), valueBuilder.result())
  }

  override def toArray: Array[V] =
    totalIter.map(_._2).toArray

  override def totalIter: Iterator[(K, V)] = new Iterator[(K, V)]() {
    private val intK = implicitly[Integral[K]]
    private val numV = implicitly[Numeric[V]]

    private var i = 0
    private var j = 0

    override def hasNext: Boolean = i < len

    override def next: (K, V) = {
      val v = if (j == indices.length) {
        numV.zero
      } else {
        val k = intK.toInt(indices(j))
        if (i == k) {
          j += 1
          values(j - 1)
        } else {
          numV.zero
        }
      }
      i += 1
      (intK.fromInt(i - 1), v)
    }
  }

  override def activeIter: Iterator[(K, V)] = {
    val numV = implicitly[Numeric[V]]
    indices.iterator
      .zip(values.iterator)
      .filter(t => t._2 != numV.zero)
  }

  override def toDense: GBMVector[K, V] =
    GBMVector.dense[K, V](toArray)

  override def toSparse: GBMVector[K, V] = this
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

