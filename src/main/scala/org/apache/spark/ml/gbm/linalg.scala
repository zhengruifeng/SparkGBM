package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.{ClassTag, classTag}
import scala.{specialized => spec}

import org.apache.spark.SparkContext


private[gbm] trait BinVector[@spec(Byte, Short, Int) V] extends Serializable {

  def len: Int

  def apply(index: Int): V

  def slice(sorted: Array[Int]): BinVector[V]

  def toArray: Array[V]

  def totalIter: Iterator[(Int, V)]

  def activeIter: Iterator[(Int, V)]

  def computeDenseSize: Int

  def computeSparseSize: Int

  def toDense: BinVector[V]

  def toSparse: BinVector[V]

  def compress: BinVector[V] = {
    if (computeDenseSize <= computeSparseSize) {
      toDense
    } else {
      toSparse
    }
  }
}

private[gbm] object BinVector {

  def dense[@spec(Byte, Short, Int) V: Integral : ClassTag](values: Array[V]): BinVector[V] = {
    new DenseBinVector[V](values)
  }

  def sparse[@spec(Byte, Short, Int) V: Integral : ClassTag](size: Int,
                                                             indices: Array[Int],
                                                             values: Array[V]): BinVector[V] = {
    val lastIndex = indices.lastOption.getOrElse(-1)
    if (lastIndex <= Byte.MaxValue) {
      new SparseBinVector[Byte, V](size, indices.map(_.toByte), values)
    } else if (lastIndex <= Short.MaxValue) {
      new SparseBinVector[Short, V](size, indices.map(_.toShort), values)
    } else {
      new SparseBinVector[Int, V](size, indices, values)
    }
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

  private[this] var kryoRegistered: Boolean = false

  def registerKryoClasses(sc: SparkContext): Unit = {
    if (!kryoRegistered) {
      sc.getConf.registerKryoClasses(
        Array(classOf[BinVector[Object]],
          classOf[(Object, Object, BinVector[Object])],
          classOf[DenseBinVector[Object]],
          classOf[SparseBinVector[Object, Object]])
      )
      kryoRegistered = true
    }
  }
}


private class DenseBinVector[@spec(Byte, Short, Int) V: Integral : ClassTag](val values: Array[V]) extends BinVector[V] {

  override def len = values.length

  override def apply(index: Int) = values(index)

  override def slice(sorted: Array[Int]) =
    BinVector.dense(sorted.map(values))

  override def toArray: Array[V] =
    totalIter.map(_._2).toArray

  override def totalIter =
    Iterator.range(0, values.length).map(i => (i, values(i)))

  override def activeIter = {
    val intV = implicitly[Integral[V]]
    totalIter.filter(t => !intV.equiv(t._2, intV.zero))
  }

  override def toDense: BinVector[V] = this

  override def toSparse: BinVector[V] = {
    val indexBuff = mutable.ArrayBuffer.empty[Int]
    val valueBuff = mutable.ArrayBuffer.empty[V]
    activeIter.foreach { case (i, v) =>
      indexBuff.append(i)
      valueBuff.append(v)
    }
    BinVector.sparse[V](len, indexBuff.toArray, valueBuff.toArray)
  }

  override def computeDenseSize: Int = {
    BinVector.getTypeSize[V] * len + 8
  }

  override def computeSparseSize: Int = {
    var nnz = 0
    var lastIndex = -1
    activeIter.foreach { case (i, v) =>
      nnz += 1
      lastIndex = i
    }

    val kSize = if (lastIndex <= Byte.MaxValue) {
      1
    } else if (lastIndex <= Short.MaxValue) {
      2
    } else {
      4
    }

    val vSize = BinVector.getTypeSize[V]

    (kSize + vSize) * nnz + 20
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
    val indexBuff = mutable.ArrayBuffer.empty[Int]
    val valueBuff = mutable.ArrayBuffer.empty[V]
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

  override def toArray: Array[V] =
    totalIter.map(_._2).toArray

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

  override def toDense: BinVector[V] = {
    BinVector.dense[V](toArray)
  }

  override def toSparse: BinVector[V] = this

  override def computeDenseSize: Int = {
    BinVector.getTypeSize[V] * len + 8
  }

  override def computeSparseSize: Int = {
    val kSize = BinVector.getTypeSize[K]
    val vSize = BinVector.getTypeSize[V]
    (kSize + vSize) * indices.length + 20
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

