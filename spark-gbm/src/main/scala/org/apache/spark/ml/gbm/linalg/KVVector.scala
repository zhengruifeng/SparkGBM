package org.apache.spark.ml.gbm.linalg

import scala.reflect.ClassTag
import scala.{specialized => spec}

import org.apache.spark.ml.gbm.util.Utils

/**
  * Vector Type
  *
  * @tparam K Index Type
  * @tparam V Value Type
  */
private[gbm] trait KVVector[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V] extends Serializable {


  def size: Int


  /**
    * Create a new vector with negative values.
    */
  def negate()
            (implicit cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V]

  /**
    * Perform indexing.
    */
  def apply(index: Int)
           (implicit ink: Integral[K], nek: NumericExt[K],
            nuv: Numeric[V]): V

  /**
    * Create a new vector with given SORTED indices.
    * Note: for performance consideration, input indices will not be checked whether it obey the ordering.
    */
  def slice(sortedIndices: Array[Int])
           (implicit ck: ClassTag[K], ink: Integral[K],
            cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V]


  def toArray()
             (implicit ink: Integral[K],
              cv: ClassTag[V], nuv: Numeric[V]): Array[V]


  /**
    * Iterator of all values.
    */
  def iterator()
              (implicit ink: Integral[K],
               nuv: Numeric[V]): Iterator[(K, V)]


  /**
    * Iterator of all non-zero values.
    */
  def activeIterator()
                    (implicit ink: Integral[K],
                     nuv: Numeric[V]): Iterator[(K, V)]


  /**
    * Reverse iterator of all non-zero values.
    */
  def reverseActiveIterator()
                           (implicit ink: Integral[K],
                            nuv: Numeric[V]): Iterator[(K, V)]

  /**
    * Number of non-zero values.
    */
  def nnz()
         (implicit nuv: Numeric[V]): Int


  def toDense()
             (implicit ink: Integral[K],
              cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V]


  def toSparse()
              (implicit ck: ClassTag[K], ink: Integral[K],
               cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V]


  def isDense: Boolean


  def isSparse: Boolean = !isDense


  def isEmpty: Boolean = size == 0


  def nonEmpty: Boolean = !isEmpty


  /**
    * Compress a vector to alleviate memory footprint.
    */
  def compress()
              (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
               cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): KVVector[K, V] = {
    if (nev.size * size + 8 <= (nek.size + nev.size) * nnz + 20) {
      toDense
    } else {
      toSparse
    }
  }


  /**
    * Update the value of given index by plus.
    * Note: This may change the vector size if needed.
    */
  def plus(index: K, value: V)
          (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
           cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V]


  /**
    * Update the value of given index by minus.
    * Note: This may change the vector size if needed.
    */
  def minus(index: K, value: V)
           (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
            cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] =
    plus(index, nuv.negate(value))


  /**
    * Merge with another vector by plus.
    * Note: The size of input vector may not equals to this one.
    */
  def plus(other: KVVector[K, V])
          (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
           cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): KVVector[K, V] = {
    import nuv._

    if (this.isEmpty) {
      other

    } else if (other.isEmpty) {
      this

    } else {

      (this, other) match {
        case (dv1: DenseKVVector[K, V], dv2: DenseKVVector[K, V]) =>
          if (dv1.size >= dv2.size) {
            dv2.activeIterator.foreach { case (k, v) => dv1.values(ink.toInt(k)) += v }
            dv1
          } else {
            dv1.activeIterator.foreach { case (k, v) => dv2.values(ink.toInt(k)) += v }
            dv2
          }


        case (dv: DenseKVVector[K, V], sv: SparseKVVector[K, V]) =>
          // update vector size if needed
          var vec = dv.plus(ink.fromInt(sv.size - 1), zero)
          sv.activeIterator.foreach { case (k, v) => vec = vec.plus(k, v) }
          vec


        case (sv: SparseKVVector[K, V], dv: DenseKVVector[K, V]) =>
          // update vector size if needed
          var vec = dv.plus(ink.fromInt(sv.size - 1), zero)
          sv.activeIterator.foreach { case (k, v) => vec = vec.plus(k, v) }
          vec


        case (sv1: SparseKVVector[K, V], sv2: SparseKVVector[K, V]) =>
          val (indices, values) =
            Utils.outerJoinSortedIters(sv1.activeIterator, sv2.activeIterator, false)
              .map { case (k, v1, v2) => (k, v1.getOrElse(zero) + v2.getOrElse(zero)) }
              .filter(_._2 != zero).toArray.unzip
          val newSize = math.max(size, other.size)
          KVVector.sparse[K, V](newSize, indices, values)
      }
    }
  }


  /**
    * Merge with another vector by minus.
    * Note: The size of input vector may not equals to this one.
    */
  def minus(other: KVVector[K, V])
           (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
            cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): KVVector[K, V] =
    plus(other.negate)
}


private[gbm] object KVVector {

  def empty[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V]()
                                                                                      (implicit cv: ClassTag[V], nev: NumericExt[V]): KVVector[K, V] = {
    dense[K, V](nev.emptyArray)
  }

  def dense[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V](values: Array[V]): KVVector[K, V] = {
    new DenseKVVector[K, V](values)
  }

  def sparse[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V](size: Int,
                                                                                        indices: Array[K],
                                                                                        values: Array[V]): KVVector[K, V] = {
    new SparseKVVector[K, V](size, indices, values)
  }
}


private[gbm] class DenseKVVector[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V](val values: Array[V]) extends KVVector[K, V] {

  override def size: Int = values.length


  override def negate()
                     (implicit cv: ClassTag[V],
                      nuv: Numeric[V]): KVVector[K, V] =
    KVVector.dense[K, V](values.map(nuv.negate))


  override def apply(index: Int)
                    (implicit ink: Integral[K], nek: NumericExt[K],
                     nuv: Numeric[V]) = values(index)


  override def plus(index: K, value: V)
                   (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
                    cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] = {
    require(ink.gteq(index, ink.zero))

    val newSize = math.max(ink.toInt(index) + 1, size)

    import nuv._
    if (value == zero) {
      if (size == newSize) {
        this
      } else {
        KVVector.dense[K, V](values ++ Array.fill(newSize - size)(zero))
      }

    } else {
      val i = ink.toInt(index)
      if (i < size) {
        values(i) += value
        this

      } else {
        val newValues = values ++ Array.fill(i - size + 1)(zero)
        newValues(i) = value
        KVVector.dense[K, V](newValues)
      }
    }
  }


  override def slice(sortedIndices: Array[Int])
                    (implicit ck: ClassTag[K], ink: Integral[K],
                     cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] =
    KVVector.dense(sortedIndices.map(values))


  override def toArray()
                      (implicit ink: Integral[K],
                       cv: ClassTag[V], nuv: Numeric[V]): Array[V] = values


  override def iterator()
                       (implicit ink: Integral[K],
                        nuv: Numeric[V]): Iterator[(K, V)] =
    values.iterator
      .zipWithIndex.map { case (v, i) => (ink.fromInt(i), v) }


  override def activeIterator()
                             (implicit ink: Integral[K],
                              nuv: Numeric[V]): Iterator[(K, V)] =
    iterator.filter(t => t._2 != nuv.zero)


  override def reverseActiveIterator()
                                    (implicit ink: Integral[K],
                                     nuv: Numeric[V]): Iterator[(K, V)] =
    values.reverseIterator
      .zipWithIndex
      .map { case (v, i) => (ink.fromInt(size - 1 - i), v) }
      .filter(t => t._2 != nuv.zero)


  override def nnz()
                  (implicit nuv: Numeric[V]): Int =
    values.count(_ != nuv.zero)


  override def toDense()
                      (implicit ink: Integral[K],
                       cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] = this


  override def toSparse()
                       (implicit ck: ClassTag[K], ink: Integral[K],
                        cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] = {
    val (newIndices, newValues) = activeIterator.toArray.unzip
    KVVector.sparse[K, V](size, newIndices, newValues)
  }


  override def isDense: Boolean = true


  override def toString: String =
    s"DenseKVVector[${values.mkString("(", ",", ")")}]"
}


class SparseKVVector[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V](val size: Int,
                                                                                                val indices: Array[K],
                                                                                                val values: Array[V]) extends KVVector[K, V] {

  require(indices.length == values.length)
  require(size >= 0)


  override def negate()
                     (implicit cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] =
    KVVector.sparse[K, V](size, indices, values.map(nuv.negate))


  override def apply(index: Int)
                    (implicit ink: Integral[K], nek: NumericExt[K],
                     nuv: Numeric[V]): V = {
    require(0 <= index && index < size)
    val j = nek.search(indices, ink.fromInt(index))
    if (j >= 0) {
      values(j)
    } else {
      nuv.zero
    }
  }


  override def plus(index: K, value: V)
                   (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
                    cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] = {
    require(ink.gteq(index, ink.zero))

    import nuv._

    val newSize = math.max(ink.toInt(index) + 1, size)

    if (value == zero) {
      if (size == newSize) {
        this
      } else {
        KVVector.sparse[K, V](newSize, indices, values)
      }

    } else {
      val j = nek.search(indices, index)
      if (j >= 0) {
        values(j) += value
        this

      } else {
        val left = -j - 1
        val right = indices.length - left
        val newIndices = indices.take(left) ++ Array(index) ++ indices.takeRight(right)
        val newValues = values.take(left) ++ Array(value) ++ values.takeRight(right)
        KVVector.sparse[K, V](newSize, newIndices, newValues)
      }
    }
  }


  override def slice(sortedIndices: Array[Int])
                    (implicit ck: ClassTag[K], ink: Integral[K],
                     cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] = {
    val iter = sortedIndices.iterator.map(i => (ink.fromInt(i), true))

    val (newIndices, newValues) =
      Utils.innerJoinSortedIters(activeIterator, iter, false)
        .map(t => (t._1, t._2)).toArray.unzip

    KVVector.sparse[K, V](sortedIndices.length, newIndices, newValues)
  }


  override def toArray()
                      (implicit ink: Integral[K],
                       cv: ClassTag[V], nuv: Numeric[V]): Array[V] =
    iterator.map(_._2).toArray


  override def iterator()
                       (implicit ink: Integral[K],
                        nuv: Numeric[V]): Iterator[(K, V)] = {
    val size_ = size

    new Iterator[(K, V)]() {
      private var i = 0
      private var j = 0

      override def hasNext: Boolean = i < size_

      override def next: (K, V) = {
        val v = if (j == indices.length) {
          nuv.zero
        } else {
          val k = ink.toInt(indices(j))
          if (i == k) {
            j += 1
            values(j - 1)
          } else {
            nuv.zero
          }
        }
        i += 1
        (ink.fromInt(i - 1), v)
      }
    }
  }


  override def activeIterator()
                             (implicit ink: Integral[K],
                              nuv: Numeric[V]): Iterator[(K, V)] =
    indices.iterator.zip(values.iterator)
      .filter(t => t._2 != nuv.zero)


  override def reverseActiveIterator()
                                    (implicit ink: Integral[K],
                                     nuv: Numeric[V]): Iterator[(K, V)] =
    indices.reverseIterator.zip(values.reverseIterator)
      .filter(t => t._2 != nuv.zero)


  override def nnz()
                  (implicit nuv: Numeric[V]): Int =
    values.count(_ != nuv.zero)


  override def toDense()
                      (implicit ink: Integral[K],
                       cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] =
    KVVector.dense[K, V](toArray)


  override def toSparse()
                       (implicit ck: ClassTag[K], ink: Integral[K],
                        cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] = {
    if (indices.length == nnz) {
      this

    } else {
      val (newIndices, newValues) = activeIterator.toArray.unzip
      KVVector.sparse[K, V](size, newIndices, newValues)
    }
  }


  override def isDense: Boolean = false


  override def toString: String =
    s"SparseKVVector[$size, ${indices.zip(values).map { case (k, v) => s"$k->$v" }.mkString("(", ",", ")")}]"
}