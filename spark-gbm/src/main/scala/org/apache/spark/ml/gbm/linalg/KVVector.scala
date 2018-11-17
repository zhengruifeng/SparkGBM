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
trait KVVector[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V] extends Serializable {


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
           cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): KVVector[K, V]


  /**
    * Merge with another vector by minus.
    * Note: The size of input vector may not equals to this one.
    */
  def minus(other: KVVector[K, V])
           (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
            cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): KVVector[K, V] =
    plus(other.negate)


  def copy(): KVVector[K, V]
}


object KVVector {

  def empty[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V]()
                                                                                      (implicit cv: ClassTag[V], nev: NumericExt[V]): KVVector[K, V] =
    dense[K, V](nev.emptyArray)


  def dense[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V](values: Array[V]): KVVector[K, V] =
    new DenseKVVector[K, V](values)


  def sparse[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V](size: Int,
                                                                                        indices: Array[K],
                                                                                        values: Array[V]): KVVector[K, V] =
    new SparseKVVector[K, V](size, indices, values)

}


class DenseKVVector[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int, Long, Float, Double) V](val values: Array[V]) extends KVVector[K, V] {

  override def size: Int = values.length


  override def negate()
                     (implicit cv: ClassTag[V],
                      nuv: Numeric[V]): KVVector[K, V] =
    KVVector.dense[K, V](values.map(nuv.negate))


  override def apply(index: Int)
                    (implicit ink: Integral[K], nek: NumericExt[K],
                     nuv: Numeric[V]) =
    values(index)


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
        val newValues = Array.ofDim[V](newSize)
        System.arraycopy(values, 0, newValues, 0, values.length)
        KVVector.dense[K, V](newValues)
      }

    } else {
      val i = ink.toInt(index)
      if (size == newSize) {
        values(i) += value
        this

      } else {
        val newValues = Array.ofDim[V](newSize)
        System.arraycopy(values, 0, newValues, 0, values.length)
        newValues(i) = value
        KVVector.dense[K, V](newValues)
      }
    }
  }


  override def plus(other: KVVector[K, V])
                   (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
                    cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): KVVector[K, V] = {
    import nuv._

    if (other.isEmpty) {
      return this
    }

    if (this.isEmpty) {
      return other
    }

    other match {
      case dv: DenseKVVector[K, V] =>
        if (size >= dv.size) {
          val iter = dv.activeIterator
          while (iter.hasNext) {
            val (k, v) = iter.next()
            values(ink.toInt(k)) += v
          }
          this

        } else {
          val iter = activeIterator
          while (iter.hasNext) {
            val (k, v) = iter.next()
            dv.values(ink.toInt(k)) += v
          }
          dv
        }


      case sv: SparseKVVector[K, V] =>
        if (size >= sv.size) {
          val iter = sv.activeIterator
          while (iter.hasNext) {
            val (k, v) = iter.next()
            values(ink.toInt(k)) += v
          }
          this

        } else {
          val newValues = Array.ofDim[V](sv.size)
          System.arraycopy(values, 0, newValues, 0, values.length)
          val iter = sv.activeIterator
          while (iter.hasNext) {
            val (k, v) = iter.next()
            newValues(ink.toInt(k)) += v
          }
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
                       cv: ClassTag[V], nuv: Numeric[V]): Array[V] =
    values


  override def iterator()
                       (implicit ink: Integral[K],
                        nuv: Numeric[V]): Iterator[(K, V)] =
    values.iterator.zipWithIndex
      .map { case (v, i) => (ink.fromInt(i), v) }


  override def activeIterator()
                             (implicit ink: Integral[K],
                              nuv: Numeric[V]): Iterator[(K, V)] =
    iterator.filter(t => t._2 != nuv.zero)


  override def reverseActiveIterator()
                                    (implicit ink: Integral[K],
                                     nuv: Numeric[V]): Iterator[(K, V)] =
    values.reverseIterator.zipWithIndex
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


  override def copy(): KVVector[K, V] =
    KVVector.dense[K, V](values.clone())


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
        val k = -j - 1

        val newIndices = Array.ofDim[K](indices.length + 1)
        val newValues = Array.ofDim[V](values.length + 1)

        if (k != 0) {
          System.arraycopy(indices, 0, newIndices, 0, k)
          System.arraycopy(values, 0, newValues, 0, k)
        }

        if (k != indices.length) {
          System.arraycopy(indices, k, newIndices, k + 1, indices.length - k)
          System.arraycopy(values, k, newValues, k + 1, values.length - k)
        }

        newIndices(k) = index
        newValues(k) = value

        KVVector.sparse[K, V](newSize, newIndices, newValues)
      }
    }
  }


  override def plus(other: KVVector[K, V])
                   (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
                    cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): KVVector[K, V] = {
    import nuv._

    if (other.isEmpty) {
      return this
    }

    if (this.isEmpty) {
      return other
    }

    other match {
      case dv: DenseKVVector[K, V] =>
        if (size <= dv.size) {
          val iter = activeIterator
          while (iter.hasNext) {
            val (k, v) = iter.next()
            dv.values(ink.toInt(k)) += v
          }
          dv

        } else {
          val newValues = Array.ofDim[V](size)
          System.arraycopy(dv.values, 0, newValues, 0, dv.values.length)
          val iter = activeIterator
          while (iter.hasNext) {
            val (k, v) = iter.next()
            newValues(ink.toInt(k)) += v
          }
          KVVector.dense[K, V](newValues)
        }


      case sv: SparseKVVector[K, V] =>
        val (indices, values) =
          Utils.outerJoinSortedIters(activeIterator, sv.activeIterator, false)
            .map { case (k, v1, v2) => (k, v1.getOrElse(zero) + v2.getOrElse(zero)) }
            .filter(_._2 != zero).toArray.unzip

        val newSize = math.max(size, other.size)
        KVVector.sparse[K, V](newSize, indices, values)
    }
  }


  override def slice(sortedIndices: Array[Int])
                    (implicit ck: ClassTag[K], ink: Integral[K],
                     cv: ClassTag[V], nuv: Numeric[V]): KVVector[K, V] = {
    val iter = sortedIndices.iterator.map(ink.fromInt).zipWithIndex

    val (newIndices, newValues) =
      Utils.innerJoinSortedIters(activeIterator, iter)
        .map { case (_, value, newIndex) => (ink.fromInt(newIndex), value) }
        .toArray.unzip

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
    if (!values.contains(nuv.zero)) {
      this

    } else {
      val (newIndices, newValues) = activeIterator.toArray.unzip
      KVVector.sparse[K, V](size, newIndices, newValues)
    }
  }


  override def isDense: Boolean = false

  override def copy(): KVVector[K, V] =
    KVVector.sparse[K, V](size, indices.clone(), values.clone())

  override def toString: String = {
    s"SparseKVVector[$size," +
      s" ${indices.iterator.zip(values.iterator).map { case (k, v) => s"$k->$v" }.mkString("(", ",", ")")}]"
  }
}