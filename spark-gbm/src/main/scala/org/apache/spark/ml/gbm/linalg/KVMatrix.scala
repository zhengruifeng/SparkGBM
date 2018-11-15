package org.apache.spark.ml.gbm.linalg

import java.{util => ju}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.{specialized => spec}


/**
  * Compress a block of vectors in a compact fashion.
  * Note: all vectors must be of the same length.
  *
  */
class KVMatrix[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int) V](val indices: Array[K],
                                                                     val values: Array[V],
                                                                     val status: Array[Int]) extends Serializable {

  require(status.length >= 3)

  require(status(0) >= 0 && status(0) <= 4)
  require(status(1) >= 0)
  require(status(2) >= -1)

  if (status(0) == 0) {
    require(indices.isEmpty)
    require(values.isEmpty)
    require(status.length == 3)

  } else if (status(0) == 1) {
    require(status.length == status(1) + 3)

  } else if (status(0) == 2) {
    require(indices.isEmpty)
    require(values.length == status(1) * status(2))
    require(status.length == 3)

  } else if (status(0) == 3) {
    require(indices.length * status(1) == values.length)
    require(status.length == 3)
  }

  def mode: Int = status(0)

  def size: Int = status(1)

  def vectorSize: Int = status(2)

  def isEmpty: Boolean = size == 0

  def nonEmpty: Boolean = !isEmpty

  def iterator()
              (implicit ck: ClassTag[K], nek: NumericExt[K],
               cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): Iterator[KVVector[K, V]] = {

    mode match {
      case 0 =>
        // all vectors are empty
        if (size != 0) {
          val empty = KVVector.empty[K, V]
          Iterator.fill(size)(empty)
        } else {
          Iterator.empty
        }


      case 1 =>
        // dense and sparse
        val size_ = size

        new Iterator[KVVector[K, V]]() {
          private var i = 0
          private var indexIdx = 0
          private var valueIdx = 0

          private val emptyVec = KVVector.sparse[K, V](vectorSize, nek.emptyArray, nev.emptyArray)

          override def hasNext: Boolean = i < size_

          override def next(): KVVector[K, V] = {
            val step = status(i + 3)

            if (step > 0) {
              val vector = KVVector.dense[K, V](nev.slice(values, valueIdx, valueIdx + step))

              valueIdx += step
              i += 1

              vector

            } else if (step < 0) {
              val vector = KVVector.sparse[K, V](vectorSize,
                nek.slice(indices, indexIdx, indexIdx - step),
                nev.slice(values, valueIdx, valueIdx - step))

              indexIdx -= step
              valueIdx -= step
              i += 1

              vector

            } else {
              i += 1
              emptyVec
            }
          }
        }


      case 2 =>
        // all vectors are dense
        values.grouped(vectorSize)
          .map(KVVector.dense[K, V])


      case 3 =>
        // all vectors are sparse, and share the same indices.
        // the indices are stored in array `indices`,
        // the values of all vectors are stored in array `values`
        values.grouped(indices.length)
          .map { array => KVVector.sparse[K, V](vectorSize, indices, array) }


      case 4 =>
        // all vectors are sparse, and share the same indices.
        // the indices are stored in array `indices`,
        // the values of all vectors are stored as a sparse vector,
        // whose indices are in array `status` (after 3 elements), and values in array `values`.
        val valueVec = KVVector.sparse[Int, V](indices.length * size,
          ju.Arrays.copyOfRange(status, 3, status.length), values)

        valueVec.iterator.map(_._2)
          .grouped(indices.length)
          .map { seq => KVVector.sparse[K, V](vectorSize, indices, seq.toArray) }
    }
  }


  def activeIterator()
                    (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
                     cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): Iterator[Iterator[(K, V)]] = {

    mode match {
      case 0 =>
        // all vectors are empty
        if (size != 0) {
          Iterator.fill(size)(Iterator.empty)
        } else {
          Iterator.empty
        }


      case 1 =>
        // dense and sparse
        val size_ = size

        new Iterator[Iterator[(K, V)]]() {
          private var i = 0
          private var indexIdx = 0
          private var valueIdx = 0

          override def hasNext: Boolean = i < size_

          override def next(): Iterator[(K, V)] = {
            val step = status(i + 3)

            if (step > 0) {
              val vi = valueIdx

              valueIdx += step
              i += 1

              Iterator.range(0, step)
                .flatMap { j =>
                  val v = values(vi + j)
                  if (v != nuv.zero) {
                    Iterator.single((ink.fromInt(j), v))
                  } else {
                    Iterator.empty
                  }
                }

            } else if (step < 0) {
              val ii = indexIdx
              val vi = valueIdx

              indexIdx -= step
              valueIdx -= step
              i += 1

              Iterator.range(0, -step)
                .flatMap { j =>
                  val v = values(vi + j)
                  if (v != nuv.zero) {
                    Iterator.single((indices(ii + j), v))
                  } else {
                    Iterator.empty
                  }
                }

            } else {
              i += 1
              Iterator.empty
            }
          }
        }


      case 2 =>
        // all vectors are dense
        val size_ = size

        new Iterator[Iterator[(K, V)]]() {
          private var i = 0
          private var valueIdx = 0

          override def hasNext: Boolean = i < size_

          override def next(): Iterator[(K, V)] = {
            val vi = valueIdx

            valueIdx += vectorSize
            i += 1

            Iterator.range(0, vectorSize)
              .flatMap { j =>
                val v = values(vi + j)
                if (v != nuv.zero) {
                  Iterator.single((ink.fromInt(j), v))
                } else {
                  Iterator.empty
                }
              }
          }
        }


      case 3 =>
        // all vectors are sparse, and share the same indices.
        // the indices are stored in array `indices`,
        // the values of all vectors are stored in array `values`
        val size_ = size

        new Iterator[Iterator[(K, V)]]() {
          private var i = 0
          private var valueIdx = 0

          override def hasNext: Boolean = i < size_

          override def next(): Iterator[(K, V)] = {
            val vi = valueIdx

            valueIdx += indices.length
            i += 1

            Iterator.range(0, indices.length)
              .flatMap { j =>
                val v = values(vi + j)
                if (v != nuv.zero) {
                  Iterator.single((indices(j), v))
                } else {
                  Iterator.empty
                }
              }
          }
        }


      case 4 =>
        // all vectors are sparse, and share the same indices.
        // the indices are stored in array `indices`,
        // the values of all vectors are stored as a sparse vector,
        // whose indices are in array `status` (after 3 elements), and values in array `values`.
        val valueVec = KVVector.sparse[Int, V](indices.length * size,
          ju.Arrays.copyOfRange(status, 3, status.length), values)

        valueVec.iterator.map(_._2)
          .grouped(indices.length)
          .map {
            _.iterator.zip(indices.iterator)
              .flatMap { case (v, j) =>
                if (v != nuv.zero) {
                  Iterator.single((j, v))
                } else {
                  Iterator.empty
                }
              }
          }
    }
  }
}


object KVMatrix extends Serializable {

  def build[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int) V](iterator: Iterator[KVVector[K, V]])
                                                                 (implicit ck: ClassTag[K], nek: NumericExt[K],
                                                                  cv: ClassTag[V], nev: NumericExt[V]): KVMatrix[K, V] = {

    val indexBuilder = mutable.ArrayBuilder.make[K]
    val valueBuilder = mutable.ArrayBuilder.make[V]
    val stepBuilder = mutable.ArrayBuilder.make[Int]

    var count = 0
    var vectorSize = -1
    var allDense = true

    while (iterator.hasNext) {
      val vector = iterator.next()

      count += 1

      if (vectorSize < 0) {
        vectorSize = vector.size
      }
      require(vectorSize == vector.size)

      vector match {
        case dv: DenseKVVector[K, V] =>
          valueBuilder ++= dv.values
          stepBuilder += dv.values.length

        case sv: SparseKVVector[K, V] =>
          allDense = false
          indexBuilder ++= sv.indices
          valueBuilder ++= sv.values
          stepBuilder += -sv.values.length
      }
    }

    if (vectorSize < 0) {
      // mode 0: empty input
      new KVMatrix[K, V](nek.emptyArray, nev.emptyArray, Array(0, 0, -1))

    } else if (vectorSize == 0) {
      // mode 0: all vectors are empty
      new KVMatrix[K, V](nek.emptyArray, nev.emptyArray, Array(0, count, 0))

    } else {

      if (!allDense) {
        // mode 1: dense and sparse
        new KVMatrix[K, V](indexBuilder.result(), valueBuilder.result(), Array(1, count, vectorSize) ++ stepBuilder.result())

      } else {
        // mode 2: all vectors are dense
        new KVMatrix[K, V](nek.emptyArray, valueBuilder.result(), Array(2, count, vectorSize))
      }
    }
  }


  def build[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int) V](list: Iterable[KVVector[K, V]])
                                                                 (implicit ck: ClassTag[K], nek: NumericExt[K],
                                                                  cv: ClassTag[V], nev: NumericExt[V]): KVMatrix[K, V] = {
    build[K, V](list.iterator)
  }


  def sliceAndBuild[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int) V](indices: Array[K],
                                                                          iterator: Iterator[KVVector[K, V]])
                                                                         (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
                                                                          cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): KVMatrix[K, V] = {

    require(indices.nonEmpty)

    val indices_ = nek.toInt(indices)

    var count = 0
    var vectorSize = -1
    val valueBuilder = mutable.ArrayBuilder.make[V]

    while (iterator.hasNext) {
      val vector = iterator.next()

      count += 1

      if (vectorSize < 0) {
        vectorSize = vector.size
      }
      require(vectorSize == vector.size)

      var i = 0
      while (i < indices.length) {
        valueBuilder += vector(indices_(i))
        i += 1
      }
    }


    if (vectorSize < 0) {
      // mode 0: empty input
      new KVMatrix[K, V](nek.emptyArray, nev.emptyArray, Array(0, 0, -1))

    } else {

      val valueVector = KVVector.dense[Int, V](valueBuilder.result()).compress()

      valueVector match {
        case dv: DenseKVVector[Int, V] =>
          // mode 3
          new KVMatrix[K, V](indices, dv.values, Array(3, count, vectorSize))

        case sv: SparseKVVector[Int, V] =>
          // mode 4
          new KVMatrix[K, V](indices, sv.values, Array(4, count, vectorSize) ++ sv.indices)
      }
    }
  }


  def sliceAndBuild[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int) V](indices: Array[K],
                                                                          list: Iterable[KVVector[K, V]])
                                                                         (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
                                                                          cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): KVMatrix[K, V] = {
    sliceAndBuild[K, V](indices, list.iterator)
  }
}



