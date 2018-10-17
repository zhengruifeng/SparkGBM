package org.apache.spark.ml.gbm.linalg

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.{specialized => spec}


/**
  * Compress a block of vectors in a compact fashion.
  * Note: all vectors are of the same length.
  *
  * @param indices    concatenated indices of SparseVectors
  * @param values     concatenated indices of both DenseVectors and SparseVectors
  * @param steps      length of vector-values (not vector size). If empty, means all vectors are dense.
  *                   Positive step indicate that the vector is a DenseVector,
  *                   Negative step indicate that the vector is a SparseVector.
  * @param vectorSize length of vector.
  */
class KVMatrix[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int) V](val indices: Array[K],
                                                                     val values: Array[V],
                                                                     val steps: Array[Int],
                                                                     val vectorSize: Int) extends Serializable {
  require(vectorSize >= 0)

  if (steps.nonEmpty) {
    require(steps.length == size)
  } else if (vectorSize > 0) {
    require(values.length % vectorSize == 0)
  }


  def size: Int = {
    if (steps.nonEmpty) {
      steps.length
    } else if (vectorSize > 0) {
      values.length / vectorSize
    } else {
      0
    }
  }

  def isEmpty: Boolean = size == 0

  def nonEmpty: Boolean = !isEmpty

  private def getStep(i: Int): Int = {
    if (steps.nonEmpty) {
      steps(i)
    } else {
      vectorSize
    }
  }

  def iterator()
              (implicit ck: ClassTag[K], nek: NumericExt[K],
               cv: ClassTag[V], nev: NumericExt[V]): Iterator[KVVector[K, V]] = {
    val size_ = size

    new Iterator[KVVector[K, V]]() {
      private var i = 0
      private var indexIdx = 0
      private var valueIdx = 0

      private val indexBuilder = mutable.ArrayBuilder.make[K]
      private val valueBuilder = mutable.ArrayBuilder.make[V]

      {
        indexBuilder.sizeHint(vectorSize)
        valueBuilder.sizeHint(vectorSize)
      }

      private val emptyVec = KVVector.sparse[K, V](vectorSize, nek.emptyArray, nev.emptyArray)

      override def hasNext: Boolean = i < size_

      override def next(): KVVector[K, V] = {
        val step = getStep(i)

        if (step > 0) {
          valueBuilder.clear()

          var j = 0
          while (j < step) {
            valueBuilder += values(valueIdx + j)
            j += 1
          }

          i += 1
          valueIdx += step

          KVVector.dense[K, V](valueBuilder.result())

        } else if (step < 0) {
          indexBuilder.clear()
          valueBuilder.clear()

          var j = 0
          while (j < -step) {
            indexBuilder += indices(indexIdx + j)
            valueBuilder += values(valueIdx + j)
            j += 1
          }

          i += 1
          indexIdx -= step
          valueIdx -= step

          KVVector.sparse[K, V](vectorSize, indexBuilder.result(), valueBuilder.result())

        } else {

          i += 1
          emptyVec
        }
      }
    }
  }


  def activeIterator()
                    (implicit ck: ClassTag[K], ink: Integral[K], nek: NumericExt[K],
                     cv: ClassTag[V], nuv: Numeric[V], nev: NumericExt[V]): Iterator[Iterator[(K, V)]] = {

    val size_ = size

    new Iterator[Iterator[(K, V)]]() {
      private var i = 0
      private var indexIdx = 0
      private var valueIdx = 0

      override def hasNext: Boolean = i < size_

      override def next(): Iterator[(K, V)] = {
        val step = getStep(i)

        if (step > 0) {

          val ret = Iterator.range(0, step).flatMap { j =>
            val v = values(valueIdx + j)
            if (v != nuv.zero) {
              Iterator.single(ink.fromInt(j), v)
            } else {
              Iterator.empty
            }
          }

          i += 1
          valueIdx += step

          ret

        } else if (step < 0) {

          val ret = Iterator.range(0, -step).flatMap { j =>
            val k = indices(indexIdx + j)
            val v = values(valueIdx + j)

            if (v != nuv.zero) {
              Iterator.single(k, v)
            } else {
              Iterator.empty
            }
          }

          i += 1
          indexIdx -= step
          valueIdx -= step

          ret

        } else {

          i += 1

          Iterator.empty
        }
      }
    }
  }
}


private[gbm] object KVMatrix extends Serializable {

  def build[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int) V](iterator: Iterator[KVVector[K, V]])
                                                                 (implicit ck: ClassTag[K],
                                                                  cv: ClassTag[V]): KVMatrix[K, V] = {
    val indexBuilder = mutable.ArrayBuilder.make[K]
    val valueBuilder = mutable.ArrayBuilder.make[V]
    val stepBuilder = mutable.ArrayBuilder.make[Int]

    var allDense = true
    var vecSize = -1

    iterator.foreach { vec =>
      require(vec.size > 0)
      if (vecSize < 0) {
        vecSize = vec.size
      }
      require(vecSize == vec.size)

      vec match {
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

    val steps = if (allDense) {
      Array.emptyIntArray
    } else {
      stepBuilder.result
    }

    new KVMatrix[K, V](indexBuilder.result(), valueBuilder.result(), steps, vecSize)
  }

  def build[@spec(Byte, Short, Int) K, @spec(Byte, Short, Int) V](seq: Iterable[KVVector[K, V]])
                                                                 (implicit ck: ClassTag[K],
                                                                  cv: ClassTag[V]): KVMatrix[K, V] = {
    build[K, V](seq.iterator)
  }
}

