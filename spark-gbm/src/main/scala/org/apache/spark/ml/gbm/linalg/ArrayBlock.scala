package org.apache.spark.ml.gbm.linalg

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.{specialized => spec}


/**
  * Compress a block of arrays in a compact fashion.
  *
  * @param values Values of all arrays.
  * @param steps  Lengths of all arrays. If all arrays are of the same length, it will be empty.
  * @param flag   If positive, means all arrays are of the same positive length, the length is here.
  *               If negative, means all arrays are the identical (same length and values), the number of arrays is here.
  *               If zero, meaningless.
  */
class ArrayBlock[@spec(Byte, Short, Int, Long, Float, Double) V](val values: Array[V],
                                                                 val steps: Array[Int],
                                                                 val flag: Int) extends Serializable {

  if (flag > 0) {
    require(values.length % flag == 0)
  }

  def isEmpty: Boolean = size == 0

  def nonEmpty: Boolean = !isEmpty

  def size: Int = {
    if (flag == 0) {
      steps.length
    } else if (flag > 0) {
      values.length / flag
    } else {
      -flag
    }
  }


  def iterator()
              (implicit cv: ClassTag[V], nev: NumericExt[V]): Iterator[Array[V]] = {

    if (flag == 0) {
      new Iterator[Array[V]]() {
        var i = 0
        var offset = 0

        override def hasNext: Boolean = i < steps.length

        override def next(): Array[V] = {
          val step = steps(i)

          val ret = nev.slice(values, offset, offset + step)

          offset += step
          i += 1

          ret
        }
      }

    } else if (flag > 0) {
      values.grouped(flag)

    } else {
      // IMPORTANT!
      // Iterator.fill(-flag)(values) DO NOT work!
      // adopt `clone` to avoid possible in-place modification.
      Iterator.range(0, -flag).map(_ => values.clone())
    }
  }
}


object ArrayBlock extends Serializable {

  def empty[@spec(Byte, Short, Int, Long, Float, Double) V]()
                                                           (implicit cv: ClassTag[V], nev: NumericExt[V]): ArrayBlock[V] = {
    new ArrayBlock[V](nev.emptyArray, Array.emptyIntArray, 0)
  }

  def build[@spec(Byte, Short, Int, Long, Float, Double) V](iterator: Iterator[Array[V]])
                                                           (implicit cv: ClassTag[V], nev: NumericExt[V]): ArrayBlock[V] = {
    val builder = new ArrayBlockBuilder[V]

    while (iterator.hasNext) {
      builder += iterator.next()
    }

    builder.result()
  }


  def build[@spec(Byte, Short, Int, Long, Float, Double) V](seq: Iterable[Array[V]])
                                                           (implicit cv: ClassTag[V], nev: NumericExt[V]): ArrayBlock[V] = {
    build[V](seq.iterator)
  }


  def fill[@spec(Byte, Short, Int, Long, Float, Double) V](n: Int,
                                                           array: Array[V])
                                                          (implicit cv: ClassTag[V], nev: NumericExt[V]): ArrayBlock[V] = {
    require(array != null)
    require(n >= 0)

    if (n > 0) {
      new ArrayBlock[V](array, Array.emptyIntArray, -n)
    } else {
      new ArrayBlock[V](nev.emptyArray, Array.emptyIntArray, 0)
    }
  }
}


private[gbm] class ArrayBlockBuilder[@spec(Byte, Short, Int, Long, Float, Double) V](implicit cv: ClassTag[V], nev: NumericExt[V]) extends Serializable {

  private val valueBuilder = mutable.ArrayBuilder.make[V]
  private val stepBuilder = mutable.ArrayBuilder.make[Int]

  private var identical = true
  private var prevArray: Array[V] = null

  def +=(array: Array[V]): this.type = {
    require(array != null)

    if (prevArray == null) {
      prevArray = array
    } else if (identical) {
      identical = nev.equalsArray(prevArray, array)
      prevArray = array
    }

    valueBuilder ++= array
    stepBuilder += array.length

    this
  }


  def result(): ArrayBlock[V] = {
    val values = valueBuilder.result()
    val steps = stepBuilder.result()

    if (identical && prevArray != null) {
      // all arrays are identical
      new ArrayBlock[V](prevArray, Array.emptyIntArray, -steps.length)

    } else if (steps.distinct.length == 1 && steps.head > 0) {
      // all arrays are of the same size
      new ArrayBlock[V](values, Array.emptyIntArray, steps.head)

    } else {

      new ArrayBlock[V](values, steps, 0)
    }
  }
}


