package org.apache.spark.ml.gbm.linalg

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.{specialized => spec}


/**
  * Compress a block of arrays in a compact fashion.
  *
  * @param values      Values of all arrays.
  * @param steps       Lengths of all arrays. If all arrays are of the same length, it will be empty.
  * @param defaultStep If all arrays are of the same length, the length is here.
  */
class ArrayBlock[@spec(Byte, Short, Int, Long, Float, Double) V](val values: Array[V],
                                                                 val steps: Array[Int],
                                                                 val defaultStep: Int) extends Serializable {
  if (steps.nonEmpty) {
    require(defaultStep == 0)
  } else if (defaultStep > 0) {
    require(values.length % defaultStep == 0)
  }

  def isEmpty: Boolean = size == 0

  def size: Int = {
    if (steps.nonEmpty) {
      steps.length
    } else if (defaultStep > 0) {
      values.length / defaultStep
    } else {
      0
    }
  }

  def iterator()
              (implicit cv: ClassTag[V]): Iterator[Array[V]] = {
    if (steps.nonEmpty) {

      new Iterator[Array[V]]() {
        var i = 0
        var offset = 0

        val builder = mutable.ArrayBuilder.make[V]

        override def hasNext: Boolean = i < steps.length

        override def next(): Array[V] = {
          builder.clear()

          val step = steps(i)

          var j = 0
          while (j < step) {
            builder += values(offset + j)
            j += 1
          }

          i += 1
          offset += step
          builder.result()
        }
      }

    } else if (defaultStep > 0) {
      values.grouped(defaultStep)

    } else {
      Iterator.empty
    }
  }
}


object ArrayBlock extends Serializable {

  def empty[V]()
              (implicit cv: ClassTag[V], nev: NumericExt[V]): ArrayBlock[V] = {
    new ArrayBlock[V](nev.emptyArray, Array.emptyIntArray, 0)
  }

  def build[V](iterator: Iterator[Array[V]])
              (implicit cv: ClassTag[V]): ArrayBlock[V] = {
    val valueBuilder = mutable.ArrayBuilder.make[V]
    val stepBuilder = mutable.ArrayBuilder.make[Int]

    iterator.foreach { array =>
      valueBuilder ++= array
      stepBuilder += array.length
    }

    val values = valueBuilder.result()
    val steps = stepBuilder.result()

    if (steps.distinct.length == 1 && steps.head > 0) {
      new ArrayBlock[V](values, Array.emptyIntArray, steps.head)
    } else {
      new ArrayBlock[V](values, steps, 0)
    }
  }

  def fill[V](array: Array[V], n: Int)
             (implicit cv: ClassTag[V]): ArrayBlock[V] = {
    val iter = Iterator.range(0, n).map(_ => array)
    build[V](iter)
  }
}

