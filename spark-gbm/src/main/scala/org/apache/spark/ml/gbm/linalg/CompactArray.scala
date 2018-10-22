package org.apache.spark.ml.gbm.linalg

import scala.reflect.ClassTag
import scala.{specialized => spec}

import org.apache.spark.ml.gbm.util.Utils

/**
  * Compress a block of values in a compact fashion.
  * The continuous identical values will be compressed.
  */
class CompactArray[@spec(Byte, Short, Int, Long, Float, Double) V](val values: Array[V],
                                                                   val times: Array[Int]) extends Serializable {

  require(values.length == times.length)
  require(times.forall(_ > 0))

  def size: Int = times.sum

  def isEmpty: Boolean = size == 0

  def nonEmpty: Boolean = !isEmpty

  def iterator(): Iterator[V] =
    values.iterator.zip(times.iterator)
      .flatMap { case (value, time) => Iterator.fill(time)(value) }
}


object CompactArray extends Serializable {

  def build[@spec(Byte, Short, Int, Long, Float, Double) V](iterator: Iterator[V])
                                                           (implicit cv: ClassTag[V], orv: Ordering[V]): CompactArray[V] = {

    val iter2 = iterator.map(v => (v, 1))
    val (values, times) = Utils.reduceByKey[V, Int](iter2, _ + _).toArray.unzip
    new CompactArray[V](values, times)
  }

  def build[@spec(Byte, Short, Int, Long, Float, Double) V](seq: Iterable[V])
                                                           (implicit cv: ClassTag[V], orv: Ordering[V]): CompactArray[V] = {
    build[V](seq.iterator)
  }
}

