package org.apache.spark.ml.gbm.rdd

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom


private[gbm] class RDDFunctions[T: ClassTag](self: RDD[T]) extends Serializable {

  /**
    * Reorganize and concatenate current partitions to form new partitions.
    * Note: This may harms task locality.
    */
  def reorgPartitions(partIds: Array[Array[Int]]): RDD[T] = {
    new PartitionReorganizedRDD[T](self, partIds)
  }


  def reorgPartitions(partIds: Array[Int]): RDD[T] = {
    val array = partIds.map(p => Array(p))
    reorgPartitions(array)
  }


  /**
    * for partition with weight = 1, directly sample it instead of rows.
    * for partition with weight < 1, perform row sampling.
    */
  def samplePartitions(weights: Map[Int, Double], seed: Long): RDD[T] = {
    val (partIdArr, weightArr) = weights.toArray.unzip

    reorgPartitions(partIdArr)
      .mapPartitionsWithIndex { case (pid, iter) =>
        val w = weightArr(pid)
        if (w == 0) {
          Iterator.empty
        } else if (w == 1) {
          iter
        } else {
          val rng = new XORShiftRandom(pid + seed)
          iter.filter(_ => rng.nextDouble < w)
        }
      }
  }


  def samplePartitions(fraction: Double, seed: Long): RDD[T] = {
    require(fraction > 0 && fraction <= 1)

    val rng = new Random(seed)

    val numPartitions = self.getNumPartitions

    val n = numPartitions * fraction
    val m = n.toInt
    val r = n - m

    val shuffled = rng.shuffle(Seq.range(0, numPartitions))

    val weights = mutable.OpenHashMap.empty[Int, Double]

    shuffled.take(m).foreach { p => weights.update(p, 1.0) }

    if (r > 0) {
      weights.update(shuffled.last, r)
    }

    samplePartitions(weights.toMap, seed)
  }


  /**
    * Enlarge the number of partitions without shuffle, by traversing parent partitions several times.
    */
  def extendPartitions(numPartitions: Int): RDD[T] = {
    val prevNumPartitions = self.getNumPartitions
    require(numPartitions >= prevNumPartitions)

    if (numPartitions == prevNumPartitions) {
      self

    } else {
      val n = numPartitions / prevNumPartitions
      val r = numPartitions % prevNumPartitions
      val partIds = Array.tabulate(numPartitions)(_ % prevNumPartitions)

      reorgPartitions(partIds)
        .mapPartitionsWithIndex { case (partId, iter) =>
          val i = partId / prevNumPartitions
          val j = partId % prevNumPartitions

          val k = if (j < r) {
            n + 1
          } else {
            n
          }

          if (k == 1 && i == 0) {
            iter
          } else {
            iter.zipWithIndex
              .filter(_._2 % k == i)
              .map(_._1)
          }
        }
    }
  }
}


private[gbm] object RDDFunctions {

  /** Implicit conversion from an RDD to RDDFunctions. */
  implicit def fromRDD[T: ClassTag](rdd: RDD[T]): RDDFunctions[T] = {
    new RDDFunctions[T](rdd)
  }
}