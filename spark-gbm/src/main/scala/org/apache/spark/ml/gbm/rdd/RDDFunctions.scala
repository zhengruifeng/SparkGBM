package org.apache.spark.ml.gbm.rdd

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom


private[gbm] class RDDFunctions[T: ClassTag](self: RDD[T]) extends Serializable {

  /**
    * Concatenate current partitions to form new partitions.
    */
  def catPartitions(partIds: Array[Array[Int]]): RDD[T] = {
    new PartitionConcatenatedRDD[T](self, partIds)
  }


  def catPartitions(partIds: Array[Int]): RDD[T] = {
    val array = partIds.map(p => Array(p))
    catPartitions(array)
  }


  /**
    * for partition with weight = 1, directly sample it instead of rows.
    * for partition with weight < 1, perform row sampling.
    */
  def samplePartitions(weights: Map[Int, Double], seed: Long): RDD[T] = {
    val (partIdArr, weightArr) = weights.toArray.unzip

    catPartitions(partIdArr)
      .mapPartitionsWithIndex { case (pid, iter) =>
        val w = weightArr(pid)
        if (w == 0) {
          Iterator.empty
        } else if (w == 1) {
          iter
        } else {
          val rng = new XORShiftRandom(seed + pid)
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

      catPartitions(partIds)
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


  /**
    * Perform `allgather` among partitions so that each result partition contains all the
    * values, and keep the original global ordering.
    */
  def allgather(numParts: Int = self.getNumPartitions): RDD[T] = {
    allgather(numParts, Array.range(0, numParts))
  }


  /**
    * Perform `allgather` among partitions so that each selected result partition
    * contains all the values, and keep the original global ordering.
    */
  def allgather(numParts: Int,
                partIds: Array[Int]): RDD[T] = {
    require(partIds.nonEmpty)
    require(partIds.distinct.length == partIds.length)
    require(partIds.forall(partId => 0 <= partId && partId < numParts))

    self.mapPartitionsWithIndex { case (sourcePartId, iter) =>
      var cnt = -1L

      iter.flatMap { value =>
        cnt += 1
        partIds.iterator.map { destPartId =>
          ((sourcePartId, cnt, destPartId), value)
        }
      }

    }.repartitionAndSortWithinPartitions(new Partitioner {

      override def numPartitions: Int = numParts

      override def getPartition(key: Any): Int = key match {
        case (_, _, destPartId: Int) => destPartId
      }

    }).map(_._2)
  }


  def zip2[V1: ClassTag, V2: ClassTag](rdd1: RDD[V1],
                                       rdd2: RDD[V2],
                                       validate: Boolean = true): RDD[(T, V1, V2)] = {

    self.zipPartitions(rdd1, rdd2, false) {
      (thisIter, iter1, iter2) => Utils.zip3(thisIter, iter1, iter2, validate)
    }
  }


  def zip3[V1: ClassTag, V2: ClassTag, V3: ClassTag](rdd1: RDD[V1],
                                                     rdd2: RDD[V2],
                                                     rdd3: RDD[V3],
                                                     validate: Boolean = true): RDD[(T, V1, V2, V3)] = {

    self.zipPartitions(rdd1, rdd2, rdd3, false) {
      (thisIter, iter1, iter2, iter3) => Utils.zip4(thisIter, iter1, iter2, iter3, validate)
    }
  }
}


private[gbm] object RDDFunctions {

  /** Implicit conversion from an RDD to RDDFunctions. */
  implicit def fromRDD[T: ClassTag](rdd: RDD[T]): RDDFunctions[T] = {
    new RDDFunctions[T](rdd)
  }
}