package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random
import org.apache.spark.rdd._
import org.apache.spark.util.random.XORShiftRandom

private[gbm] class RDDFunctions[T: ClassTag](self: RDD[T]) extends Serializable {

  def samplePartitions(weights: Map[Int, Double], seed: Long): RDD[T] = {
    weights.foreach { case (pid, weight) =>
      require(pid >= 0 && pid < self.getNumPartitions)
      require(weight >= 0 && weight <= 1)
    }

    self.mapPartitionsWithIndex { case (pid, it) =>
      val w = weights.getOrElse(pid, 0.0)
      if (w == 1) {
        it
      } else if (w == 0) {
        Iterator.empty
      } else {
        val rng = new XORShiftRandom(pid + seed)
        it.filter(_ => rng.nextDouble < w)
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

    shuffled.take(m).foreach { p =>
      weights.update(p, 1.0)
    }

    if (r > 0) {
      weights.update(shuffled.last, r)
    }

    samplePartitions(weights.toMap, seed)
  }
}


private[gbm] object RDDFunctions {

  /** Implicit conversion from an RDD to RDDFunctions. */
  implicit def fromRDD[T: ClassTag](rdd: RDD[T]): RDDFunctions[T] = new RDDFunctions[T](rdd)
}

