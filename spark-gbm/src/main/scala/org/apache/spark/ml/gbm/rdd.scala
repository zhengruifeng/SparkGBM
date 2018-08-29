package org.apache.spark.ml.gbm

import java.nio.ByteBuffer

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark._
import org.apache.spark.internal.Logging
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


class PairRDDFunctions[K, V](self: RDD[(K, V)])
                            (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K] = null)
  extends Logging with Serializable {

  def aggregatePartitionsByKey[U: ClassTag](zeroValue: U)
                                           (seqOp: (U, V) => U,
                                            combOp: (U, U) => U): RDD[(K, U)] = self.withScope {

    // Serialize the zero value to a byte array so that we can get a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    lazy val cachedSerializer = SparkEnv.get.serializer.newInstance()
    val createZero = () => cachedSerializer.deserialize[U](ByteBuffer.wrap(zeroArray))

    // We will clean the combiner closure later in `combineByKey`
    val mergeValue = self.context.clean(seqOp)

    val createCombiner = (v: V) => mergeValue(createZero(), v)
    val mergeCombiners = combOp

    require(mergeCombiners != null, "mergeCombiners must be defined") // required as of Spark 0.9.0
    if (kt.runtimeClass.isArray) {
      throw new SparkException("Cannot use map-side combining with array keys.")
    }
    val aggregator = new Aggregator[K, V, U](
      self.context.clean(createCombiner),
      self.context.clean(mergeValue),
      self.context.clean(mergeCombiners))
    self.mapPartitions(iter => {
      val context = TaskContext.get()
      new InterruptibleIterator(context, aggregator.combineValuesByKey(iter, context))
    }, preservesPartitioning = true)
  }
}

private[gbm] object PairRDDFunctions {
  /** Implicit conversion from an RDD to PairRDDFunctions. */
  implicit def fromRDD[K: ClassTag, V: Ordering : ClassTag](rdd: RDD[(K, V)]): PairRDDFunctions[K, V] = new PairRDDFunctions[K, V](rdd)
}