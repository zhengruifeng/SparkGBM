package org.apache.spark.ml.gbm

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random
import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.rdd._
import org.apache.spark.serializer.Serializer
import org.apache.spark.util.random.XORShiftRandom

private[gbm] class PartitionSampledRDDPartition[T](val idx: Int,
                                                   val prev: Partition,
                                                   val weight: Double,
                                                   val seed: Long)
  extends Partition with Serializable {

  require(weight >= 0 && weight <= 1)

  override val index: Int = idx
}

private[gbm] class PartitionSampledRDD[T: ClassTag](@transient val parent: RDD[T],
                                                    val weights: Map[Int, Double],
                                                    @transient private val seed: Long = System.nanoTime)
  extends RDD[T](parent) {

  require(weights.keys.forall(p => p >= 0))
  require(weights.values.forall(w => w >= 0 && w <= 1))

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val part = split.asInstanceOf[PartitionSampledRDDPartition[T]]

    if (part.weight == 0) {
      Iterator.empty
    } else if (part.weight == 1) {
      firstParent[T].iterator(part.prev, context)
    } else {
      val rng = new XORShiftRandom(part.seed)
      firstParent[T].iterator(part.prev, context)
        .filter(_ => rng.nextDouble < part.weight)
    }
  }

  override def getPreferredLocations(split: Partition): Seq[String] =
    firstParent[T].preferredLocations(split.asInstanceOf[PartitionSampledRDDPartition[T]].prev)

  override protected def getPartitions: Array[Partition] = {
    val rng = new XORShiftRandom(seed)
    var idx = -1

    firstParent[T].partitions
      .filter(p => weights.contains(p.index))
      .map { p =>
        idx += 1
        new PartitionSampledRDDPartition(idx, p, weights(p.index), rng.nextLong)
      }
  }
}

private[gbm] class RDDFunctions[T: ClassTag](self: RDD[T]) extends Serializable {

  def samplePartitions(weights: Map[Int, Double], seed: Long): RDD[T] = {
    new PartitionSampledRDD[T](self, weights, seed)
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

  def aggregatePartitionsByKey[C](createCombiner: V => C,
                                  mergeValue: (C, V) => C,
                                  mergeCombiners: (C, C) => C,
                                  partitioner: Partitioner,
                                  mapSideCombine: Boolean = true,
                                  serializer: Serializer = null)(implicit ct: ClassTag[C]): RDD[(K, C)] = self.withScope {
    require(mergeCombiners != null, "mergeCombiners must be defined") // required as of Spark 0.9.0
    if (kt.runtimeClass.isArray) {
      if (mapSideCombine) {
        throw new SparkException("Cannot use map-side combining with array keys.")
      }
      if (partitioner.isInstanceOf[HashPartitioner]) {
        throw new SparkException("HashPartitioner cannot partition array keys.")
      }
    }
    val aggregator = new Aggregator[K, V, C](
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