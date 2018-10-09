package org.apache.spark.ml.gbm

import java.nio.ByteBuffer

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.rdd._
import org.apache.spark.util.collection.ExternalSorter
import org.apache.spark.util.random.XORShiftRandom


/**
  * Partition of `PartitionReorganizedRDD`
  */
private[gbm] class PartitionReorganizedRDDPartition[T](val index: Int,
                                                       val prevs: Array[Partition]) extends Partition


/**
  * Reorganize and concatenate current partitions to form new partitions.
  * E.g `partIds` = Array(Array(0,0), Array(3,4)), will create a new RDD with 2 partitions,
  * the first one is two copies of current part0, and the second one is composed of current part3 and part4.
  * @param partIds indicate how to form new partitions
  */
private[gbm] class PartitionReorganizedRDD[T: ClassTag](@transient val parent: RDD[T],
                                                        val partIds: Array[Array[Int]]) extends RDD[T](parent) {
  require(partIds.iterator.flatten
    .forall(p => p >= 0 && p < parent.getNumPartitions))

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val part = split.asInstanceOf[PartitionReorganizedRDDPartition[T]]
    part.prevs.iterator.flatMap { prev =>
      firstParent[T].iterator(prev, context)
    }
  }

  override protected def getPartitions: Array[Partition] = {
    partIds.zipWithIndex.map { case (partId, i) =>
      val prevs = partId.map { pid => firstParent[T].partitions(pid) }
      new PartitionReorganizedRDDPartition(i, prevs)
    }
  }

  override def getPreferredLocations(split: Partition): Seq[String] = {
    val prefs = split.asInstanceOf[PartitionReorganizedRDDPartition[T]].prevs
      .map { prev => firstParent[T].preferredLocations(prev) }

    val intersect = prefs.reduce((p1, p2) => p1.intersect(p2))

    if (intersect.nonEmpty) {
      intersect
    } else {
      prefs.flatten.distinct
    }
  }

  override def getDependencies: Seq[Dependency[_]] = Seq(
    new NarrowDependency(parent) {
      def getParents(id: Int): Seq[Int] = partIds(id)
    })
}



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


private[gbm] class PairRDDFunctions[K, V](self: RDD[(K, V)])
                                         (implicit ct: ClassTag[K], cv: ClassTag[V])
  extends Logging with Serializable {

  /**
    * Perform `aggregateByKey` within partitions.
    * If ordering is provided, sorted the result partitions.
    */
  def aggregatePartitionsByKey[C: ClassTag](zeroValue: C,
                                            ordering: Option[Ordering[K]])
                                           (seqOp: (C, V) => C,
                                            combOp: (C, C) => C): RDD[(K, C)] = self.withScope {
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    lazy val cachedSerializer = SparkEnv.get.serializer.newInstance()
    val createZero = () => cachedSerializer.deserialize[C](ByteBuffer.wrap(zeroArray))

    val mergeValue = self.context.clean(seqOp)
    val createCombiner = (v: V) => mergeValue(createZero(), v)
    val mergeCombiners = combOp

    require(mergeCombiners != null, "mergeCombiners must be defined")
    require(!ct.runtimeClass.isArray, "Cannot use map-side combining with array keys.")

    val aggregator = new Aggregator[K, V, C](
      self.context.clean(createCombiner),
      self.context.clean(mergeValue),
      self.context.clean(mergeCombiners))

    self.mapPartitions(iter => {
      val context = TaskContext.get()

      if (ordering.nonEmpty) {
        val sorter = new ExternalSorter[K, V, C](context, Some(aggregator), None, ordering)
        sorter.insertAll(iter)
        val outIter = sorter.iterator.asInstanceOf[Iterator[(K, C)]]
        new InterruptibleIterator(context, outIter)

      } else {
        val outIter = aggregator.combineValuesByKey(iter, context)
        new InterruptibleIterator(context, outIter)
      }

    }, preservesPartitioning = true)
  }


  /**
    * Perform `aggregateByKey` within partitions, and ignore output ordering.
    */
  def aggregatePartitionsByKey[C: ClassTag](zeroValue: C)
                                           (seqOp: (C, V) => C,
                                            combOp: (C, C) => C): RDD[(K, C)] = {
    aggregatePartitionsByKey[C](zeroValue, None)(seqOp, combOp)
  }
}

private[gbm] object PairRDDFunctions {

  /** Implicit conversion from an RDD to PairRDDFunctions. */
  implicit def fromRDD[K: ClassTag, V: Ordering : ClassTag](rdd: RDD[(K, V)]): PairRDDFunctions[K, V] = {
    new PairRDDFunctions[K, V](rdd)
  }
}