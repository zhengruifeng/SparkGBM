package org.apache.spark.ml.gbm

import java.nio.ByteBuffer

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.rdd._
import org.apache.spark.util.random.XORShiftRandom


private[gbm] class PartitionSelectedRDDPartition[T](val idx: Int,
                                                    val prev: Partition)
  extends Partition with Serializable {

  override val index: Int = idx
}


private[gbm] class PartitionSelectedRDD[T: ClassTag](@transient val parent: RDD[T],
                                                     val partIds: Array[Int]) extends RDD[T](parent) {
  require(partIds.forall(p => p >= 0 && p < parent.getNumPartitions))

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val part = split.asInstanceOf[PartitionSelectedRDDPartition[T]]
    firstParent[T].iterator(part.prev, context)
  }

  override protected def getPartitions: Array[Partition] = {
    partIds.zipWithIndex.map { case (partId, i) =>
      val partition = firstParent[T].partitions(partId)
      new PartitionSelectedRDDPartition(i, partition)
    }
  }

  override def getPreferredLocations(split: Partition): Seq[String] =
    firstParent[T].preferredLocations(split.asInstanceOf[PartitionSelectedRDDPartition[T]].prev)

  override def getDependencies: Seq[Dependency[_]] = Seq(
    new NarrowDependency(parent) {
      def getParents(id: Int): Seq[Int] = Seq(partIds(id))
    })
}



private[gbm] class RDDFunctions[T: ClassTag](self: RDD[T]) extends Serializable {

  def selectPartitions(partIds: Array[Int]): RDD[T] = {
    new PartitionSelectedRDD[T](self, partIds)
  }

  def samplePartitions(weights: Map[Int, Double], seed: Long): RDD[T] = {
    val (partIdArr, weightArr) = weights.toArray.unzip

    selectPartitions(partIdArr)
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

  def extendPartitions(numPartitions: Int): RDD[T] = {
    val prevNumPartitions = self.getNumPartitions
    require(numPartitions >= prevNumPartitions)

    if (numPartitions == prevNumPartitions) {
      self

    } else {
      val n = numPartitions / prevNumPartitions
      val r = numPartitions % prevNumPartitions
      val partIds = Array.tabulate(numPartitions)(_ % prevNumPartitions)

      selectPartitions(partIds)
        .mapPartitionsWithIndex { case (partId, iter) =>
          val i = partId / prevNumPartitions
          val j = partId % prevNumPartitions

          val k = if (j < r) {
            n + 1
          } else {
            n
          }

          iter.zipWithIndex
            .filter(_._2 % k == i)
            .map(_._1)
        }
    }
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
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    lazy val cachedSerializer = SparkEnv.get.serializer.newInstance()
    val createZero = () => cachedSerializer.deserialize[U](ByteBuffer.wrap(zeroArray))

    val mergeValue = self.context.clean(seqOp)

    val createCombiner = (v: V) => mergeValue(createZero(), v)
    val mergeCombiners = combOp

    require(mergeCombiners != null, "mergeCombiners must be defined")
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