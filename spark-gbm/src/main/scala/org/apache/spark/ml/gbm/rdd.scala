package org.apache.spark.ml.gbm

import java.nio.ByteBuffer

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.rdd._
import org.apache.spark.util.random.XORShiftRandom


private[gbm] class M2NRDDPartition[T](val idx: Int,
                                      val prevs: Array[Partition]) extends Partition {
  override val index: Int = idx
}

private[gbm] class M2NRDD[T: ClassTag](@transient val parent: RDD[T],
                                       val partIds: Array[Array[Int]]) extends RDD[T](parent) {
  require(partIds.iterator.flatten
    .forall(p => p >= 0 && p < parent.getNumPartitions))

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val part = split.asInstanceOf[M2NRDDPartition[T]]
    part.prevs.iterator.flatMap { prev =>
      firstParent[T].iterator(prev, context)
    }
  }

  override protected def getPartitions: Array[Partition] = {
    partIds.zipWithIndex.map { case (partId, i) =>
      val prevs = partId.map { pid => firstParent[T].partitions(pid) }
      new M2NRDDPartition(i, prevs)
    }
  }

  override def getPreferredLocations(split: Partition): Seq[String] = {
    val votes = mutable.OpenHashMap.empty[String, Int]

    split.asInstanceOf[M2NRDDPartition[T]].prevs.iterator
      .flatMap { prev => firstParent[T].preferredLocations(prev) }
      .foreach { location =>
        val cnt = votes.getOrElse(location, 0)
        votes.update(location, cnt + 1)
      }

    votes.toArray.sortBy(_._2).map(_._1).reverse
  }

  override def getDependencies: Seq[Dependency[_]] = Seq(
    new NarrowDependency(parent) {
      def getParents(id: Int): Seq[Int] = partIds(id)
    })
}


private[gbm] class RDDFunctions[T: ClassTag](self: RDD[T]) extends Serializable {

  def selectPartitions(partIds: Array[Int]): RDD[T] = {
    val array = partIds.map(p => Array(p))
    new M2NRDD[T](self, array)
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