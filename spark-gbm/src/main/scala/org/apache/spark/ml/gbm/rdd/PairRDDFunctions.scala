package org.apache.spark.ml.gbm.rdd

import java.nio.ByteBuffer

import scala.reflect.ClassTag

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.util.collection.ExternalSorter


private[gbm] class PairRDDFunctions[K, V](self: RDD[(K, V)])
                                         (implicit ct: ClassTag[K], cv: ClassTag[V])
  extends Logging with Serializable {

  /**
    * Perform `aggregateByKey` within partitions.
    * If ordering is provided, sorted the result partitions.
    */
  def aggregateByKeyWithinPartitions[C: ClassTag](zeroValue: C,
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
  def aggregateByKeyWithinPartitions[C: ClassTag](zeroValue: C)
                                                 (seqOp: (C, V) => C,
                                                  combOp: (C, C) => C): RDD[(K, C)] = {
    aggregateByKeyWithinPartitions[C](zeroValue, None)(seqOp, combOp)
  }
}

private[gbm] object PairRDDFunctions {

  /** Implicit conversion from an RDD to PairRDDFunctions. */
  implicit def fromRDD[K: ClassTag, V: Ordering : ClassTag](rdd: RDD[(K, V)]): PairRDDFunctions[K, V] = {
    new PairRDDFunctions[K, V](rdd)
  }
}

