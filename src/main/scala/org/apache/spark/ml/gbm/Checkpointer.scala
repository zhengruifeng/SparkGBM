package org.apache.spark.ml.gbm

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.util.PeriodicRDDCheckpointer
import org.apache.spark.storage.StorageLevel

/**
  * This class helps with persisting and checkpointing RDDs.
  */
private[gbm] class Checkpointer[T](sc: SparkContext,
                                   checkpointInterval: Int,
                                   val storageLevel: StorageLevel)
  extends PeriodicRDDCheckpointer[T](checkpointInterval, sc) {
  require(storageLevel != StorageLevel.NONE)

  override protected def persist(data: RDD[T]): Unit = {
    if (data.getStorageLevel == StorageLevel.NONE) {
      data.persist(storageLevel)
    }
  }
}
