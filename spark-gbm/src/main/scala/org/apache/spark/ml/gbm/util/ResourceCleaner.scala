package org.apache.spark.ml.gbm.util

import scala.collection.mutable

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel


private[gbm] class ResourceCleaner extends Logging {

  private val cachedDatasets = mutable.ArrayBuffer.empty[Dataset[_]]

  private val cachedRDDs = mutable.ArrayBuffer.empty[RDD[_]]

  private val checkpointedRDDs = mutable.ArrayBuffer.empty[RDD[_]]

  private val broadcastedObjects = mutable.ArrayBuffer.empty[Broadcast[_]]

  def registerCachedDatasets(datasets: Dataset[_]*): Unit = {
    cachedDatasets.appendAll(datasets)
  }

  def registerCachedRDDs(rdds: RDD[_]*): Unit = {
    cachedRDDs.appendAll(rdds)
  }

  def registerCheckpointedRDDs(rdds: RDD[_]*): Unit = {
    checkpointedRDDs.appendAll(rdds)
  }

  def registerBroadcastedObjects(bcs: Broadcast[_]*): Unit = {
    broadcastedObjects.appendAll(bcs)
  }

  def clear(blocking: Boolean = true): Unit = {
    cachedDatasets.foreach { dataset =>
      if (dataset.storageLevel != StorageLevel.NONE) {
        dataset.unpersist(blocking)
      }
    }
    cachedDatasets.clear()

    cachedRDDs.foreach { rdd =>
      if (rdd.getStorageLevel != StorageLevel.NONE) {
        rdd.unpersist(blocking)
      }
    }
    cachedRDDs.clear()

    checkpointedRDDs.foreach { rdd =>
      Utils.removeCheckpointFile(rdd, blocking)
    }
    checkpointedRDDs.clear()

    broadcastedObjects.foreach(_.destroy(blocking))
    broadcastedObjects.clear()
  }
}
