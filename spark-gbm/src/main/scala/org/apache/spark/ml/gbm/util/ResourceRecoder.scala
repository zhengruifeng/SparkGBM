package org.apache.spark.ml.gbm.util

import scala.collection.mutable

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel


private[gbm] class ResourceRecoder extends Logging {

  private val datasetBuff = mutable.ArrayBuffer.empty[Dataset[_]]

  private val rddBuff = mutable.ArrayBuffer.empty[RDD[_]]

  private val bcBuff = mutable.ArrayBuffer.empty[Broadcast[_]]

  def append(dataset: Dataset[_]): Unit = {
    datasetBuff.append(dataset)
  }

  def append(rdd: RDD[_]): Unit = {
    rddBuff.append(rdd)
  }

  def append(bc: Broadcast[_]): Unit = {
    bcBuff.append(bc)
  }

  def clear(blocking: Boolean = true): Unit = {
    datasetBuff.foreach { dataset =>
      if (dataset.storageLevel != StorageLevel.NONE) {
        dataset.unpersist(blocking)
      }
    }
    datasetBuff.clear()

    rddBuff.foreach { rdd =>
      if (rdd.getStorageLevel != StorageLevel.NONE) {
        rdd.unpersist(blocking)
      }
    }
    rddBuff.clear()

    bcBuff.foreach(_.destroy(blocking))
    bcBuff.clear()
  }
}
