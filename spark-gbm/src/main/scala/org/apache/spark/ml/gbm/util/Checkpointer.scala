package org.apache.spark.ml.gbm.util

import scala.collection.mutable

import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel


/**
  * This class helps with persisting and checkpointing RDDs.
  *
  * Specifically, this abstraction automatically handles persisting and (optionally) checkpointing,
  * as well as unpersisting and removing checkpoint files.
  *
  * Users should call update() when a new Dataset has been created,
  * before the Dataset has been materialized.  After updating [[Checkpointer]], users are
  * responsible for materializing the Dataset to ensure that persisting and checkpointing actually
  * occur.
  *
  * When update() is called, this does the following:
  *  - Persist new Dataset (if not yet persisted), and put in queue of persisted Datasets.
  *  - Unpersist Datasets from queue until there are at most 2 persisted Datasets.
  *  - If using checkpointing and the checkpoint interval has been reached,
  *     - Checkpoint the new Dataset, and put in a queue of checkpointed Datasets.
  *     - Remove older checkpoints.
  *
  * WARNINGS:
  *  - This class should NOT be copied (since copies may conflict on which Datasets should be
  * checkpointed).
  *  - This class removes checkpoint files once later Datasets have been checkpointed.
  * However, references to the older Datasets will still return isCheckpointed = true.
  *
  * @param sc                 SparkContext for the Datasets given to this checkpointer
  * @param checkpointInterval Datasets will be checkpointed at this interval.
  *                           If this interval was set as -1, then checkpointing will be disabled.
  * @param storageLevel       caching storageLevel
  * @tparam T Dataset type, such as Double
  */
private[gbm] class Checkpointer[T](val sc: SparkContext,
                                   val checkpointInterval: Int,
                                   val storageLevel: StorageLevel,
                                   val maxPersisted: Int) extends Logging {
  def this(sc: SparkContext, checkpointInterval: Int, storageLevel: StorageLevel) =
    this(sc, checkpointInterval, storageLevel, 2)

  require(storageLevel != StorageLevel.NONE)
  require(maxPersisted > 1)

  /** FIFO queue of past checkpointed Datasets */
  private val checkpointQueue = mutable.Queue.empty[RDD[T]]

  /** FIFO queue of past persisted Datasets */
  private val persistedQueue = mutable.Queue.empty[RDD[T]]

  /** Number of times [[update()]] has been called */
  private var updateCount = 0

  /**
    * Update with a new Dataset. Handle persistence and checkpointing as needed.
    * Since this handles persistence and checkpointing, this should be called before the Dataset
    * has been materialized.
    *
    * @param data New Dataset created from previous Datasets in the lineage.
    */
  def update(data: RDD[T]): Unit = {
    data.persist(storageLevel)
    persistedQueue.enqueue(data)
    while (persistedQueue.length > maxPersisted) {
      persistedQueue.dequeue.unpersist(false)
    }
    updateCount += 1

    // Handle checkpointing (after persisting)
    if (checkpointInterval != -1 && updateCount % checkpointInterval == 0
      && sc.getCheckpointDir.nonEmpty) {
      // Add new checkpoint before removing old checkpoints.
      data.checkpoint()
      checkpointQueue.enqueue(data)
      // Remove checkpoints before the latest one.
      var canDelete = true
      while (checkpointQueue.length > 1 && canDelete) {
        // Delete the oldest checkpoint only if the next checkpoint exists.
        if (checkpointQueue.head.isCheckpointed) {
          Utils.removeCheckpointFile(checkpointQueue.dequeue, false)
        } else {
          canDelete = false
        }
      }
    }
  }

  def clear(blocking: Boolean = true): Unit = {
    while (persistedQueue.nonEmpty) {
      persistedQueue.dequeue.unpersist(blocking)
    }
    persistedQueue.clear()

    while (checkpointQueue.nonEmpty) {
      Utils.removeCheckpointFile(checkpointQueue.dequeue, false)
    }
    checkpointQueue.clear()

    updateCount = 0
  }
}