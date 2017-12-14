package org.apache.spark.ml.gbm

import java.{util => ju}

import scala.collection.mutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.reflect.{ClassTag, classTag}
import scala.util.{Failure, Random, Success}

import org.apache.hadoop.fs.Path

import org.apache.spark.Partitioner
import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
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
                                   val storageLevel: StorageLevel) extends Logging {
  require(storageLevel != StorageLevel.NONE)

  /** FIFO queue of past checkpointed Datasets */
  private val checkpointQueue = mutable.Queue[RDD[T]]()

  /** FIFO queue of past persisted Datasets */
  private val persistedQueue = mutable.Queue[RDD[T]]()

  /** Number of times [[update()]] has been called */
  private var updateCount = 0

  /**
    * Update with a new Dataset. Handle persistence and checkpointing as needed.
    * Since this handles persistence and checkpointing, this should be called before the Dataset
    * has been materialized.
    *
    * @param newData New Dataset created from previous Datasets in the lineage.
    */
  def update(newData: RDD[T]): Unit = {
    persist(newData)
    persistedQueue.enqueue(newData)
    while (persistedQueue.length > 2) {
      val dataToUnpersist = persistedQueue.dequeue()
      unpersist(dataToUnpersist)
    }
    updateCount += 1

    // Handle checkpointing (after persisting)
    if (checkpointInterval != -1 && (updateCount % checkpointInterval) == 0
      && sc.getCheckpointDir.nonEmpty) {
      // Add new checkpoint before removing old checkpoints.
      checkpoint(newData)
      checkpointQueue.enqueue(newData)
      // Remove checkpoints before the latest one.
      var canDelete = true
      while (checkpointQueue.length > 1 && canDelete) {
        // Delete the oldest checkpoint only if the next checkpoint exists.
        if (isCheckpointed(checkpointQueue.head)) {
          removeCheckpointFile()
        } else {
          canDelete = false
        }
      }
    }
  }

  /** Checkpoint the Dataset */
  protected def checkpoint(data: RDD[T]): Unit = {
    data.checkpoint()
  }

  /** Return true iff the Dataset is checkpointed */
  protected def isCheckpointed(data: RDD[T]): Boolean = {
    data.isCheckpointed
  }

  /**
    * Persist the Dataset.
    * Note: This should handle checking the current [[StorageLevel]] of the Dataset.
    */
  protected def persist(data: RDD[T]): Unit = {
    if (data.getStorageLevel == StorageLevel.NONE) {
      data.persist(storageLevel)
    }
  }

  /** Unpersist the Dataset */
  protected def unpersist(data: RDD[T]): Unit = {
    data.unpersist(blocking = false)
  }

  /** Get list of checkpoint files for this given Dataset */
  protected def getCheckpointFiles(data: RDD[T]): Iterable[String] = {
    data.getCheckpointFile.map(x => x)
  }

  /**
    * Call this to unpersist the Dataset.
    */
  def unpersistDataSet(): Unit = {
    while (persistedQueue.nonEmpty) {
      val dataToUnpersist = persistedQueue.dequeue()
      unpersist(dataToUnpersist)
    }
  }

  /**
    * Call this at the end to delete any remaining checkpoint files.
    */
  def deleteAllCheckpoints(): Unit = {
    while (checkpointQueue.nonEmpty) {
      removeCheckpointFile()
    }
  }

  /**
    * Call this at the end to delete any remaining checkpoint files, except for the last checkpoint.
    * Note that there may not be any checkpoints at all.
    */
  def deleteAllCheckpointsButLast(): Unit = {
    while (checkpointQueue.length > 1) {
      removeCheckpointFile()
    }
  }

  /**
    * Get all current checkpoint files.
    * This is useful in combination with [[deleteAllCheckpointsButLast()]].
    */
  def getAllCheckpointFiles: Array[String] = {
    checkpointQueue.flatMap(getCheckpointFiles).toArray
  }

  /**
    * Dequeue the oldest checkpointed Dataset, and remove its checkpoint files.
    * This prints a warning but does not fail if the files cannot be removed.
    */
  private def removeCheckpointFile(): Unit = {
    val old = checkpointQueue.dequeue()

    // Since the old checkpoint is not deleted by Spark, we manually delete it
    getCheckpointFiles(old).foreach { file =>
      Future {
        val start = System.nanoTime
        val path = new Path(file)
        val fs = path.getFileSystem(sc.hadoopConfiguration)
        fs.delete(path, true)
        (System.nanoTime - start) / 1e9

      }.onComplete {
        case Success(v) =>
          logInfo(s"successfully remove old checkpoint file: $file, duration $v seconds")

        case Failure(t) =>
          logWarning(s"fail to remove old checkpoint file: $file, ${t.toString}")
      }
    }
  }
}


/**
  * Partitioner that partitions data according to the given ranges
  *
  * @param splits splitting points
  * @tparam K
  */
private[gbm] class GBMRangePartitioner[K: Ordering : ClassTag](val splits: Array[K],
                                                               val by: Any => K = (a: Any) => a.asInstanceOf[K]) extends Partitioner {

  {
    require(splits.nonEmpty)
    val ordering = implicitly[Ordering[K]]
    var i = 0
    while (i < splits.length - 1) {
      require(ordering.lt(splits(i), splits(i + 1)))
      i += 1
    }
  }

  private val binarySearch = Utils.makeBinarySearch[K]

  private val search = {
    if (splits.length <= 128) {
      val ordering = implicitly[Ordering[K]]
      (array: Array[K], k: K) =>
        var i = 0
        while (i < array.length && ordering.gt(k, array(i))) {
          i += 1
        }
        i
    } else {
      (array: Array[K], k: K) =>
        var i = binarySearch(array, k)
        if (i < 0) {
          i = -i - 1
        }
        math.min(i, splits.length)
    }
  }

  override def numPartitions: Int = splits.length + 1

  override def getPartition(key: Any): Int = {
    search(splits, by(key))
  }
}


/**
  * Partitioner that partitions data according to the given key function
  */
private[gbm] class GBMHashPartitioner(val partitions: Int,
                                      val by: Any => Any = (a: Any) => a) extends Partitioner {
  require(partitions > 0)

  override def numPartitions: Int = partitions

  override def getPartition(key: Any): Int = by(key) match {
    case null => 0
    case k =>
      val p = k.hashCode % partitions
      if (p < 0) {
        p + partitions
      } else {
        p
      }
  }
}


private[gbm] object Utils extends Logging {

  /**
    * traverse all the elements of a vector
    */
  def getTotalIter(vec: Vector): Iterator[(Int, Double)] = {
    vec match {
      case dv: DenseVector =>
        Iterator.range(0, dv.size)
          .map(i => (i, dv.values(i)))

      case sv: SparseVector =>
        new Iterator[(Int, Double)]() {
          private var i = 0
          private var j = 0

          override def hasNext: Boolean = i < sv.size

          override def next(): (Int, Double) = {
            val v = if (j == sv.indices.length) {
              0.0
            } else {
              val k = sv.indices(j)
              if (i == k) {
                j += 1
                sv.values(j - 1)
              } else {
                0.0
              }
            }
            i += 1
            (i - 1, v)
          }
        }
    }
  }


  /**
    * traverse only the non-zero elements of a vector
    */
  def getActiveIter(vec: Vector): Iterator[(Int, Double)] = {
    vec match {
      case dv: DenseVector =>
        Iterator.range(0, dv.size)
          .map(i => (i, dv.values(i)))
          .filter(t => t._2 != 0)

      case sv: SparseVector =>
        Iterator.range(0, sv.indices.length)
          .map(i => (sv.indices(i), sv.values(i)))
          .filter(t => t._2 != 0)
    }
  }


  /**
    * data sampling, if number of partitions is large, directly sample the partitions is more efficient
    */
  def sample[T: ClassTag](data: RDD[T],
                          fraction: Double,
                          seed: Long): RDD[T] = {

    if (fraction == 1.0) {
      data

    } else {

      val numPartitions = data.getNumPartitions
      val parallelism = data.sparkContext.defaultParallelism

      if (numPartitions * fraction < parallelism * 10) {
        data.sample(false, fraction, seed)

      } else {

        logInfo(s"Using partition-based sampling")

        val rng = new Random(seed)
        val shuffled = rng.shuffle(Seq.range(0, numPartitions)).toArray

        val n = numPartitions * fraction
        val m = n.toInt
        val r = n - m
        val eps = 1e-2

        val (selected, partial) = if (m > 0 && r < eps) {
          (shuffled.take(m).sorted, -1)
        } else if (m >= 0 && r > 1 - eps) {
          (shuffled.take(m + 1).sorted, -1)
        } else {
          (shuffled.take(m).sorted, shuffled.last)
        }

        data.mapPartitionsWithIndex { case (pid, it) =>
          if (ju.Arrays.binarySearch(selected, pid) >= 0) {
            it
          } else if (pid == partial) {
            val rng = new Random(seed + pid)
            it.filter(_ => rng.nextDouble() < r)
          } else {
            Iterator.empty
          }
        }
      }
    }
  }


  /**
    * helper function to obtain a binary search function
    */
  def makeBinarySearch[K: Ordering : ClassTag]: (Array[K], K) => Int = {
    // For primitive keys, we can use the natural ordering. Otherwise, use the Ordering comparator.
    classTag[K] match {
      case ClassTag.Float =>
        (l, x) => ju.Arrays.binarySearch(l.asInstanceOf[Array[Float]], x.asInstanceOf[Float])
      case ClassTag.Double =>
        (l, x) => ju.Arrays.binarySearch(l.asInstanceOf[Array[Double]], x.asInstanceOf[Double])
      case ClassTag.Byte =>
        (l, x) => ju.Arrays.binarySearch(l.asInstanceOf[Array[Byte]], x.asInstanceOf[Byte])
      case ClassTag.Char =>
        (l, x) => ju.Arrays.binarySearch(l.asInstanceOf[Array[Char]], x.asInstanceOf[Char])
      case ClassTag.Short =>
        (l, x) => ju.Arrays.binarySearch(l.asInstanceOf[Array[Short]], x.asInstanceOf[Short])
      case ClassTag.Int =>
        (l, x) => ju.Arrays.binarySearch(l.asInstanceOf[Array[Int]], x.asInstanceOf[Int])
      case ClassTag.Long =>
        (l, x) => ju.Arrays.binarySearch(l.asInstanceOf[Array[Long]], x.asInstanceOf[Long])
      case _ =>
        val comparator = implicitly[Ordering[K]].asInstanceOf[java.util.Comparator[Any]]
        (l, x) => ju.Arrays.binarySearch(l.asInstanceOf[Array[AnyRef]], x, comparator)
    }
  }


  /**
    * helper function to save dataframes
    */
  def saveDataFrames(dataframes: Array[DataFrame],
                     names: Array[String],
                     path: String): Unit = {
    require(dataframes.length == names.length)
    require(names.length == names.distinct.length)
    dataframes.zip(names).foreach { case (df, name) =>
      df.write.parquet(new Path(path, name).toString)
    }
  }


  /**
    * helper function to load dataframes
    */
  def loadDataFrames(spark: SparkSession,
                     names: Array[String],
                     path: String): Array[DataFrame] = {
    names.map { name =>
      spark.read.parquet(new Path(path, name).toString)
    }
  }
}


