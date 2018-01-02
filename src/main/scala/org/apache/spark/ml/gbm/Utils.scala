package org.apache.spark.ml.gbm

import java.{util => ju}

import scala.collection.mutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.reflect.{ClassTag, classTag}
import scala.util.{Failure, Random, Success}

import org.apache.hadoop.fs.Path

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.XORShiftRandom


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
    persist(data)
    persistedQueue.enqueue(data)
    while (persistedQueue.length > maxPersisted) {
      unpersist(persistedQueue.dequeue)
    }
    updateCount += 1

    // Handle checkpointing (after persisting)
    if (checkpointInterval != -1 && (updateCount % checkpointInterval) == 0
      && sc.getCheckpointDir.nonEmpty) {
      // Add new checkpoint before removing old checkpoints.
      checkpoint(data)
      checkpointQueue.enqueue(data)
      // Remove checkpoints before the latest one.
      var canDelete = true
      while (checkpointQueue.length > 1 && canDelete) {
        // Delete the oldest checkpoint only if the next checkpoint exists.
        if (isCheckpointed(checkpointQueue.head)) {
          removeCheckpointFile(checkpointQueue.dequeue)
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

  /** Call this to unpersist the Dataset. */
  def unpersistDataSet(): Unit = {
    while (persistedQueue.nonEmpty) {
      unpersist(persistedQueue.dequeue)
    }
  }

  /** Call this at the end to delete any remaining checkpoint files. */
  def deleteAllCheckpoints(): Unit = {
    while (checkpointQueue.nonEmpty) {
      removeCheckpointFile(checkpointQueue.dequeue)
    }
  }

  /**
    * Dequeue the oldest checkpointed Dataset, and remove its checkpoint files.
    * This prints a warning but does not fail if the files cannot be removed.
    */
  private def removeCheckpointFile(data: RDD[T]): Unit = {
    // Since the old checkpoint is not deleted by Spark, we manually delete it
    data.getCheckpointFile.foreach { file =>
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
    require(fraction >= 0 && fraction <= 1)

    val rng = new Random(seed)

    val numGroups = self.sparkContext.defaultParallelism
    val numPartitions = self.getNumPartitions
    val groupSize = numPartitions / numGroups

    val n = numPartitions * fraction
    val m = n.toInt
    val r = n - m

    val start = rng.nextInt(numPartitions)
    val pids = Array.range(start, numPartitions) ++ Array.range(0, start)

    val weights = mutable.OpenHashMap.empty[Int, Double]
    val remains = mutable.BitSet.empty

    pids.grouped(groupSize).foreach { group =>
      val shuffled = rng.shuffle(group.toSeq).toArray

      val s = (group.length * fraction).floor.toInt

      // select s pids per group
      var i = 0
      while (i < s) {
        weights.update(shuffled(i), 1.0)
        i += 1
      }

      while (i < group.length) {
        remains.add(shuffled(i))
        i += 1
      }
    }

    if (weights.size < m) {
      // select groups
      val gids = rng.shuffle(Seq.range(0, numGroups))
        .take(m - weights.size).toSet

      val buff = mutable.ArrayBuffer.empty[Int]

      pids.grouped(groupSize).zipWithIndex
        .foreach { case (group, gid) =>
          if (gids.contains(gid)) {
            buff.clear()

            group.filter(p => !weights.contains(p))
              .foreach(p => buff.append(p))

            rng.shuffle(buff).headOption
              .foreach { p =>
                weights.update(p, 1.0)
                remains.remove(p)
              }
          }
        }
    }

    if (weights.size < m) {
      rng.shuffle(remains.toSeq).take(m - weights.size)
        .foreach { p =>
          weights.update(p, 1.0)
          remains.remove(p)
        }
    }

    if (r > 0) {
      rng.shuffle(remains.toSeq).headOption
        .foreach { p =>
          weights.update(p, r)
          remains.remove(p)
        }
    }

    samplePartitions(weights.toMap, seed)
  }
}


private[gbm] object RDDFunctions {

  /** Implicit conversion from an RDD to RDDFunctions. */
  implicit def fromRDD[T: ClassTag](rdd: RDD[T]): RDDFunctions[T] = new RDDFunctions[T](rdd)
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

  override def getPartition(key: Any): Int = by(key) match {
    case null => 0
    case k => search(splits, k)
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


