package org.apache.spark.ml.gbm

import java.{util => ju}

import scala.collection.mutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.util.{Failure, Random, Success}

import org.apache.hadoop.fs.Path

import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.unsafe.hash.Murmur3_x86_32


trait ColumSelector extends Serializable {
  def contains[T, C](treeId: T, colId: C)
                    (implicit int: Integral[T], inc: Integral[C]): Boolean
}


object ColumSelector extends Serializable {

  def create(colSampleRate: Double,
             numCols: Int,
             numBaseModels: Int,
             rawSize: Int,
             seed: Long): ColumSelector = {

    if (colSampleRate == 1) {
      TotalSelector()

    } else if (numCols * colSampleRate > 32) {
      val rng = new Random(seed)
      val maximum = (Int.MaxValue * colSampleRate).ceil.toInt

      val seeds = Array.range(0, numBaseModels).flatMap { i =>
        val s = rng.nextInt
        Iterator.fill(rawSize)(s)
      }

      HashSelector(maximum, seeds)

    } else {
      val rng = new Random(seed)
      val numSelected = (numCols * colSampleRate).ceil.toInt

      val sets = Array.range(0, numBaseModels).flatMap { i =>
        val selected = rng.shuffle(Seq.range(0, numCols)).take(numSelected).toArray.sorted
        Iterator.fill(rawSize)(selected)
      }

      SetSelector(sets)
    }
  }

  def union(selectors: Array[ColumSelector]): ColumSelector = {
    UnionSelector(selectors)
  }
}


case class TotalSelector() extends ColumSelector {

  override def contains[T, C](treeId: T, colId: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = true

  override def toString: String = "TotalSelector"
}


case class HashSelector(maximum: Int,
                        seeds: Array[Int]) extends ColumSelector {
  require(maximum >= 0)

  override def contains[T, C](treeId: T, colId: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    Murmur3_x86_32.hashLong(inc.toLong(colId), seeds(int.toInt(treeId))).abs < maximum
  }

  override def toString: String = s"HashSelector(maximum: $maximum, seed: ${seeds.mkString("[", ",", "]")})"
}


case class SetSelector(sets: Array[Array[Int]]) extends ColumSelector {
  require(sets.nonEmpty && sets.forall(_.nonEmpty))

  override def contains[T, C](treeId: T, colId: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    ju.Arrays.binarySearch(sets(int.toInt(treeId)), inc.toInt(colId)) >= 0
  }

  override def toString: String = s"SetSelector(sets: ${sets.mkString("[", ",", "]")})"
}


case class UnionSelector(selectors: Array[ColumSelector]) extends ColumSelector {
  override def contains[T, C](treeId: T, colId: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    selectors.forall(_.contains[T, C](treeId, colId))
  }

  override def toString: String = s"UnionSelector(selectors: ${selectors.mkString("[", ",", "]")})"
}


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
          removeCheckpointFile(checkpointQueue.dequeue)
        } else {
          canDelete = false
        }
      }
    }
  }

  def last: RDD[T] = persistedQueue.last

  def lastOption: Option[RDD[T]] = persistedQueue.lastOption

  def clear(): Unit = {
    while (persistedQueue.nonEmpty) {
      persistedQueue.dequeue.unpersist(false)
    }
    persistedQueue.clear()

    while (checkpointQueue.nonEmpty) {
      removeCheckpointFile(checkpointQueue.dequeue)
    }
    checkpointQueue.clear()

    updateCount = 0
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
          logInfo(s"Successfully remove old checkpoint file: $file, duration $v seconds")

        case Failure(t) =>
          logWarning(s"Fail to remove old checkpoint file: $file, ${t.toString}")
      }
    }
  }
}


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

  def clear(): Unit = {
    datasetBuff.foreach { dataset =>
      if (dataset.storageLevel != StorageLevel.NONE) {
        dataset.unpersist(false)
      }
    }
    datasetBuff.clear()

    rddBuff.foreach { rdd =>
      if (rdd.getStorageLevel != StorageLevel.NONE) {
        rdd.unpersist(false)
      }
    }
    rddBuff.clear()

    bcBuff.foreach(_.destroy(false))
    bcBuff.clear()
  }
}


object Utils extends Logging {

  def getTypeByRange(value: Int): String = {
    if (value <= Byte.MaxValue) {
      "Byte"
    } else if (value <= Short.MaxValue) {
      "Short"
    } else {
      "Int"
    }
  }


  def outerJoinSortedIters[K, V1, V2](iter1: Iterator[(K, V1)],
                                      iter2: Iterator[(K, V2)])
                                     (implicit ork: Ordering[K]): Iterator[(K, Option[V1], Option[V2])] = new Iterator[(K, Option[V1], Option[V2])]() {

    var kv1 = Option.empty[(K, V1)]
    var kv2 = Option.empty[(K, V2)]

    {
      updateKV1()
      updateKV2()
    }

    def updateKV1(): Unit = {
      kv1 = if (iter1.hasNext) {
        Some(iter1.next)
      } else {
        None
      }
    }


    def updateKV2(): Unit = {
      kv2 = if (iter2.hasNext) {
        Some(iter2.next)
      } else {
        None
      }
    }

    override def hasNext: Boolean = {
      kv1.nonEmpty || kv2.nonEmpty
    }

    override def next(): (K, Option[V1], Option[V2]) = (kv1, kv2) match {
      case (Some((k1, v1)), Some((k2, v2))) =>
        val cmp = ork.compare(k1, k2)

        if (cmp == 0) {
          val t = (k1, Some(v1), Some(v2))
          updateKV1()
          updateKV2()
          t

        } else if (cmp < 0) {
          val t = (k1, Some(v1), None)
          updateKV1()
          t

        } else {
          val t = (k2, None, Some(v2))
          updateKV2()
          t
        }

      case (Some((k1, v1)), None) =>
        val t = (k1, Some(v1), None)
        updateKV1()
        t

      case (None, Some((k2, v2))) =>
        val t = (k2, None, Some(v2))
        updateKV2()
        t
    }

  }


  def innerJoinSortedIters[K, V1, V2](iter1: Iterator[(K, V1)],
                                      iter2: Iterator[(K, V2)])
                                     (implicit ork: Ordering[K]): Iterator[(K, V1, V2)] = new Iterator[(K, V1, V2)]() {

    var kv1 = Option.empty[(K, V1)]
    var kv2 = Option.empty[(K, V2)]
    var cmp = 0

    {
      updateKV()
    }

    def updateKV1(): Unit = {
      cmp = -1
      while (iter1.hasNext && cmp < 0) {
        kv1 = Some(iter1.next)
        cmp = ork.compare(kv1.get._1, kv2.get._1)
      }

      if (cmp < 0) {
        kv1 = None
      }
    }


    def updateKV2(): Unit = {
      cmp = 1
      while (iter2.hasNext && cmp > 0) {
        kv2 = Some(iter2.next)
        cmp = ork.compare(kv1.get._1, kv2.get._1)
      }

      if (cmp > 0) {
        kv2 = None
      }
    }


    def updateKV(): Unit = {
      kv1 = if (iter1.hasNext) {
        Some(iter1.next)
      } else {
        None
      }

      kv2 = if (iter2.hasNext) {
        Some(iter2.next)
      } else {
        None
      }

      if (kv1.nonEmpty && kv2.nonEmpty) {
        cmp = ork.compare(kv1.get._1, kv2.get._1)
        while (cmp != 0 && kv1.nonEmpty && kv2.nonEmpty) {
          if (cmp > 0) {
            updateKV2()
          } else if (cmp < 0) {
            updateKV1()
          }
        }
      }
    }

    override def hasNext: Boolean = {
      kv1.nonEmpty && kv2.nonEmpty
    }

    override def next(): (K, V1, V2) = {
      val t = (kv1.get._1, kv1.get._2, kv2.get._2)
      updateKV()
      t
    }
  }


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


  private[this] var kryoRegistered: Boolean = false

  def registerKryoClasses(sc: SparkContext): Unit = {
    if (!kryoRegistered) {
      sc.getConf.registerKryoClasses(Array(

        classOf[GBM],
        classOf[GBMModel],
        classOf[TreeModel],

        classOf[BoostConfig],
        classOf[BaseConfig],

        classOf[ColDiscretizer],
        classOf[Array[ColDiscretizer]],
        classOf[QuantileNumColDiscretizer],
        classOf[IntervalNumColDiscretizer],
        classOf[CatColDiscretizer],
        classOf[RankColDiscretizer],

        classOf[ColAgg],
        classOf[Array[ColAgg]],
        classOf[QuantileNumColAgg],
        classOf[IntervalNumColAgg],
        classOf[CatColAgg],
        classOf[RankColAgg],

        classOf[Split],
        classOf[SeqSplit],
        classOf[SetSplit],

        classOf[LearningNode],
        classOf[Node],
        classOf[InternalNode],
        classOf[LeafNode],
        classOf[NodeData],

        classOf[ColumSelector],
        classOf[TotalSelector],
        classOf[HashSelector],
        classOf[SetSelector],

        classOf[ObjFunc],
        classOf[ScalarObjFunc],
        classOf[SquareObj],
        classOf[LogisticObj],
        classOf[SoftmaxObj],

        classOf[EvalFunc],
        classOf[ScalarEvalFunc],
        classOf[IncEvalFunc],
        classOf[ScalarIncEvalFunc],
        classOf[SimpleEvalFunc],
        classOf[MSEEval],
        classOf[RMSEEval],
        classOf[MAEEval],
        classOf[R2Eval],
        classOf[LogLossEval],
        classOf[ErrorEval],
        classOf[AUCEval],

        classOf[CallbackFunc],
        classOf[EarlyStop],
        classOf[MetricRecoder],
        classOf[ModelCheckpoint],
        classOf[ClassificationModelCheckpoint],
        classOf[RegressionModelCheckpoint],

        classOf[KVVector[Any, Any]],
        classOf[KVVector[Byte, Byte]],
        classOf[KVVector[Byte, Short]],
        classOf[KVVector[Byte, Int]],
        classOf[KVVector[Byte, Float]],
        classOf[KVVector[Byte, Double]],
        classOf[KVVector[Short, Byte]],
        classOf[KVVector[Short, Short]],
        classOf[KVVector[Short, Int]],
        classOf[KVVector[Short, Float]],
        classOf[KVVector[Short, Double]],
        classOf[KVVector[Int, Byte]],
        classOf[KVVector[Int, Short]],
        classOf[KVVector[Int, Int]],
        classOf[KVVector[Int, Float]],
        classOf[KVVector[Int, Double]],

        classOf[DenseKVVector[Any, Any]],
        classOf[DenseKVVector[Byte, Byte]],
        classOf[DenseKVVector[Byte, Short]],
        classOf[DenseKVVector[Byte, Int]],
        classOf[DenseKVVector[Byte, Float]],
        classOf[DenseKVVector[Byte, Double]],
        classOf[DenseKVVector[Short, Byte]],
        classOf[DenseKVVector[Short, Short]],
        classOf[DenseKVVector[Short, Int]],
        classOf[DenseKVVector[Short, Float]],
        classOf[DenseKVVector[Short, Double]],
        classOf[DenseKVVector[Int, Byte]],
        classOf[DenseKVVector[Int, Short]],
        classOf[DenseKVVector[Int, Int]],
        classOf[DenseKVVector[Int, Float]],
        classOf[DenseKVVector[Int, Double]],

        classOf[SparseKVVector[Any, Any]],
        classOf[SparseKVVector[Byte, Byte]],
        classOf[SparseKVVector[Byte, Short]],
        classOf[SparseKVVector[Byte, Int]],
        classOf[SparseKVVector[Byte, Float]],
        classOf[SparseKVVector[Byte, Double]],
        classOf[SparseKVVector[Short, Byte]],
        classOf[SparseKVVector[Short, Short]],
        classOf[SparseKVVector[Short, Int]],
        classOf[SparseKVVector[Short, Float]],
        classOf[SparseKVVector[Short, Double]],
        classOf[SparseKVVector[Int, Byte]],
        classOf[SparseKVVector[Int, Short]],
        classOf[SparseKVVector[Int, Int]],
        classOf[SparseKVVector[Int, Float]],
        classOf[SparseKVVector[Int, Double]],

        classOf[ArrayBlock[Any]],
        classOf[ArrayBlock[Byte]],
        classOf[ArrayBlock[Short]],
        classOf[ArrayBlock[Int]],
        classOf[ArrayBlock[Long]],
        classOf[ArrayBlock[Float]],
        classOf[ArrayBlock[Double]],

        classOf[KVMatrix[Any, Any]],
        classOf[KVMatrix[Byte, Byte]],
        classOf[KVMatrix[Byte, Short]],
        classOf[KVMatrix[Byte, Int]],
        classOf[KVMatrix[Short, Byte]],
        classOf[KVMatrix[Short, Short]],
        classOf[KVMatrix[Short, Int]],
        classOf[KVMatrix[Int, Byte]],
        classOf[KVMatrix[Int, Short]],
        classOf[KVMatrix[Int, Int]],

        classOf[InstanceBlock[Any, Any, Any]],
        classOf[InstanceBlock[Byte, Byte, Float]],
        classOf[InstanceBlock[Byte, Byte, Double]],
        classOf[InstanceBlock[Byte, Short, Float]],
        classOf[InstanceBlock[Byte, Short, Double]],
        classOf[InstanceBlock[Byte, Int, Float]],
        classOf[InstanceBlock[Byte, Int, Double]],
        classOf[InstanceBlock[Short, Byte, Float]],
        classOf[InstanceBlock[Short, Byte, Double]],
        classOf[InstanceBlock[Short, Short, Float]],
        classOf[InstanceBlock[Short, Short, Double]],
        classOf[InstanceBlock[Short, Int, Float]],
        classOf[InstanceBlock[Short, Int, Double]],
        classOf[InstanceBlock[Int, Byte, Float]],
        classOf[InstanceBlock[Int, Byte, Double]],
        classOf[InstanceBlock[Int, Short, Float]],
        classOf[InstanceBlock[Int, Short, Double]],
        classOf[InstanceBlock[Int, Int, Float]],
        classOf[InstanceBlock[Int, Int, Double]],

        classOf[SkipNodePratitioner[Any, Any, Any]],
        classOf[SkipNodePratitioner[Byte, Byte, Byte]],
        classOf[SkipNodePratitioner[Byte, Byte, Short]],
        classOf[SkipNodePratitioner[Byte, Byte, Int]],
        classOf[SkipNodePratitioner[Byte, Short, Byte]],
        classOf[SkipNodePratitioner[Byte, Short, Short]],
        classOf[SkipNodePratitioner[Byte, Short, Int]],
        classOf[SkipNodePratitioner[Byte, Int, Byte]],
        classOf[SkipNodePratitioner[Byte, Int, Short]],
        classOf[SkipNodePratitioner[Byte, Int, Int]],
        classOf[SkipNodePratitioner[Short, Byte, Byte]],
        classOf[SkipNodePratitioner[Short, Byte, Short]],
        classOf[SkipNodePratitioner[Short, Byte, Int]],
        classOf[SkipNodePratitioner[Short, Short, Byte]],
        classOf[SkipNodePratitioner[Short, Short, Short]],
        classOf[SkipNodePratitioner[Short, Short, Int]],
        classOf[SkipNodePratitioner[Short, Int, Byte]],
        classOf[SkipNodePratitioner[Short, Int, Short]],
        classOf[SkipNodePratitioner[Short, Int, Int]],
        classOf[SkipNodePratitioner[Int, Byte, Byte]],
        classOf[SkipNodePratitioner[Int, Byte, Short]],
        classOf[SkipNodePratitioner[Int, Byte, Int]],
        classOf[SkipNodePratitioner[Int, Short, Byte]],
        classOf[SkipNodePratitioner[Int, Short, Short]],
        classOf[SkipNodePratitioner[Int, Short, Int]],
        classOf[SkipNodePratitioner[Int, Int, Byte]],
        classOf[SkipNodePratitioner[Int, Int, Short]],
        classOf[SkipNodePratitioner[Int, Int, Int]],

        classOf[DepthPratitioner[Any, Any, Any]],
        classOf[DepthPratitioner[Byte, Byte, Byte]],
        classOf[DepthPratitioner[Byte, Byte, Short]],
        classOf[DepthPratitioner[Byte, Byte, Int]],
        classOf[DepthPratitioner[Byte, Short, Byte]],
        classOf[DepthPratitioner[Byte, Short, Short]],
        classOf[DepthPratitioner[Byte, Short, Int]],
        classOf[DepthPratitioner[Byte, Int, Byte]],
        classOf[DepthPratitioner[Byte, Int, Short]],
        classOf[DepthPratitioner[Byte, Int, Int]],
        classOf[DepthPratitioner[Short, Byte, Byte]],
        classOf[DepthPratitioner[Short, Byte, Short]],
        classOf[DepthPratitioner[Short, Byte, Int]],
        classOf[DepthPratitioner[Short, Short, Byte]],
        classOf[DepthPratitioner[Short, Short, Short]],
        classOf[DepthPratitioner[Short, Short, Int]],
        classOf[DepthPratitioner[Short, Int, Byte]],
        classOf[DepthPratitioner[Short, Int, Short]],
        classOf[DepthPratitioner[Short, Int, Int]],
        classOf[DepthPratitioner[Int, Byte, Byte]],
        classOf[DepthPratitioner[Int, Byte, Short]],
        classOf[DepthPratitioner[Int, Byte, Int]],
        classOf[DepthPratitioner[Int, Short, Byte]],
        classOf[DepthPratitioner[Int, Short, Short]],
        classOf[DepthPratitioner[Int, Short, Int]],
        classOf[DepthPratitioner[Int, Int, Byte]],
        classOf[DepthPratitioner[Int, Int, Short]],
        classOf[DepthPratitioner[Int, Int, Int]],


        classOf[IDRangePratitioner[Any, Any, Any]],
        classOf[IDRangePratitioner[Byte, Byte, Byte]],
        classOf[IDRangePratitioner[Byte, Byte, Short]],
        classOf[IDRangePratitioner[Byte, Byte, Int]],
        classOf[IDRangePratitioner[Byte, Short, Byte]],
        classOf[IDRangePratitioner[Byte, Short, Short]],
        classOf[IDRangePratitioner[Byte, Short, Int]],
        classOf[IDRangePratitioner[Byte, Int, Byte]],
        classOf[IDRangePratitioner[Byte, Int, Short]],
        classOf[IDRangePratitioner[Byte, Int, Int]],
        classOf[IDRangePratitioner[Short, Byte, Byte]],
        classOf[IDRangePratitioner[Short, Byte, Short]],
        classOf[IDRangePratitioner[Short, Byte, Int]],
        classOf[IDRangePratitioner[Short, Short, Byte]],
        classOf[IDRangePratitioner[Short, Short, Short]],
        classOf[IDRangePratitioner[Short, Short, Int]],
        classOf[IDRangePratitioner[Short, Int, Byte]],
        classOf[IDRangePratitioner[Short, Int, Short]],
        classOf[IDRangePratitioner[Short, Int, Int]],
        classOf[IDRangePratitioner[Int, Byte, Byte]],
        classOf[IDRangePratitioner[Int, Byte, Short]],
        classOf[IDRangePratitioner[Int, Byte, Int]],
        classOf[IDRangePratitioner[Int, Short, Byte]],
        classOf[IDRangePratitioner[Int, Short, Short]],
        classOf[IDRangePratitioner[Int, Short, Int]],
        classOf[IDRangePratitioner[Int, Int, Byte]],
        classOf[IDRangePratitioner[Int, Int, Short]],
        classOf[IDRangePratitioner[Int, Int, Int]]))

      kryoRegistered = true
    }
  }
}


