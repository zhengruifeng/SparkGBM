package org.apache.spark.ml.gbm.util

import java.io._

import scala.collection._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.util.{Failure, Success}

import org.apache.hadoop.fs.Path

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.gbm.func._
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.impl._
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._


private[gbm] object Utils extends Logging {

  val BYTE = "Byte"
  val SHORT = "Short"
  val INT = "Int"
  val LONG = "Long"


  /**
    * deep copy an object
    */
  def deepCopy[T](data: T): T = {
    val bos = new ByteArrayOutputStream
    val oos = new ObjectOutputStream(bos)

    oos.writeObject(data)
    oos.flush()
    oos.close()
    bos.close()

    val objBytes = bos.toByteArray

    val bis = new ByteArrayInputStream(objBytes)
    val ois = new ObjectInputStream(bis)
    ois.readObject().asInstanceOf[T]
  }


  /**
    * Choose the most compact Integer Type according to the range
    */
  def getTypeByRange(range: Long): String = {
    require(range >= 0)

    if (range <= Byte.MaxValue) {
      BYTE
    } else if (range <= Short.MaxValue) {
      SHORT
    } else if (range <= Int.MaxValue) {
      INT
    } else {
      LONG
    }
  }


  /**
    * An iterator producing equally spaced values in some integer interval.
    */
  def range(start: Long, end: Long, step: Long): Iterator[Long] = {
    require(step != 0)

    new Iterator[Long] {
      private var i = start

      def hasNext: Boolean = (step <= 0 || i < end) && (step >= 0 || i > end)

      def next(): Long =
        if (hasNext) {
          val result = i
          i += step
          result
        } else {
          Iterator.empty.next()
        }
    }
  }


  /**
    * zip two iterators
    */
  def zip2[V1, V2](iter1: Iterator[V1],
                   iter2: Iterator[V2],
                   validate: Boolean = true) = {

    if (validate) {
      new Iterator[(V1, V2)] {
        def hasNext: Boolean = (iter1.hasNext, iter2.hasNext) match {

          case (true, true) => true

          case (false, false) => false

          case t => throw new Exception(s"Input iterators have different lengths: $t")
        }

        def next(): (V1, V2) = (iter1.next(), iter2.next())
      }

    } else {
      new Iterator[(V1, V2)] {
        def hasNext: Boolean = iter1.hasNext && iter2.hasNext

        def next: (V1, V2) = (iter1.next(), iter2.next())
      }
    }
  }


  /**
    * zip three iterators
    */
  def zip3[V1, V2, V3](iter1: Iterator[V1],
                       iter2: Iterator[V2],
                       iter3: Iterator[V3],
                       validate: Boolean = true) = {

    if (validate) {
      new Iterator[(V1, V2, V3)] {
        def hasNext: Boolean = (iter1.hasNext, iter2.hasNext, iter3.hasNext) match {

          case (true, true, true) => true

          case (false, false, false) => false

          case t => throw new Exception(s"Input iterators have different lengths: $t")
        }

        def next(): (V1, V2, V3) = (iter1.next(), iter2.next(), iter3.next())
      }

    } else {
      new Iterator[(V1, V2, V3)] {
        def hasNext: Boolean = iter1.hasNext && iter2.hasNext && iter3.hasNext

        def next: (V1, V2, V3) = (iter1.next(), iter2.next(), iter3.next())
      }
    }
  }


  /**
    * zip four iterators
    */
  def zip4[V1, V2, V3, V4](iter1: Iterator[V1],
                           iter2: Iterator[V2],
                           iter3: Iterator[V3],
                           iter4: Iterator[V4],
                           validate: Boolean = true) = {

    if (validate) {
      new Iterator[(V1, V2, V3, V4)] {
        def hasNext: Boolean = (iter1.hasNext, iter2.hasNext, iter3.hasNext, iter4.hasNext) match {

          case (true, true, true, true) => true

          case (false, false, false, false) => false

          case t => throw new Exception(s"Input iterators have different lengths: $t")
        }

        def next(): (V1, V2, V3, V4) = (iter1.next(), iter2.next(), iter3.next(), iter4.next())
      }

    } else {
      new Iterator[(V1, V2, V3, V4)] {
        def hasNext: Boolean = iter1.hasNext && iter2.hasNext && iter3.hasNext && iter4.hasNext

        def next(): (V1, V2, V3, V4) = (iter1.next(), iter2.next(), iter3.next(), iter4.next())
      }
    }
  }


  /**
    * Validate the ordering, and return the same iterator
    */
  def validateOrdering[K](iterator: Iterator[K],
                          ascending: Boolean = true,
                          strictly: Boolean = true)
                         (implicit ork: Ordering[K]) = new Iterator[K]() {
    private var k = Option.empty[K]

    private val check = (ascending, strictly) match {

      case (true, true) => (i: K, j: K) =>
        require(ork.compare(i, j) < 0, s"($i, $j) breaks strictly ascending")

      case (true, false) => (i: K, j: K) =>
        require(ork.compare(i, j) <= 0, s"($i, $j) breaks ascending (non-descending)")

      case (false, true) => (i: K, j: K) =>
        require(ork.compare(i, j) > 0, s"($i, $j) breaks strictly descending")

      case (false, false) => (i: K, j: K) =>
        require(ork.compare(i, j) >= 0, s"($i, $j) breaks descending (non-ascending)")
    }

    {
      update()
    }

    private def update(): Unit = {
      if (iterator.hasNext) {
        val t = iterator.next
        if (k.nonEmpty) {
          check(k.get, t)
        }
        k = Some(t)
      } else {
        k = None
      }
    }

    override def hasNext: Boolean = {
      k.nonEmpty
    }

    override def next(): K = {
      val ret = k.get
      update()
      ret
    }
  }


  /**
    * Validate the ordering of keys, and return the same iterator
    */
  def validateKeyOrdering[K, V](iterator: Iterator[(K, V)],
                                ascending: Boolean = true,
                                strictly: Boolean = true)
                               (implicit ork: Ordering[K]) = new Iterator[(K, V)]() {
    private var kv = Option.empty[(K, V)]

    private val check = (ascending, strictly) match {

      case (true, true) => (i: K, j: K) =>
        require(ork.compare(i, j) < 0, s"($i, $j) breaks strictly ascending")

      case (true, false) => (i: K, j: K) =>
        require(ork.compare(i, j) <= 0, s"($i, $j) breaks ascending (non-descending)")

      case (false, true) => (i: K, j: K) =>
        require(ork.compare(i, j) > 0, s"($i, $j) breaks strictly descending")

      case (false, false) => (i: K, j: K) =>
        require(ork.compare(i, j) >= 0, s"($i, $j) breaks descending (non-ascending)")
    }

    {
      update()
    }

    private def update(): Unit = {
      if (iterator.hasNext) {
        val t = iterator.next
        if (kv.nonEmpty) {
          check(kv.get._1, t._1)
        }
        kv = Some(t)
      } else {
        kv = None
      }
    }

    override def hasNext: Boolean = {
      kv.nonEmpty
    }

    override def next(): (K, V) = {
      val ret = kv.get
      update()
      ret
    }
  }


  /**
    * Outer join of two Strictly Ascending iterators
    */
  def outerJoinSortedIters[K, V1, V2](iterator1: Iterator[(K, V1)],
                                      iterator2: Iterator[(K, V2)],
                                      validate: Boolean = true)
                                     (implicit ork: Ordering[K]) = new Iterator[(K, Option[V1], Option[V2])]() {

    private val iterator1_ = if (validate) {
      validateKeyOrdering(iterator1)(ork)
    } else {
      iterator1
    }

    private val iterator2_ = if (validate) {
      validateKeyOrdering(iterator2)(ork)
    } else {
      iterator2
    }

    private var kv1 = Option.empty[(K, V1)]
    private var kv2 = Option.empty[(K, V2)]

    {
      updateKV1()
      updateKV2()
    }

    private def updateKV1(): Unit = {
      kv1 = if (iterator1_.hasNext) {
        Some(iterator1_.next)
      } else {
        None
      }
    }

    private def updateKV2(): Unit = {
      kv2 = if (iterator2_.hasNext) {
        Some(iterator2_.next)
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
          val ret = (k1, Some(v1), Some(v2))
          updateKV1()
          updateKV2()
          ret

        } else if (cmp < 0) {
          val ret = (k1, Some(v1), None)
          updateKV1()
          ret

        } else {
          val ret = (k2, None, Some(v2))
          updateKV2()
          ret
        }

      case (Some((k1, v1)), None) =>
        val ret = (k1, Some(v1), None)
        updateKV1()
        ret

      case (None, Some((k2, v2))) =>
        val ret = (k2, None, Some(v2))
        updateKV2()
        ret
    }
  }


  /**
    * Inner join of two Strictly Ascending iterators
    */
  def innerJoinSortedIters[K, V1, V2](iterator1: Iterator[(K, V1)],
                                      iterator2: Iterator[(K, V2)],
                                      validate: Boolean = true)
                                     (implicit ork: Ordering[K]) = new Iterator[(K, V1, V2)]() {

    private val iterator1_ = if (validate) {
      validateKeyOrdering(iterator1)(ork)
    } else {
      iterator1
    }

    private val iterator2_ = if (validate) {
      validateKeyOrdering(iterator2)(ork)
    } else {
      iterator2
    }

    private var kv1 = Option.empty[(K, V1)]
    private var kv2 = Option.empty[(K, V2)]
    private var cmp = 0

    {
      update()
    }

    private def updateKV1(): Unit = {
      cmp = -1
      while (iterator1_.hasNext && cmp < 0) {
        kv1 = Some(iterator1_.next)
        cmp = ork.compare(kv1.get._1, kv2.get._1)
      }

      if (cmp < 0) {
        kv1 = None
      }
    }

    private def updateKV2(): Unit = {
      cmp = 1
      while (iterator2_.hasNext && cmp > 0) {
        kv2 = Some(iterator2_.next)
        cmp = ork.compare(kv1.get._1, kv2.get._1)
      }

      if (cmp > 0) {
        kv2 = None
      }
    }

    private def update(): Unit = {
      kv1 = if (iterator1_.hasNext) {
        Some(iterator1_.next)
      } else {
        None
      }

      kv2 = if (iterator2_.hasNext) {
        Some(iterator2_.next)
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
      val ret = (kv1.get._1, kv1.get._2, kv2.get._2)
      update()
      ret
    }
  }


  /**
    * Perform reduce operation by continuous identical keys
    * E.g, keys = (1,1,1,5,5,2,2,1), will reduce on keysets (1,1,1),(5,5),(2,2),(1)
    */
  def reduceByKey[K, V](iterator: Iterator[(K, V)],
                        func: (V, V) => V)
                       (implicit ork: Ordering[K]) = new Iterator[(K, V)]() {

    private var kv1 = Option.empty[(K, V)]
    private var kv2 = Option.empty[(K, V)]
    private var cmp = 0

    {
      update()
    }

    private def update(): Unit = {

      if (kv1.isEmpty && iterator.hasNext) {
        kv1 = Some(iterator.next)
      } else if (kv2.nonEmpty) {
        kv1 = kv2
      } else {
        kv1 = None
      }

      kv2 = None

      if (kv1.nonEmpty) {
        cmp = 0
        while (cmp == 0 && iterator.hasNext) {
          kv2 = Some(iterator.next)
          cmp = ork.compare(kv1.get._1, kv2.get._1)
          if (cmp == 0) {
            kv1 = Some(kv1.get._1, func(kv1.get._2, kv2.get._2))
          }
        }

        if (cmp == 0) {
          kv2 = None
        }
      }
    }

    override def hasNext: Boolean = {
      kv1.nonEmpty
    }

    override def next(): (K, V) = {
      val ret = kv1.get
      update()
      ret
    }
  }


  /**
    * Perform aggregate operation by continuous identical keys
    * E.g, keys = (1,1,1,5,5,2,2,1), will aggregate on keysets (1,1,1),(5,5),(2,2),(1)
    */
  def aggregateByKey[K, V, C](iterator: Iterator[(K, V)],
                              createCombiner: () => C,
                              func: (C, V) => C)
                             (implicit ork: Ordering[K]) = new Iterator[(K, C)]() {

    private var kc = Option.empty[(K, C)]
    private var kv = Option.empty[(K, V)]
    private var cmp = 0

    {
      update()
    }

    private def update(): Unit = {
      if (kc.isEmpty && iterator.hasNext) {
        val t = iterator.next
        kc = Some(t._1, func(createCombiner(), t._2))
      } else if (kv.nonEmpty) {
        kc = Some(kv.get._1, func(createCombiner(), kv.get._2))
      } else {
        kc = None
      }

      kv = None

      if (kc.nonEmpty) {
        cmp = 0
        while (cmp == 0 && iterator.hasNext) {
          kv = Some(iterator.next)
          cmp = ork.compare(kv.get._1, kc.get._1)
          if (cmp == 0) {
            kc = Some(kc.get._1, func(kc.get._2, kv.get._2))
          }
        }

        if (cmp == 0) {
          kv = None
        }
      }
    }

    override def hasNext: Boolean = {
      kc.nonEmpty
    }

    override def next(): (K, C) = {
      val ret = kc.get
      update()
      ret
    }
  }


  /**
    * Traverse all the elements of a vector
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
    * Traverse only the non-zero elements of a vector
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
    * Helper function to save dataframes
    */
  def saveDataFrames(dataframes: Array[DataFrame],
                     names: Array[String],
                     path: String): Unit = {
    require(dataframes.length == names.length)
    require(names.length == names.distinct.length)

    var i = 0
    while (i < dataframes.length) {
      dataframes(i).write
        .parquet(new Path(path, names(i)).toString)
      i += 1
    }
  }


  /**
    * Helper function to load dataframes
    */
  def loadDataFrames(spark: SparkSession,
                     names: Array[String],
                     path: String): Array[DataFrame] = {
    names.map { name =>
      spark.read.parquet(new Path(path, name).toString)
    }
  }


  /**
    * Remove RDD's checkpoint files.
    * This prints a warning but does not fail if the files cannot be removed.
    */
  def removeCheckpointFile(data: RDD[_],
                           blocking: Boolean = true): Unit = {
    if (blocking) {
      data.getCheckpointFile.foreach { file =>
        try {
          val tic = System.nanoTime()
          val path = new Path(file)
          val fs = path.getFileSystem(data.sparkContext.hadoopConfiguration)
          fs.delete(path, true)
          logInfo(s"Successfully remove old checkpoint file: $file, " +
            s"duration ${(System.nanoTime() - tic) / 1e9} seconds")
        } catch {
          case e: Exception =>
            logWarning(s"Fail to remove old checkpoint file: $file, ${e.toString}")
        }
      }

    } else {
      data.getCheckpointFile.foreach { file =>
        Future {
          val tic = System.nanoTime()
          val path = new Path(file)
          val fs = path.getFileSystem(data.sparkContext.hadoopConfiguration)
          fs.delete(path, true)
          (System.nanoTime() - tic) / 1e9

        }.onComplete {
          case Success(v) =>
            logInfo(s"Successfully remove old checkpoint file: $file, duration $v seconds")

          case Failure(t) =>
            logWarning(s"Fail to remove old checkpoint file: $file, ${t.toString}")
        }
      }
    }
  }


  private[this] var kryoRegistered: Boolean = false

  def registerKryoClasses(sc: SparkContext): Unit = {
    if (!kryoRegistered) {
      sc.getConf.registerKryoClasses(
        Array(
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

          classOf[Selector],
          classOf[TrueSelector],
          classOf[HashSelector],
          classOf[SetSelector],
          classOf[UnionSelector],

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
          classOf[AUROCEval],
          classOf[AUPRCEval],

          classOf[CallbackFunc],
          classOf[EarlyStop],
          classOf[MetricRecoder],
          classOf[ModelCheckpoint],
          classOf[ClassificationModelCheckpoint],
          classOf[RegressionModelCheckpoint],

          classOf[KVVector[_, _]],
          classOf[DenseKVVector[_, _]],
          classOf[SparseKVVector[_, _]],
          classOf[KVMatrix[_, _]],
          classOf[ArrayBlock[_]],
          classOf[CompactArray[_]],

          classOf[SkipNodePratitioner[_, _, _]],
          classOf[DepthPratitioner[_, _, _]],
          classOf[RangePratitioner[_, _, _]]))

      kryoRegistered = true
    }
  }
}
