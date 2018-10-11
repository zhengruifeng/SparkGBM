package org.apache.spark.ml.gbm.util

import org.apache.hadoop.fs.Path

import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.{DataFrame, SparkSession}


private[gbm] object Utils extends Logging {

  /**
    * Choose the most compact Integer Type according to the range
    */
  def getTypeByRange(range: Int): String = {
    if (range <= Byte.MaxValue) {
      "Byte"
    } else if (range <= Short.MaxValue) {
      "Short"
    } else {
      "Int"
    }
  }


  /**
    * Validate the ordering, and return the same iterator
    */
  def validateOrdering[K](iterator: Iterator[K],
                          ascending: Boolean = true,
                          strictly: Boolean = true)
                         (implicit ork: Ordering[K]): Iterator[K] = new Iterator[K]() {
    private var k = Option.empty[K]

    private val check = (ascending, strictly) match {
      case (true, true) => (i: K, j: K) => require(ork.compare(i, j) < 0, s"($i, $j) breaks strictly ascending")
      case (true, false) => (i: K, j: K) => require(ork.compare(i, j) <= 0, s"($i, $j) breaks ascending (non-descending)")
      case (false, true) => (i: K, j: K) => require(ork.compare(i, j) > 0, s"($i, $j) breaks strictly descending")
      case (false, false) => (i: K, j: K) => require(ork.compare(i, j) >= 0, s"($i, $j) breaks descending (non-ascending)")
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
      val t = k.get
      update()
      t
    }
  }


  /**
    * Validate the ordering of keys, and return the same iterator
    */
  def validateKeyOrdering[K, V](iterator: Iterator[(K, V)],
                                ascending: Boolean = true,
                                strictly: Boolean = true)
                               (implicit ork: Ordering[K]): Iterator[(K, V)] = new Iterator[(K, V)]() {
    private var kv = Option.empty[(K, V)]

    private val check = (ascending, strictly) match {
      case (true, true) => (i: K, j: K) => require(ork.compare(i, j) < 0, s"($i, $j) breaks strictly ascending")
      case (true, false) => (i: K, j: K) => require(ork.compare(i, j) <= 0, s"($i, $j) breaks ascending (non-descending)")
      case (false, true) => (i: K, j: K) => require(ork.compare(i, j) > 0, s"($i, $j) breaks strictly descending")
      case (false, false) => (i: K, j: K) => require(ork.compare(i, j) >= 0, s"($i, $j) breaks descending (non-ascending)")
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
      val t = kv.get
      update()
      t
    }
  }


  /**
    * Outer join of two Strictly Ascending iterators
    */
  def outerJoinSortedIters[K, V1, V2](iterator1: Iterator[(K, V1)],
                                      iterator2: Iterator[(K, V2)],
                                      validate: Boolean = true)
                                     (implicit ork: Ordering[K]): Iterator[(K, Option[V1], Option[V2])] = new Iterator[(K, Option[V1], Option[V2])]() {

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


  /**
    * Inner join of two Strictly Ascending iterators
    */
  def innerJoinSortedIters[K, V1, V2](iterator1: Iterator[(K, V1)],
                                      iterator2: Iterator[(K, V2)],
                                      validate: Boolean = true)
                                     (implicit ork: Ordering[K]): Iterator[(K, V1, V2)] = new Iterator[(K, V1, V2)]() {

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
      val t = (kv1.get._1, kv1.get._2, kv2.get._2)
      update()
      t
    }
  }


  /**
    * Perform reduce operation by continuous identical keys
    * E.g, keys = (1,1,1,5,5,2,2,1), will reduce on keysets (1,1,1),(5,5),(2,2),(1)
    */
  def reduceIterByKey[K, V](iterator: Iterator[(K, V)],
                            func: (V, V) => V)
                           (implicit ork: Ordering[K]): Iterator[(K, V)] = new Iterator[(K, V)]() {

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
      val t = kv1.get
      update()
      t
    }
  }


  /**
    * Perform aggregate operation by continuous identical keys
    * E.g, keys = (1,1,1,5,5,2,2,1), will aggregate on keysets (1,1,1),(5,5),(2,2),(1)
    */
  def aggregateIterByKey[K, V, C](iterator: Iterator[(K, V)],
                                  createZero: () => C,
                                  func: (C, V) => C)
                                 (implicit ork: Ordering[K]): Iterator[(K, C)] = new Iterator[(K, C)]() {

    private var kc = Option.empty[(K, C)]
    private var kv = Option.empty[(K, V)]
    private var cmp = 0

    {
      update()
    }

    private def update(): Unit = {
      if (kc.isEmpty && iterator.hasNext) {
        val t = iterator.next
        kc = Some(t._1, func(createZero(), t._2))
      } else if (kv.nonEmpty) {
        kc = Some(kv.get._1, func(createZero(), kv.get._2))
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
      val t = kc.get
      update()
      t
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
    dataframes.zip(names).foreach { case (df, name) =>
      df.write.parquet(new Path(path, name).toString)
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

        classOf[ArrayBlock[Any]],
        classOf[ArrayBlock[Byte]],
        classOf[ArrayBlock[Short]],
        classOf[ArrayBlock[Int]],
        classOf[ArrayBlock[Long]],
        classOf[ArrayBlock[Float]],
        classOf[ArrayBlock[Double]],

        classOf[CompactArray[Any]],
        classOf[CompactArray[Byte]],
        classOf[CompactArray[Short]],
        classOf[CompactArray[Int]],
        classOf[CompactArray[Long]],
        classOf[CompactArray[Float]],
        classOf[CompactArray[Double]],

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
