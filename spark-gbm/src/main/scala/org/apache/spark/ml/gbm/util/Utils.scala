package org.apache.spark.ml.gbm.util

import scala.collection._

import org.apache.hadoop.fs.Path

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.linalg._
import org.apache.spark.sql._


private[gbm] object Utils extends Logging {

  val BYTE = "Byte"
  val SHORT = "Short"
  val INT = "Int"
  val LONG = "Long"

  /**
    * Choose the most compact Integer Type according to the range
    */
  def getTypeByRange(range: Int): String = {
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
    * return true if two arrays are of the same length and contain the same values.
    */
  def arrayEquiv[V](array1: Array[V], array2: Array[V])
                   (implicit orv: Ordering[V]): Boolean = {
    (array1 != null, array2 != null) match {
      case (true, true) =>
        (array1.length == array2.length) &&
          array1.iterator.zip(array2.iterator).forall { case (v1, v2) => orv.equiv(v1, v2) }

      case (true, false) => false

      case (false, true) => false

      case (false, false) => true
    }
  }


  /**
    * zip three iterators
    */
  def zip3[V1, V2, V3](iter1: Iterator[V1],
                       iter2: Iterator[V2],
                       iter3: Iterator[V3]): Iterator[(V1, V2, V3)] = new AbstractIterator[(V1, V2, V3)] {
    def hasNext: Boolean = iter1.hasNext && iter2.hasNext && iter3.hasNext

    def next: (V1, V2, V3) = (iter1.next(), iter2.next(), iter3.next())
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
                               (implicit ork: Ordering[K]): Iterator[(K, V)] = new Iterator[(K, V)]() {
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
    * Partially aggregate elements
    */
  def partialAggregate[V, C, U](iterator: Iterator[V],
                                createCombiner: () => C,
                                mergeValue: (C, V) => C,
                                pause: (C, Long, Long) => Boolean,
                                iterate: C => Iterator[U]): Iterator[U] = {
    var index = -1L
    var partialIndex = -1L

    var combiner = createCombiner()

    iterator.flatMap { value =>
      index += 1
      partialIndex += 1

      combiner = mergeValue(combiner, value)

      if (pause(combiner, partialIndex, index)) {
        iterate(combiner) ++ {
          partialIndex = -1
          combiner = createCombiner()
          Iterator.empty
        }

      } else {
        Iterator.empty
      }

    } ++ {
      if (partialIndex >= 0) {
        iterate(combiner)
      } else {
        Iterator.empty
      }
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

        classOf[RangePratitioner[Any, Any, Any]],
        classOf[RangePratitioner[Byte, Byte, Byte]],
        classOf[RangePratitioner[Byte, Byte, Short]],
        classOf[RangePratitioner[Byte, Byte, Int]],
        classOf[RangePratitioner[Byte, Short, Byte]],
        classOf[RangePratitioner[Byte, Short, Short]],
        classOf[RangePratitioner[Byte, Short, Int]],
        classOf[RangePratitioner[Byte, Int, Byte]],
        classOf[RangePratitioner[Byte, Int, Short]],
        classOf[RangePratitioner[Byte, Int, Int]],
        classOf[RangePratitioner[Short, Byte, Byte]],
        classOf[RangePratitioner[Short, Byte, Short]],
        classOf[RangePratitioner[Short, Byte, Int]],
        classOf[RangePratitioner[Short, Short, Byte]],
        classOf[RangePratitioner[Short, Short, Short]],
        classOf[RangePratitioner[Short, Short, Int]],
        classOf[RangePratitioner[Short, Int, Byte]],
        classOf[RangePratitioner[Short, Int, Short]],
        classOf[RangePratitioner[Short, Int, Int]],
        classOf[RangePratitioner[Int, Byte, Byte]],
        classOf[RangePratitioner[Int, Byte, Short]],
        classOf[RangePratitioner[Int, Byte, Int]],
        classOf[RangePratitioner[Int, Short, Byte]],
        classOf[RangePratitioner[Int, Short, Short]],
        classOf[RangePratitioner[Int, Short, Int]],
        classOf[RangePratitioner[Int, Int, Byte]],
        classOf[RangePratitioner[Int, Int, Short]],
        classOf[RangePratitioner[Int, Int, Int]]))

      kryoRegistered = true
    }
  }
}
