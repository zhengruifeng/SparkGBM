package org.apache.spark.ml.gbm

import java.{util => ju}

import scala.collection.{BitSet, mutable}
import scala.reflect.ClassTag
import scala.{specialized => spec}

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.gbm.linalg._
import org.apache.spark.ml.gbm.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.util.QuantileSummaries


class Discretizer(val numCols: Int,
                  val maxBins: Int,
                  val discretizationType: String,
                  val zeroAsMissing: Boolean,
                  val nonNumericalCols: BitSet,
                  val colDiscretizers: Map[Int, ColDiscretizer]) extends Serializable {

  import GBM.{Floor, Ceil, Round, Width, Depth}

  def numericalMode: String = discretizationType.split(":").head

  def nonNumericalMode: String = discretizationType.split(":").last

  require(numCols > 0)
  require(maxBins > 0)
  require(discretizationType.split(":").length == 2)
  require(Array(Floor, Ceil, Round, Width, Depth).contains(numericalMode))
  require(Array(Floor, Ceil, Round).contains(nonNumericalMode))


  @transient private lazy val numericalBinFunc = numericalMode match {
    case Floor =>
      (v: Double) =>
        val b = v.floor.toInt
        math.max(0, math.min(b, maxBins - 1))

    case Ceil =>
      (v: Double) =>
        val b = v.ceil.toInt
        math.max(0, math.min(b, maxBins - 1))

    case Round =>
      (v: Double) =>
        val b = v.round.toInt
        math.max(0, math.min(b, maxBins - 1))

    case _ =>
      null
  }


  @transient private lazy val nonNumericalBinFunc = nonNumericalMode match {
    case Floor =>
      (v: Double) =>
        val b = v.floor.toInt
        math.max(0, math.min(b, maxBins - 1))

    case Ceil =>
      (v: Double) =>
        val b = v.ceil.toInt
        math.max(0, math.min(b, maxBins - 1))

    case Round =>
      (v: Double) =>
        val b = v.round.toInt
        math.max(0, math.min(b, maxBins - 1))
  }


  @transient private lazy val discretizeFunc = (nonNumericalCols.nonEmpty, colDiscretizers.nonEmpty) match {
    case (true, true) =>
      (v: Double, i: Int) =>

        if (nonNumericalCols.contains(i)) {
          nonNumericalBinFunc(v)

        } else {

          if (v.isNaN || v.isInfinity) {
            0
          } else if (zeroAsMissing && v == 0) {
            0
          } else {
            colDiscretizers(i).transform(v)
          }
        }


    case (true, false) =>
      (v: Double, i: Int) =>
        if (nonNumericalCols.contains(i)) {
          nonNumericalBinFunc(v)
        } else {
          numericalBinFunc(v)
        }


    case (false, true) =>
      (v: Double, i: Int) =>
        if (v.isNaN || v.isInfinity) {
          0
        } else if (zeroAsMissing && v == 0) {
          0
        } else {
          colDiscretizers(i).transform(v)
        }


    case (false, false) =>
      (v: Double, i: Int) =>
        numericalBinFunc(v)
  }


  def transform[@spec(Byte, Short, Int) B](vec: Vector)
                                          (implicit cb: ClassTag[B], inb: Integral[B]): Array[B] = {

    require(vec.size == numCols)

    val bins = Array.ofDim[B](numCols)

    val iter = if (zeroAsMissing) {
      Utils.getActiveIter(vec)
    } else {
      Utils.getTotalIter(vec)
    }

    while (iter.hasNext) {
      val (i, v) = iter.next()
      val b = discretizeFunc(v, i)
      bins(i) = inb.fromInt(b)
    }

    bins
  }


  private[gbm] def transformToKVVector[@spec(Byte, Short, Int) C, @spec(Byte, Short, Int) B](vec: Vector)
                                                                                            (implicit cc: ClassTag[C], inc: Integral[C], nec: NumericExt[C],
                                                                                             cb: ClassTag[B], inb: Integral[B], neb: NumericExt[B]): KVVector[C, B] = {
    require(vec.size == numCols)

    vec match {
      case sv: SparseVector if colDiscretizers.isEmpty || zeroAsMissing =>
        val indices = nec.fromInt(sv.indices)
        val bins = Array.ofDim[B](sv.values.length)

        var i = 0
        while (i < sv.values.length) {
          val b = discretizeFunc(sv.values(i), sv.indices(i))
          bins(i) = inb.fromInt(b)
          i += 1
        }
        KVVector.sparse[C, B](numCols, indices, bins)


      case _ =>
        val bins = transform[B](vec)
        KVVector.dense[C, B](bins)
    }
  }

  def numFeatures: Int = numCols

  def copy(): Discretizer = {
    new Discretizer(numCols, maxBins, discretizationType, zeroAsMissing, nonNumericalCols, colDiscretizers)
  }


  def save(spark: SparkSession,
           path: String): Unit = {

    val paramsDF = spark.createDataFrame(Seq((classOf[Discretizer].getName, numCols, maxBins, discretizationType, zeroAsMissing)))
      .toDF("class", "numCols", "maxBins", "discretizationType", "zeroAsMissing")


    val nonNumColsDF = spark.createDataFrame(nonNumericalCols.toSeq.map((_, true)))
      .toDF("featureIndex", "nonNumerical")


    val colDiscretizersDF = {
      val datum = colDiscretizers.iterator.map {
        case (i, num: DepthNumColDiscretizer) =>
          (i, "depth", num.splits, Array.emptyIntArray)
        case (i, num: WidthNumColDiscretizer) =>
          (i, "width", Array(num.start, num.step), Array(num.numBins))
      }.toArray

      spark.createDataFrame(datum)
        .toDF("featureIndex", "type", "doubles", "ints")
    }


    Utils.saveDataFrames(Array(paramsDF, nonNumColsDF, colDiscretizersDF),
      Array("params", "nonNumCols", "colDiscretizers"), path)
  }
}


private[gbm] object Discretizer extends Logging {

  import GBM.{Floor, Ceil, Round, Width, Depth}

  /**
    * Implementation of training a discretizer
    *
    * @param vectors            input dataset
    * @param numCols            number of columns
    * @param maxBins            maximun number of bins, staring from 0
    * @param nonNumericalCols   indices of categorical and ranking columns
    * @param discretizationType method to deal with different columns
    * @param zeroAsMissing      whether zero is viewed as missing value
    * @param depth              aggregation depth
    * @return discretizer
    */
  def fit(vectors: RDD[Vector],
          numCols: Int,
          maxBins: Int,
          nonNumericalCols: BitSet,
          discretizationType: String,
          zeroAsMissing: Boolean,
          depth: Int): Discretizer = {
    require(maxBins >= 4)
    require(numCols >= 1)
    require(discretizationType.split(":").length == 2)

    val Array(numericalMode, nonNumericalMode) = discretizationType.split(":")
    require(Array(Floor, Ceil, Round, Width, Depth).contains(numericalMode))
    require(Array(Floor, Ceil, Round).contains(nonNumericalMode))

    if (Array(Floor, Ceil, Round).contains(numericalMode) || nonNumericalCols.size == numCols) {
      return new Discretizer(numCols, maxBins, discretizationType, zeroAsMissing, nonNumericalCols, Map.empty)
    }


    val tic = System.nanoTime()
    logInfo(s"Discretizer building start")


    // zero bin index is always reserved for missing value
    val emptyAggs = numericalMode match {
      case Width =>
        Iterator.range(0, numCols)
          .filterNot(nonNumericalCols.contains)
          .map((_, new WidthNumColAgg(maxBins - 1)))
          .toMap

      case Depth =>
        Iterator.range(0, numCols)
          .filterNot(nonNumericalCols.contains)
          .map((_, new DepthNumColAgg(maxBins - 1)))
          .toMap
    }


    val (count, aggregated) =
      vectors.mapPartitions { iter =>
        var cnt = 0L
        val aggs = emptyAggs
        val nans = mutable.OpenHashMap.empty[Int, Long]

        // only absorb non-zero values
        while (iter.hasNext) {
          val vec = iter.next()
          require(vec.size == numCols)

          val iter2 = Utils.getActiveIter(vec)
            .filterNot(t => nonNumericalCols.contains(t._1))

          while (iter2.hasNext) {
            val (i, v) = iter2.next()
            if (!v.isNaN && !v.isInfinity) {
              aggs(i).update(v)
            } else if (!zeroAsMissing) {
              val cnt = nans.getOrElse(i, 0L)
              nans.update(i, cnt + 1)
            }
          }

          cnt += 1
        }

        // if zero is not missing, add zeros back
        if (!zeroAsMissing) {
          var i = 0
          while (i < numCols) {
            if (!nonNumericalCols.contains(i)) {
              val nz = cnt - aggs(i).count - nans.getOrElse(i, 0L)
              aggs(i).updateZeros(nz)
            }
            i += 1
          }
        }
        nans.clear()


        if (cnt > 0) {
          Iterator.single((cnt, aggs))
        } else {
          Iterator.empty
        }

      }.treeReduce(f = {
        case ((cnt1, aggs1), (cnt2, aggs2)) =>
          var i = 0
          while (i < numCols) {
            aggs1(i).merge(aggs2(i))
            i += 1
          }
          (cnt1 + cnt2, aggs1)
      }, depth = depth)

    val colDiscretizers = aggregated.map { case (col, agg) => (col, agg.toColDiscretizer) }

    logInfo(s"Discretizer building finished, duration: ${(System.nanoTime() - tic) / 1e9} sec")

    new Discretizer(numCols, maxBins, discretizationType, zeroAsMissing, nonNumericalCols, colDiscretizers)
  }


  /**
    * Implementation of training a discretizer, by the way, compute the avg of labels.
    */
  private[gbm] def fit2(instances: RDD[(Double, Array[Double], Vector)],
                        numCols: Int,
                        maxBins: Int,
                        nonNumericalCols: BitSet,
                        discretizationType: String,
                        zeroAsMissing: Boolean,
                        depth: Int): (Discretizer, Array[Double]) = {
    require(maxBins >= 4)
    require(numCols >= 1)
    require(discretizationType.split(":").length == 2)

    val Array(numericalMode, nonNumericalMode) = discretizationType.split(":")
    require(Array(Floor, Ceil, Round, Width, Depth).contains(numericalMode))
    require(Array(Floor, Ceil, Round).contains(nonNumericalMode))

    if (Array(Floor, Ceil, Round).contains(numericalMode) || nonNumericalCols.size == numCols) {
      val labelAvg = computeAverageLabel(instances.map(t => (t._1, t._2)), depth)
      return (new Discretizer(numCols, maxBins, discretizationType, zeroAsMissing, nonNumericalCols, Map.empty), labelAvg)
    }


    val tic = System.nanoTime()
    logInfo(s"Discretizer building start")


    // zero bin index is always reserved for missing value
    val emptyAggs = numericalMode match {
      case Width =>
        Iterator.range(0, numCols)
          .filterNot(nonNumericalCols.contains)
          .map((_, new WidthNumColAgg(maxBins - 1)))
          .toMap

      case Depth =>
        Iterator.range(0, numCols)
          .filterNot(nonNumericalCols.contains)
          .map((_, new DepthNumColAgg(maxBins - 1)))
          .toMap
    }


    val (_, labelAvg, count, aggregated) =
      instances.mapPartitions { iter =>
        var cnt = 0L
        val aggs = emptyAggs
        val nans = mutable.OpenHashMap.empty[Int, Long]

        var labelAvg = Array.emptyDoubleArray
        var weightSum = 0.0

        // only absorb non-zero values
        while (iter.hasNext) {
          val (weight, label, vec) = iter.next()
          require(vec.size == numCols)

          // update avg of label
          if (labelAvg.isEmpty) {
            labelAvg = label
          } else {
            require(labelAvg.length == label.length)
            val f = weight / (weight + weightSum)
            var i = 0
            while (i < labelAvg.length) {
              labelAvg(i) += (label(i) - labelAvg(i)) * f
              i += 1
            }
          }
          weightSum += weight

          // update column metrics
          val iter2 = Utils.getActiveIter(vec)
            .filterNot(t => nonNumericalCols.contains(t._1))

          while (iter2.hasNext) {
            val (i, v) = iter2.next()
            if (!v.isNaN && !v.isInfinity) {
              aggs(i).update(v)
            } else if (!zeroAsMissing) {
              val cnt = nans.getOrElse(i, 0L)
              nans.update(i, cnt + 1)
            }
          }

          cnt += 1
        }

        // if zero is not missing, add zeros back
        if (!zeroAsMissing) {
          var i = 0
          while (i < numCols) {
            if (!nonNumericalCols.contains(i)) {
              val nz = cnt - aggs(i).count - nans.getOrElse(i, 0L)
              aggs(i).updateZeros(nz)
            }
            i += 1
          }
        }
        nans.clear()


        if (cnt > 0) {
          Iterator.single((weightSum, labelAvg, cnt, aggs))
        } else {
          Iterator.empty
        }

      }.treeReduce(f = {
        case ((w1, avg1, cnt1, aggs1), (w2, avg2, cnt2, aggs2)) =>
          require(avg1.length == avg2.length)
          val f = w2 / (w1 + w2)

          var i = 0
          while (i < avg1.length) {
            avg1(i) += (avg2(i) - avg1(i)) * f
            i += 1
          }

          i = 0
          while (i < numCols) {
            aggs1(i).merge(aggs2(i))
            i += 1
          }

          (w1 + w2, avg1, cnt1 + cnt2, aggs1)
      }, depth = depth)


    val colDiscretizers = aggregated.map { case (col, agg) => (col, agg.toColDiscretizer) }

    logInfo(s"Discretizer building finished, duration: ${(System.nanoTime() - tic) / 1e9} sec")

    val discretizer = new Discretizer(numCols, maxBins, discretizationType, zeroAsMissing, nonNumericalCols, colDiscretizers)

    (discretizer, labelAvg)
  }


  def computeAverageLabel(data: RDD[(Double, Array[Double])],
                          depth: Int): Array[Double] = {
    val tic = System.nanoTime()
    logInfo(s"Average label computation start")

    val (_, labelAvg) = data.mapPartitions { iter =>
      var labelAvg = Array.emptyDoubleArray
      var weightSum = 0.0

      // only absorb non-zero values
      while (iter.hasNext) {
        val (weight, label) = iter.next()

        // update avg of label
        if (labelAvg.isEmpty) {
          labelAvg = label
        } else {
          require(labelAvg.length == label.length)
          val f = weight / (weight + weightSum)
          var i = 0
          while (i < labelAvg.length) {
            labelAvg(i) += (label(i) - labelAvg(i)) * f
            i += 1
          }
        }
        weightSum += weight
      }

      if (labelAvg.nonEmpty) {
        Iterator.single((weightSum, labelAvg))
      } else {
        Iterator.empty
      }

    }.treeReduce(f = {
      case ((w1, avg1), (w2, avg2)) =>
        require(avg1.length == avg2.length)
        val f = w2 / (w1 + w2)

        var i = 0
        while (i < avg1.length) {
          avg1(i) += (avg2(i) - avg1(i)) * f
          i += 1
        }

        (w1 + w2, avg1)
    }, depth = depth)

    logInfo(s"Average label computation finished, " +
      s"avgLabel: ${labelAvg.mkString("[", ",", "]")}, " +
      s"duration: ${(System.nanoTime() - tic) / 1e9} sec")

    labelAvg
  }


  /**
    * Comupte the proportion of missing value
    */
  def computeSparsity(vectors: RDD[Vector],
                      numCols: Int,
                      zeroAsMissing: Boolean,
                      depth: Int): Double = {

    val tic = System.nanoTime()
    logInfo(s"Dataset sparsity computation start")

    // compute number of non-missing for each row
    val countNNM = if (zeroAsMissing) {
      vec: Vector => {
        Utils.getActiveIter(vec).count { case (i, v) =>
          !v.isNaN && !v.isInfinity
        }
      }

    } else {

      vec: Vector => {
        vec.size - Utils.getActiveIter(vec).count { case (i, v) =>
          v.isNaN || v.isInfinity
        }
      }
    }

    val (_, nnm) = vectors.treeAggregate[(Long, Double)]((0L, 0.0))(
      seqOp = {
        case ((cnt, avg), vec) =>
          require(vec.size == numCols)
          val nnm = countNNM(vec)
          val diff = (nnm - avg) / (cnt + 1)
          (cnt + 1, avg + diff)
      },
      combOp = {
        case ((cnt1, avg1), (cnt2, avg2)) =>
          if (cnt1 + cnt2 > 0) {
            val diff = cnt2 / (cnt1 + cnt2) * (avg2 - avg1)
            (cnt1 + cnt2, avg1 + diff)
          } else {
            (0L, 0.0)
          }
      },
      depth = depth
    )

    val sparsity = 1 - nnm / numCols
    logInfo(s"Dataset sparsity computation finished, sparsity: $sparsity, " +
      s"duration: ${(System.nanoTime() - tic) / 1e9} sec")
    sparsity
  }


  def load(spark: SparkSession,
           path: String): Discretizer = {
    import spark.implicits._

    val Array(paramsDF, nonNumColsDF, colDiscretizersDF) =
      Utils.loadDataFrames(spark, Array("params", "nonNumCols", "colDiscretizers"), path)


    val (numCols, maxBins, discretizationType, zeroAsMissing) =
      paramsDF.select("numCols", "maxBins", "discretizationType", "zeroAsMissing")
        .as[(Int, Int, String, Boolean)]
        .first()


    val nonNumCols =
      nonNumColsDF.select("featureIndex")
        .as[Int]
        .collect()

    val builder = BitSet.newBuilder
    builder ++= nonNumCols
    val nonNumericalCols = builder.result()


    val colDiscretizers = colDiscretizersDF.select("featureIndex", "type", "doubles", "ints")
      .as[(Int, String, Array[Double], Array[Int])]
      .rdd
      .map { case (i, tpe, doubles, ints) =>
        val col = tpe match {
          case "depth" =>
            require(ints.isEmpty)
            new DepthNumColDiscretizer(doubles)
          case "width" =>
            require(doubles.length == 2 && ints.length == 1)
            new WidthNumColDiscretizer(doubles.head, doubles.last, ints.head)
        }
        (i, col)
      }.collect().toMap

    new Discretizer(numCols, maxBins, discretizationType, zeroAsMissing, nonNumericalCols, colDiscretizers)
  }
}


/**
  * discretizer for one column
  */
private[gbm] trait ColDiscretizer extends Serializable {
  /**
    * convert real values into bins, indices of bins start from 1.
    *
    * @param value real value
    * @return bin index
    */
  def transform(value: Double): Int

  def numBins: Int
}


/**
  * discretizer for one numerical column, all intervals are of the same depth (quantile)
  *
  * @param splits splitting points
  */
private[gbm] class DepthNumColDiscretizer(val splits: Array[Double]) extends ColDiscretizer {

  // splits = [q0.25, q0.75]
  // value <= q0.25           -> bin = 1
  // q0.25 < value <= q0.75   -> bin = 2
  // value > q0.75            -> bin = 3
  override def transform(value: Double): Int = {
    if (splits.isEmpty) {
      1
    } else {
      val index = ju.Arrays.binarySearch(splits, value)
      if (index >= 0) {
        index + 1
      } else {
        -index
      }
    }
  }

  override def numBins: Int = splits.length + 1
}


/**
  * discretizer for one numerical column, all intervals are of the same length
  *
  * @param start   start point
  * @param step    length of each interval
  * @param numBins number of bins
  */
private[gbm] class WidthNumColDiscretizer(val start: Double,
                                          val step: Double,
                                          val numBins: Int) extends ColDiscretizer {

  override def transform(value: Double): Int = {
    if (step == 0 || value <= start) {
      1
    } else {
      var index = (value - start) / step
      if (index == index.toInt) {
        index = index.toInt - 1
      }
      math.min(index.toInt + 2, numBins)
    }
  }
}


/**
  * aggregrator to build column discretizer
  */
private[gbm] trait ColAgg extends Serializable {

  def update(value: Double): ColAgg

  def updateZeros(nz: Long): ColAgg

  def merge(other: ColAgg): ColAgg

  def toColDiscretizer: ColDiscretizer

  def count: Long
}


/**
  * aggregrator for numerical column, find splits of same depth
  */
private[gbm] class DepthNumColAgg(val maxBins: Int) extends ColAgg {
  require(maxBins >= 2)

  var count = 0L
  var summary = DepthNumColAgg.createSummary

  override def update(value: Double): DepthNumColAgg = {
    summary = summary.insert(value)
    count += 1
    this
  }

  override def updateZeros(nz: Long): DepthNumColAgg = {
    if (nz > 0) {
      val nzSummary = DepthNumColAgg.createNZSummary(nz)
      summary = summary.compress.merge(nzSummary).compress
      count += nz
    }
    this
  }

  override def merge(other: ColAgg): DepthNumColAgg = {
    val o = other.asInstanceOf[DepthNumColAgg]
    summary = summary.compress.merge(o.summary.compress).compress
    count += o.count
    this
  }

  // maxBins = 3 -> interval = 0.5, queries = [0.25, 0.75], splits = [q0.25, q0.75]
  override def toColDiscretizer: DepthNumColDiscretizer = {
    if (count != 0) {
      summary = summary.compress
      val interval = 1.0 / (maxBins - 1)
      val start = interval / 2
      val queries = Array.range(0, maxBins - 1).map(i => start + interval * i)
      val splits = queries.flatMap(summary.query).distinct.sorted
      new DepthNumColDiscretizer(splits)
    } else {
      // all values in this column are missing value
      new DepthNumColDiscretizer(Array.emptyDoubleArray)
    }
  }
}


private[gbm] object DepthNumColAgg {

  val compressThreshold: Int = QuantileSummaries.defaultCompressThreshold

  val relativeError: Double = QuantileSummaries.defaultRelativeError

  def createSummary = new QuantileSummaries(compressThreshold, relativeError)

  private[this] lazy val nzSummaries = {
    val summaries = Array.ofDim[QuantileSummaries](63)
    summaries(0) = createSummary.insert(0.0).compress
    var i = 1
    while (i < summaries.length) {
      val s = summaries(i - 1).compress
      summaries(i) = s.merge(s).compress
      i += 1
    }
    summaries
  }

  /**
    * create a QuantileSummaries which has absorbed nz zeros
    */
  def createNZSummary(nz: Long): QuantileSummaries = {
    require(nz >= 0L)

    var summary = createSummary

    val binStr = nz.toBinaryString.reverse
    val char1 = "1".head

    var i = 0
    while (i < binStr.length) {
      if (binStr(i) == char1) {
        summary = summary.compress.merge(nzSummaries(i)).compress
      }
      i += 1
    }

    summary = summary.compress
    require(summary.count == nz)
    summary
  }
}


/**
  * aggregrator for numerical column, find splits of same width
  */
private[gbm] class WidthNumColAgg(val maxBins: Int) extends ColAgg {
  require(maxBins >= 2)

  var count = 0L
  var max = Double.MinValue
  var min = Double.MaxValue

  override def update(value: Double): WidthNumColAgg = {
    max = math.max(max, value)
    min = math.min(min, value)
    count += 1
    this
  }

  override def updateZeros(nz: Long): WidthNumColAgg = {
    if (nz > 0) {
      max = math.max(max, 0.0)
      min = math.min(min, 0.0)
      count += nz
    }
    this
  }

  override def merge(other: ColAgg): WidthNumColAgg = {
    val o = other.asInstanceOf[WidthNumColAgg]
    max = math.max(max, o.max)
    min = math.min(min, o.min)
    count += o.count
    this
  }

  // min = 0, max = 10, maxBins = 11, step = 10/10 = 1
  // if less than min+step/2 = 0.5 => 1, if greater than max-step/2 = 9.5 => 10
  override def toColDiscretizer: WidthNumColDiscretizer = {
    if (count > 0) {
      val step = (max - min) / (maxBins - 1)
      val start = min + step / 2
      new WidthNumColDiscretizer(start, step, maxBins)
    } else {
      // all values in this column are missing value
      new WidthNumColDiscretizer(0.0, 0.0, 1)
    }
  }
}

