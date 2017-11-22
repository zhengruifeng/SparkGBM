package org.apache.spark.ml.gbm

import scala.collection.mutable

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.catalyst.util.QuantileSummaries

class Discretizer(val colDiscretizers: Array[ColDiscretizer]) extends Serializable {

  def transform(vec: Vector): Array[Int] = {
    require(vec.size == colDiscretizers.length)

    vec.toArray.zip(colDiscretizers).map {
      case (value, col) =>
        if (value.isNaN) {
          0
        } else {
          col.transform(value)
        }
    }
  }

  def numBins: Array[Int] = {
    /** zero bin index is always reserved for missing value */
    /** column discretizers do not handle missing value, and output bin indices start from 1 */
    colDiscretizers.map(_.numBins + 1)
  }
}

private[gbm] object Discretizer extends Logging {

  def fit(vectors: RDD[Vector],
          numCols: Int,
          catCols: Set[Int],
          rankCols: Set[Int],
          maxBins: Int,
          numericalBinType: String,
          depth: Int): Discretizer = {
    require(maxBins >= 4)
    require(numCols >= 1)

    val start = System.nanoTime
    logWarning(s"Discretizer building start")

    /** zero bin index is always reserved for missing value */
    val emptyAggs = Array.range(0, numCols).map { col =>
      if (catCols.contains(col)) {
        new CatColAgg(maxBins - 1)
      } else if (rankCols.contains(col)) {
        new RankAgg(maxBins - 1)
      } else if (numericalBinType == GBM.Depth) {
        new QuantileNumColAgg(maxBins - 1)
      } else {
        new IntervalNumColAgg(maxBins - 1)
      }
    }

    val aggregated = vectors.treeAggregate(emptyAggs)(
      seqOp = {
        case (aggs, vec) =>
          require(aggs.length == vec.size)
          var i = 0
          while (i < aggs.length) {
            /** column aggs do not deal with missing value */
            if (!vec(i).isNaN) {
              aggs(i).update(vec(i))
            }
            i += 1
          }
          aggs

      }, combOp = {
        case (aggs1, aggs2) =>
          require(aggs1.length == aggs2.length)
          var i = 0
          while (i < aggs1.length) {
            aggs1(i).merge(aggs2(i))
            i += 1
          }
          aggs1
      }, depth = depth)

    val colDiscretizers = aggregated.map(_.toColDiscretizer)

    logWarning(s"Discretizer building finished, duration ${(System.nanoTime - start) / 1e9} seconds")

    new Discretizer(colDiscretizers)
  }

  def toDF(discretizer: Discretizer): DataFrame = {
    val spark = SparkSession.builder.getOrCreate()

    val datum = discretizer.colDiscretizers.zipWithIndex.map {
      case (num: QuantileNumColDiscretizer, i) =>
        (i, "quantile", num.splits, Array.emptyIntArray)
      case (num: IntervalNumColDiscretizer, i) =>
        (i, "interval", Array(num.start, num.step), Array(num.numBins))
      case (cat: CatColDiscretizer, i) =>
        (i, "cat", Array.emptyDoubleArray, cat.array)
      case (rank: RankColDiscretizer, i) =>
        (i, "rank", Array.emptyDoubleArray, rank.array)
    }

    spark.createDataFrame(datum).toDF("featureIndex", "type", "doubles", "ints")
  }

  def fromDF(df: DataFrame): Discretizer = {
    val (indices, colDiscretizers) =
      df.select("featureIndex", "type", "doubles", "ints").rdd
        .map { row =>
          val i = row.getInt(0)
          val tpe = row.getString(1)
          val doubles = row.getSeq[Double](2)
          val ints = row.getSeq[Int](3)

          val col = tpe match {
            case "quantile" =>
              new QuantileNumColDiscretizer(doubles.toArray.sorted)
            case "interval" =>
              require(doubles.length == 2 && ints.length == 1)
              new IntervalNumColDiscretizer(doubles.head, doubles.last, ints.head)
            case "cat" =>
              new CatColDiscretizer(ints.toArray.sorted)
            case "rank" =>
              new RankColDiscretizer(ints.toArray.sorted)
          }

          (i, col)
        }.collect().sortBy(_._1).unzip

    require(indices.length == indices.distinct.length)
    require(indices.length == indices.max + 1)

    new Discretizer(colDiscretizers)
  }
}


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

private[gbm] class QuantileNumColDiscretizer(val splits: Array[Double]) extends ColDiscretizer {

  /** splits = [q0.25, q0.75] */
  /** value <= q0.25 -> j = 0 -> bin = 1 */
  /** q0.25 < value <= q0.75 -> j = 1 -> bin = 2 */
  /** value > q0.75 -> j = -1 -> bin = 3 */
  override def transform(value: Double): Int = {
    var j = -1
    var i = 0
    while (j < 0 && i < splits.length) {
      if (value <= splits(i)) {
        j = i
      }
      i += 1
    }
    if (j < 0) {
      splits.length + 1
    } else {
      j + 1
    }
  }

  override def numBins: Int = splits.length + 1
}


private[gbm] class IntervalNumColDiscretizer(val start: Double,
                                             val step: Double,
                                             val numBins: Int) extends ColDiscretizer {
  override def transform(value: Double): Int = {
    if (step == 0) {
      return 1
    }

    val index = ((value - start) / step).floor.toInt
    if (index < 0) {
      1
    } else {
      math.min(index + 2, numBins)
    }
  }
}


private[gbm] class CatColDiscretizer(val array: Array[Int]) extends ColDiscretizer {

  override def transform(value: Double): Int = {
    require(value.toInt == value)
    val index = array.indexOf(value.toInt)
    require(index >= 0, s"value $value not in ${array.mkString("(", ", ", ")")}")
    index + 1
  }

  override def numBins: Int = array.length
}

private[gbm] class RankColDiscretizer(val array: Array[Int]) extends ColDiscretizer {

  override def transform(value: Double): Int = {
    require(value.toInt == value)
    val index = java.util.Arrays.binarySearch(array, value.toInt)
    require(index >= 0, s"value $value not in ${array.mkString("(", ", ", ")")}")
    index + 1
  }

  override def numBins: Int = array.length
}


private[gbm] trait ColAgg extends Serializable {
  def update(value: Double): ColAgg

  def merge(other: ColAgg): ColAgg

  def toColDiscretizer: ColDiscretizer
}


private[gbm] class QuantileNumColAgg(val maxBins: Int) extends ColAgg {
  var summary = new QuantileSummaries(QuantileSummaries.defaultCompressThreshold, 0.001)

  override def update(value: Double): QuantileNumColAgg = {
    summary = summary.insert(value)
    this
  }

  override def merge(other: ColAgg): QuantileNumColAgg = {
    val otherSummary = other.asInstanceOf[QuantileNumColAgg].summary
    summary = summary.compress().merge(otherSummary.compress())
    this
  }

  /** maxBins = 3 -> interval = 0.5, queries = [0.25, 0.75], splits = [q0.25, q0.75] */
  override def toColDiscretizer: QuantileNumColDiscretizer = {
    summary = summary.compress()
    require(summary.count != 0)
    val interval = 1.0 / (maxBins - 1)
    val start = interval / 2
    val queries = Array.range(0, maxBins - 1).map(i => start + interval * i)
    val splits = queries.flatMap(summary.query).distinct.sorted
    new QuantileNumColDiscretizer(splits)
  }
}

private[gbm] class IntervalNumColAgg(val maxBins: Int) extends ColAgg {
  var max = Double.MinValue
  var min = Double.MaxValue

  override def update(value: Double): IntervalNumColAgg = {
    max = math.max(max, value)
    min = math.min(min, value)
    this
  }

  override def merge(other: ColAgg): IntervalNumColAgg = {
    val o = other.asInstanceOf[IntervalNumColAgg]
    max = math.max(max, o.max)
    min = math.min(min, o.min)
    this
  }

  /** min = 0, max = 10, maxBins = 11, step = 10/10 = 1
    * if less than min+step/2 = 0.5 => 1, if greater than max-step/2 = 9.5 => 10 */
  override def toColDiscretizer: IntervalNumColDiscretizer = {
    require(max >= min)
    val step = (max - min) / (maxBins - 1)
    val start = min + step / 2
    new IntervalNumColDiscretizer(start, step, maxBins)
  }
}

private[gbm] class CatColAgg(val maxBins: Int) extends ColAgg {
  val counter: mutable.Map[Int, Long] = mutable.Map[Int, Long]()

  override def update(value: Double): CatColAgg = {
    require(value.toInt == value)
    val cnt = counter.getOrElse(value.toInt, 0L)
    counter.update(value.toInt, cnt + 1)
    require(counter.size <= maxBins)
    this
  }

  override def merge(other: ColAgg): CatColAgg = {
    other.asInstanceOf[CatColAgg].counter
      .foreach { case (v, c) =>
        val cnt = counter.getOrElse(v, 0L)
        counter.update(v, cnt + c)
        require(counter.size <= maxBins)
      }
    this
  }

  override def toColDiscretizer: CatColDiscretizer = {
    val array = counter.toArray.sortBy(_._2).map(_._1).reverse
    new CatColDiscretizer(array)
  }
}

private[gbm] class RankAgg(val maxBins: Int) extends ColAgg {
  val set: mutable.Set[Int] = mutable.Set[Int]()

  override def update(value: Double): RankAgg = {
    require(value.toInt == value)
    set.add(value.toInt)
    require(set.size <= maxBins)
    this
  }

  override def merge(other: ColAgg): RankAgg = {
    other.asInstanceOf[RankAgg].set
      .foreach { v =>
        set.add(v)
        require(set.size <= maxBins)
      }
    this
  }

  override def toColDiscretizer: RankColDiscretizer = {
    val array = set.toArray.sorted
    new RankColDiscretizer(array)
  }
}

