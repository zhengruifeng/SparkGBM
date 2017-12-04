package org.apache.spark.ml.gbm

import java.{util => ju}

import scala.{specialized => spec}

import scala.collection.mutable
import scala.reflect.ClassTag
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.catalyst.util.QuantileSummaries


/**
  * discretizer for the rows
  *
  * @param colDiscretizers column discretizers for each column
  * @param zeroAsMissing   whether zero is viewed as missing value
  * @param sparsity        the sparsity of dataset to create this discretizer
  */
class Discretizer(val colDiscretizers: Array[ColDiscretizer],
                  val zeroAsMissing: Boolean,
                  val sparsity: Double) extends Serializable {


  def transformToArray(vec: Vector): Array[Int] = {
    require(vec.size == numCols)

    val bins = Array.ofDim[Int](numCols)

    if (zeroAsMissing) {
      vec.foreachActive { (i, v) =>
        bins(i) = discretizeWithIndex(v, i)
      }
    } else {
      var i = 0
      while (i < numCols) {
        bins(i) = discretizeWithIndex(vec(i), i)
        i += 1
      }
    }

    bins
  }


  private[gbm] def transformToVector[@spec(Byte, Short, Int) B: Integral : ClassTag](vec: Vector,
                                                                                     handleSparsity: Boolean): BinVector[B] = {
    require(vec.size == numCols)
    if (handleSparsity) {
      transformToSparse[B](vec)
    } else {
      transformToDense[B](vec)
    }
  }


  private def transformToSparse[@spec(Byte, Short, Int) B: Integral : ClassTag](vec: Vector): BinVector[B] = {
    val intB: Integral[B] = implicitly[Integral[B]]

    val indexBuff = mutable.ArrayBuffer[Int]()
    val valueBuff = mutable.ArrayBuffer[B]()

    if (zeroAsMissing) {
      vec.foreachActive { (i, v) =>
        val bin = discretizeWithIndex(v, i)
        if (bin != 0) {
          indexBuff.append(i)
          valueBuff.append(intB.fromInt(bin))
        }
      }
    } else {
      var i = 0
      while (i < numCols) {
        val bin = discretizeWithIndex(vec(i), i)
        if (bin != 0) {
          indexBuff.append(i)
          valueBuff.append(intB.fromInt(bin))
        }
        i += 1
      }
    }

    BinVector.sparse[B](numCols, indexBuff.toArray, valueBuff.toArray)
  }


  private def transformToDense[@spec(Byte, Short, Int) B: Integral : ClassTag](vec: Vector): BinVector[B] = {
    val intB = implicitly[Integral[B]]

    val bins = Array.fill(numCols)(intB.zero)

    if (zeroAsMissing) {
      vec.foreachActive { (i, v) =>
        val bin = discretizeWithIndex(v, i)
        if (bin != 0) {
          bins(i) = intB.fromInt(bin)
        }
      }
    } else {
      var i = 0
      while (i < numCols) {
        val bin = discretizeWithIndex(vec(i), i)
        bins(i) = intB.fromInt(bin)
        i += 1
      }
    }

    BinVector.dense[B](bins)
  }


  private[gbm] def discretizeWithIndex(value: Double, index: Int): Int = {
    if (value.isNaN || value.isInfinity) {
      0
    } else if (zeroAsMissing && value == 0) {
      0
    } else {
      colDiscretizers(index).transform(value)
    }
  }


  def numBins: Array[Int] = {
    // zero bin index is always reserved for missing value
    // column discretizers do not handle missing value, and output bin indices starting from 1
    colDiscretizers.map(_.numBins + 1)
  }

  def numCols: Int = colDiscretizers.length
}


private[gbm] object Discretizer extends Logging {

  /**
    * Implementation of training a discretizer
    *
    * @param vectors          input dataset
    * @param numCols          number of columns
    * @param catCols          indices of categorical columns
    * @param rankCols         indices of ranking columns
    * @param maxBins          maximun number of bins, staring from 0
    * @param numericalBinType method to deal with numerical column
    * @param zeroAsMissing    whether zero is viewed as missing value
    * @param depth            aggregation depth
    * @return discretizer
    */
  def fit(vectors: RDD[Vector],
          numCols: Int,
          catCols: Set[Int],
          rankCols: Set[Int],
          maxBins: Int,
          numericalBinType: String,
          zeroAsMissing: Boolean,
          depth: Int): Discretizer = {
    require(maxBins >= 4)
    require(numCols >= 1)

    val start = System.nanoTime
    logInfo(s"Discretizer building start")

    // zero bin index is always reserved for missing value
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


    val (count, aggregated) =
      if (zeroAsMissing) {
        vectors.treeAggregate[(Long, Array[ColAgg])]((0L, emptyAggs))(
          seqOp = {
            case ((cnt, aggs), vec) =>
              require(aggs.length == vec.size)
              vec.foreachActive { (i, v) =>
                // column aggs do not deal with missing value
                if (!v.isNaN && !v.isInfinity && v != 0) {
                  aggs(i).update(v)
                }
              }
              (cnt + 1, aggs)
          }, combOp = {
            case ((cnt1, aggs1), (cnt2, aggs2)) =>
              require(aggs1.length == aggs2.length)
              var i = 0
              while (i < aggs1.length) {
                aggs1(i).merge(aggs2(i))
                i += 1
              }
              (cnt1 + cnt2, aggs1)
          }, depth = depth)

      } else {

        vectors.treeAggregate[(Long, Array[ColAgg])]((0L, emptyAggs))(
          seqOp = {
            case ((cnt, aggs), vec) =>
              require(aggs.length == vec.size)
              var i = 0
              while (i < aggs.length) {
                // column aggs do not deal with missing value
                val v = vec(i)
                if (!v.isNaN && !v.isInfinity) {
                  aggs(i).update(v)
                }
                i += 1
              }
              (cnt + 1, aggs)
          }, combOp = {
            case ((cnt1, aggs1), (cnt2, aggs2)) =>
              require(aggs1.length == aggs2.length)
              var i = 0
              while (i < aggs1.length) {
                aggs1(i).merge(aggs2(i))
                i += 1
              }
              (cnt1 + cnt2, aggs1)
          }, depth = depth)
      }


    // number of non-missing
    val nnm = aggregated.map(_.count.toDouble).sum

    val sparsity = 1 - nnm / count / numCols

    logInfo(s"Discretizer building finished, data sparsity: $sparsity, duration: ${(System.nanoTime - start) / 1e9} sec")

    new Discretizer(aggregated.map(_.toColDiscretizer), zeroAsMissing, sparsity)
  }


  def computeSparsity(data: RDD[Vector],
                      numCols: Int,
                      zeroAsMissing: Boolean,
                      depth: Int): Double = {

    val nnmCounter = if (zeroAsMissing) {
      (vec: Vector) => {
        var nnm = 0
        vec.foreachActive { (i, v) =>
          if (!v.isNaN && !v.isInfinity && v != 0) {
            nnm += 1
          }
        }
        nnm
      }

    } else {

      (vec: Vector) => {
        var nnm = 0
        var i = 0
        while (i < numCols) {
          val v = vec(i)
          if (!v.isNaN && !v.isInfinity) {
            nnm += 1
          }
          i += 1
        }
        nnm
      }
    }

    val (_, nnm) = data.treeAggregate[(Long, Double)]((0L, 0.0))(
      seqOp = {
        case ((cnt, avg), vec) =>
          require(vec.size == numCols)
          val nnm = nnmCounter(vec)
          val diff = (nnm - avg) / (cnt + 1)
          (cnt + 1, avg + diff)
      },
      combOp = {
        case ((cnt1, avg1), (cnt2, avg2)) =>
          val diff = cnt2 / (cnt1 + cnt2) * (avg2 - avg1)
          (cnt1 + cnt2, avg1 + diff)
      },
      depth = depth
    )

    1 - nnm / numCols
  }


  /** helper function to convert Discretizer to dataframes */
  def toDF(spark: SparkSession,
           model: Discretizer): Array[DataFrame] = {

    val colDatum = model.colDiscretizers.zipWithIndex.map {
      case (num: QuantileNumColDiscretizer, i) =>
        (i, "quantile", num.splits, Array.emptyIntArray)
      case (num: IntervalNumColDiscretizer, i) =>
        (i, "interval", Array(num.start, num.step), Array(num.numBins))
      case (cat: CatColDiscretizer, i) =>
        (i, "cat", Array.emptyDoubleArray, cat.map.toArray.sortBy(_._2).map(_._1))
      case (rank: RankColDiscretizer, i) =>
        (i, "rank", Array.emptyDoubleArray, rank.array)
    }

    val colDF = spark.createDataFrame(colDatum)
      .toDF("featureIndex", "type", "doubles", "ints")

    val extraDF = spark.createDataFrame(
      Seq((model.zeroAsMissing, model.sparsity)))
      .toDF("zeroAsMissing", "sparsity")

    Array(colDF, extraDF)
  }


  /** helper function to convert dataframes back to Discretizer */
  def fromDF(dataframes: Array[DataFrame]): Discretizer = {
    val Array(colDF, extraDF) = dataframes

    val (indices, colDiscretizers) =
      colDF.select("featureIndex", "type", "doubles", "ints").rdd
        .map { row =>
          val i = row.getInt(0)
          val tpe = row.getString(1)
          val doubles = row.getSeq[Double](2)
          val ints = row.getSeq[Int](3)

          val col = tpe match {
            case "quantile" =>
              require(ints.isEmpty)
              new QuantileNumColDiscretizer(doubles.toArray)
            case "interval" =>
              require(doubles.length == 2 && ints.length == 1)
              new IntervalNumColDiscretizer(doubles.head, doubles.last, ints.head)
            case "cat" =>
              require(doubles.isEmpty)
              new CatColDiscretizer(ints.zipWithIndex.toMap)
            case "rank" =>
              require(doubles.isEmpty)
              new RankColDiscretizer(ints.toArray)
          }

          (i, col)
        }.collect().sortBy(_._1).unzip

    require(indices.length == indices.distinct.length)
    require(indices.length == indices.max + 1)

    val extraRow = extraDF.select("zeroAsMissing", "sparsity").head()
    val zeroAsMissing = extraRow.getBoolean(0)
    val sparsity = extraRow.getDouble(1)

    new Discretizer(colDiscretizers, zeroAsMissing, sparsity)
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
  * discretizer for one numerical column, each intervals are of same depth (quantile)
  *
  * @param splits splitting points
  */
private[gbm] class QuantileNumColDiscretizer(val splits: Array[Double]) extends ColDiscretizer {

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
  * discretizer for one numerical column, each intervals are of same length
  *
  * @param start   start point
  * @param step    length of each interval
  * @param numBins number of bins
  */
private[gbm] class IntervalNumColDiscretizer(val start: Double,
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
  * discretizer for one categorical column
  *
  * @param map bin mapping, from value to index of bin
  */
private[gbm] class CatColDiscretizer(val map: Map[Int, Int]) extends ColDiscretizer {

  override def transform(value: Double): Int = {
    require(value.toInt == value)
    map(value.toInt) + 1
  }

  override def numBins: Int = map.size
}


/**
  * discretizer for one ranking column
  *
  * @param array values
  */
private[gbm] class RankColDiscretizer(val array: Array[Int]) extends ColDiscretizer {

  override def transform(value: Double): Int = {
    require(value.toInt == value)
    val index = ju.Arrays.binarySearch(array, value.toInt)
    require(index >= 0, s"value $value not in ${array.mkString("(", ", ", ")")}")
    index + 1
  }

  override def numBins: Int = array.length
}


/**
  * aggregrator to build column discretizer
  */
private[gbm] trait ColAgg extends Serializable {

  def update(value: Double): ColAgg

  def merge(other: ColAgg): ColAgg

  def toColDiscretizer: ColDiscretizer

  def count: Long
}


/**
  * aggregrator for numerical column, find splits of same depth
  */
private[gbm] class QuantileNumColAgg(val maxBins: Int) extends ColAgg {
  require(maxBins >= 2)

  var summary = new QuantileSummaries(QuantileSummaries.defaultCompressThreshold, 0.001)
  var count = 0L

  override def update(value: Double): QuantileNumColAgg = {
    summary = summary.insert(value)
    count += 1
    this
  }

  override def merge(other: ColAgg): QuantileNumColAgg = {
    val o = other.asInstanceOf[QuantileNumColAgg]
    summary = summary.compress().merge(o.summary.compress())
    count += o.count
    this
  }

  // maxBins = 3 -> interval = 0.5, queries = [0.25, 0.75], splits = [q0.25, q0.75]
  override def toColDiscretizer: QuantileNumColDiscretizer = {
    if (count != 0) {
      summary = summary.compress()
      val interval = 1.0 / (maxBins - 1)
      val start = interval / 2
      val queries = Array.range(0, maxBins - 1).map(i => start + interval * i)
      val splits = queries.flatMap(summary.query).distinct.sorted
      new QuantileNumColDiscretizer(splits)
    } else {
      // all values in this column are missing value
      new QuantileNumColDiscretizer(Array.emptyDoubleArray)
    }
  }
}


/**
  * aggregrator for numerical column, find splits of same width
  */
private[gbm] class IntervalNumColAgg(val maxBins: Int) extends ColAgg {
  require(maxBins >= 2)

  var max = Double.MinValue
  var min = Double.MaxValue
  var count = 0L

  override def update(value: Double): IntervalNumColAgg = {
    max = math.max(max, value)
    min = math.min(min, value)
    count += 1
    this
  }

  override def merge(other: ColAgg): IntervalNumColAgg = {
    val o = other.asInstanceOf[IntervalNumColAgg]
    max = math.max(max, o.max)
    min = math.min(min, o.min)
    count += o.count
    this
  }

  // min = 0, max = 10, maxBins = 11, step = 10/10 = 1
  // if less than min+step/2 = 0.5 => 1, if greater than max-step/2 = 9.5 => 10
  override def toColDiscretizer: IntervalNumColDiscretizer = {
    if (count > 0) {
      val step = (max - min) / (maxBins - 1)
      val start = min + step / 2
      new IntervalNumColDiscretizer(start, step, maxBins)
    } else {
      // all values in this column are missing value
      new IntervalNumColDiscretizer(0.0, 0.0, 1)
    }
  }
}


/**
  * aggregrator for categorical column
  */
private[gbm] class CatColAgg(val maxBins: Int) extends ColAgg {
  require(maxBins >= 2)

  val counter = mutable.Map[Int, Long]()

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
    val map = array.zipWithIndex.toMap
    new CatColDiscretizer(map)
  }

  override def count: Long = counter.values.sum
}


/**
  * aggregrator for ranking column
  */
private[gbm] class RankAgg(val maxBins: Int) extends ColAgg {
  require(maxBins >= 2)

  val set = mutable.Set[Int]()
  var count = 0L

  override def update(value: Double): RankAgg = {
    require(value.toInt == value)
    set.add(value.toInt)
    require(set.size <= maxBins)
    count += 1
    this
  }

  override def merge(other: ColAgg): RankAgg = {
    val o = other.asInstanceOf[RankAgg]
    o.set.foreach { v =>
      set.add(v)
      require(set.size <= maxBins)
    }
    count += o.count
    this
  }

  override def toColDiscretizer: RankColDiscretizer = {
    val array = set.toArray.sorted
    new RankColDiscretizer(array)
  }
}

