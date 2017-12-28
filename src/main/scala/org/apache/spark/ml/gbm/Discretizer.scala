package org.apache.spark.ml.gbm

import java.{util => ju}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.{specialized => spec}

import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.catalyst.util.QuantileSummaries
import org.apache.spark.util.collection.{BitSet, OpenHashMap}


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

  def transform(vec: Vector): Array[Int] = {
    require(vec.size == numCols)

    val bins = Array.ofDim[Int](numCols)

    if (zeroAsMissing) {
      Utils.getActiveIter(vec).foreach { case (i, v) =>
        bins(i) = discretizeWithIndex(v, i)
      }

    } else {
      Utils.getTotalIter(vec).foreach { case (i, v) =>
        bins(i) = discretizeWithIndex(v, i)
      }
    }

    bins
  }


  private[gbm] def transform[@spec(Byte, Short, Int) B: Integral : ClassTag](data: RDD[(Double, Double, Vector)]): RDD[(Double, Double, BinVector[B])] = {
    val intB: Integral[B] = implicitly[Integral[B]]

    val getIter = if (zeroAsMissing) {
      (vec: Vector) => Utils.getActiveIter(vec)
    } else {
      (vec: Vector) => Utils.getTotalIter(vec)
    }

    data.mapPartitions { it =>

      val indexBuff = mutable.ArrayBuffer.empty[Int]
      val valueBuff = mutable.ArrayBuffer.empty[B]

      it.map { case (weight, label, vec) =>
        indexBuff.clear
        valueBuff.clear

        getIter(vec).foreach { case (i, v) =>
          val bin = discretizeWithIndex(v, i)
          if (bin != 0) {
            indexBuff.append(i)
            valueBuff.append(intB.fromInt(bin))
          }
        }

        val bins = BinVector.sparse[B](numCols, indexBuff.toArray, valueBuff.toArray).compress
        (weight, label, bins)
      }
    }
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
          catCols: BitSet,
          rankCols: BitSet,
          maxBins: Int,
          numericalBinType: String,
          zeroAsMissing: Boolean,
          depth: Int): Discretizer = {
    require(maxBins >= 4)
    require(numCols >= 1)

    ColAgg.registerKryoClasses(vectors.sparkContext)

    val start = System.nanoTime
    logInfo(s"Discretizer building start")

    // zero bin index is always reserved for missing value
    val emptyAggs = Array.range(0, numCols).map { col =>
      if (col < catCols.capacity && catCols.get(col)) {
        new CatColAgg(maxBins - 1)
      } else if (col < rankCols.capacity && rankCols.get(col)) {
        new RankColAgg(maxBins - 1)
      } else if (numericalBinType == GBM.Depth) {
        new QuantileNumColAgg(maxBins - 1)
      } else {
        new IntervalNumColAgg(maxBins - 1)
      }
    }

    val (count, aggregated) =
      vectors.mapPartitions { it =>
        var cnt = 0L
        val aggs = emptyAggs
        val nans = new OpenHashMap[Int, Long]()


        // only absorb non-zero values
        it.foreach { vec =>
          require(vec.size == numCols)

          Utils.getActiveIter(vec).foreach { case (i, v) =>
            if (!v.isNaN && !v.isInfinity) {
              aggs(i).update(v)
            } else if (!zeroAsMissing) {
              nans.changeValue(i, 1L, _ + 1L)
            }
          }
          cnt += 1
        }

        // if zero is not missing, add zeros back
        if (!zeroAsMissing) {
          var i = 0
          while (i < numCols) {
            nans.toMap
            val nz = cnt - aggs(i).count - nans(i)
            aggs(i).updateZeros(nz)
            i += 1
          }
        }

        Iterator.single((cnt, aggs))

      }.treeReduce(f = {
        case ((cnt1, aggs1), (cnt2, aggs2)) =>
          var i = 0
          while (i < numCols) {
            aggs1(i).merge(aggs2(i))
            i += 1
          }
          (cnt1 + cnt2, aggs1)
      }, depth = depth)


    // number of non-missing
    val nnm = aggregated.map(_.count.toDouble).sum

    val sparsity = 1 - nnm / count / numCols

    logInfo(s"Discretizer building finished, sparsity: $sparsity, duration: ${(System.nanoTime - start) / 1e9} sec")

    new Discretizer(aggregated.map(_.toColDiscretizer), zeroAsMissing, sparsity)
  }


  /**
    * Comupte the proportion of missing value
    */
  def computeSparsity(data: RDD[Vector],
                      numCols: Int,
                      zeroAsMissing: Boolean,
                      depth: Int): Double = {

    // compute number of non-missing for each row
    val countNNM = if (zeroAsMissing) {
      (vec: Vector) => {
        Utils.getActiveIter(vec).count { case (i, v) =>
          !v.isNaN && !v.isInfinity
        }
      }

    } else {

      (vec: Vector) => {
        vec.size - Utils.getActiveIter(vec).count { case (i, v) =>
          v.isNaN || v.isInfinity
        }
      }
    }

    val (_, nnm) = data.treeAggregate[(Long, Double)]((0L, 0.0))(
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

  private[this] var kryoRegistered: Boolean = false

  def registerKryoClasses(sc: SparkContext): Unit = {
    if (!kryoRegistered) {
      sc.getConf.registerKryoClasses(
        Array(classOf[ColDiscretizer],
          classOf[Array[ColDiscretizer]],
          classOf[QuantileNumColDiscretizer],
          classOf[IntervalNumColDiscretizer],
          classOf[CatColDiscretizer],
          classOf[RankColDiscretizer])
      )
      kryoRegistered = true
    }
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
  * discretizer for one numerical column, all intervals are of the same length
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

  def updateZeros(nz: Long): ColAgg

  def merge(other: ColAgg): ColAgg

  def toColDiscretizer: ColDiscretizer

  def count: Long
}


private[gbm] object ColAgg {

  private[this] var kryoRegistered: Boolean = false

  def registerKryoClasses(sc: SparkContext): Unit = {
    if (!kryoRegistered) {
      sc.getConf.registerKryoClasses(
        Array(classOf[ColAgg],
          classOf[Array[ColAgg]],
          classOf[(Long, Array[ColAgg])],
          classOf[QuantileNumColAgg],
          classOf[IntervalNumColAgg],
          classOf[CatColAgg],
          classOf[RankColAgg])
      )
      kryoRegistered = true
    }
  }
}


/**
  * aggregrator for numerical column, find splits of same depth
  */
private[gbm] class QuantileNumColAgg(val maxBins: Int) extends ColAgg {
  require(maxBins >= 2)

  var count = 0L
  var summary = QuantileNumColAgg.createSummary

  override def update(value: Double): QuantileNumColAgg = {
    summary = summary.insert(value)
    count += 1
    this
  }

  override def updateZeros(nz: Long): QuantileNumColAgg = {
    if (nz > 0) {
      val nzSummary = QuantileNumColAgg.createNZSummary(nz)
      summary = summary.compress.merge(nzSummary).compress
      count += nz
    }
    this
  }

  override def merge(other: ColAgg): QuantileNumColAgg = {
    val o = other.asInstanceOf[QuantileNumColAgg]
    summary = summary.compress.merge(o.summary.compress).compress
    count += o.count
    this
  }

  // maxBins = 3 -> interval = 0.5, queries = [0.25, 0.75], splits = [q0.25, q0.75]
  override def toColDiscretizer: QuantileNumColDiscretizer = {
    if (count != 0) {
      summary = summary.compress
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


private[gbm] object QuantileNumColAgg {

  val compressThreshold = QuantileSummaries.defaultCompressThreshold

  val relativeError = 0.01

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
private[gbm] class IntervalNumColAgg(val maxBins: Int) extends ColAgg {
  require(maxBins >= 2)

  var count = 0L
  var max = Double.MinValue
  var min = Double.MaxValue

  override def update(value: Double): IntervalNumColAgg = {
    max = math.max(max, value)
    min = math.min(min, value)
    count += 1
    this
  }

  override def updateZeros(nz: Long): IntervalNumColAgg = {
    if (nz > 0) {
      max = math.max(max, 0.0)
      min = math.min(min, 0.0)
      count += nz
    }
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

  val counter = new OpenHashMap[Int, Long](maxBins >> 1)

  override def update(value: Double): CatColAgg = {
    require(value.toInt == value)
    counter.changeValue(value.toInt, 1L, _ + 1L)
    require(counter.size <= maxBins)
    this
  }

  override def updateZeros(nz: Long): CatColAgg = {
    if (nz > 0) {
      counter.changeValue(0, nz, _ + nz)
      require(counter.size <= maxBins)
    }
    this
  }

  override def merge(other: ColAgg): CatColAgg = {
    other.asInstanceOf[CatColAgg].counter
      .foreach { case (v, c) =>
        counter.changeValue(v, c, _ + c)
        require(counter.size <= maxBins)
      }
    this
  }

  override def toColDiscretizer: CatColDiscretizer = {
    val array = counter.toArray.sortBy(_._2).map(_._1).reverse
    val map = array.zipWithIndex.toMap
    new CatColDiscretizer(map)
  }

  override def count: Long = counter.iterator.map(_._2).sum
}


/**
  * aggregrator for ranking column
  */
private[gbm] class RankColAgg(val maxBins: Int) extends ColAgg {
  require(maxBins >= 2)

  var count = 0L
  val set = mutable.Set.empty[Int]
  set.sizeHint(maxBins >> 1)

  override def update(value: Double): RankColAgg = {
    require(value.toInt == value)
    set.add(value.toInt)
    require(set.size <= maxBins)
    count += 1
    this
  }

  override def updateZeros(nz: Long): RankColAgg = {
    if (nz > 0) {
      set.add(0)
      require(set.size <= maxBins)
      count += nz
    }
    this
  }

  override def merge(other: ColAgg): RankColAgg = {
    val o = other.asInstanceOf[RankColAgg]
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

