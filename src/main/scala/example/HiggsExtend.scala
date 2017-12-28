package example

import scala.collection.mutable
import scala.util.Random

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg._

/**
  * spark-submit --class example.HiggsExtend --master yarn-client --driver-memory 8G --executor-memory 2G --num-executors 32 SparkGBM-0.0.1.jar
  */
object HiggsExtend {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("HiggsExtend")
      .getOrCreate()

    val sc = spark.sparkContext

    val data = MLUtils.loadLibSVMFile(sc, "/tmp/zrf/HIGGS", 28)

    val data2 = data.mapPartitionsWithIndex { case (index, iter) =>
      val rng = new Random(index)
      iter.map { case LabeledPoint(label, vec) =>
        val randValues = Array.fill(1000)(rng.nextDouble)
        val newVec = Vectors.dense(vec.toArray ++ randValues)
        LabeledPoint(label, newVec)
      }
    }

    val Array(train2, test2) = data2.randomSplit(Array(0.8, 0.2), 123L)
    MLUtils.saveAsLibSVMFile(train2.repartition(4096), "/tmp/zrf/HIGGS-DENSEEXT-Train")
    MLUtils.saveAsLibSVMFile(test2.repartition(1024), "/tmp/zrf/HIGGS-DENSEEXT-Test")

    val data3 = data.mapPartitionsWithIndex { case (index, iter) =>
      val rng = new Random(index)
      iter.map { case LabeledPoint(label, vec) =>
        val randIndices = mutable.Set.empty[Int]
        while (randIndices.size < 100) {
          randIndices.add(rng.nextInt(1000) + 28)
        }
        val randValues = Array.fill(100)(rng.nextDouble)

        val newVec = Vectors.sparse(1028, Array.range(0, 28) ++ randIndices.toArray.sorted, vec.toArray ++ randValues)
        LabeledPoint(label, newVec)
      }
    }

    val Array(train3, test3) = data3.randomSplit(Array(0.8, 0.2), 123L)
    MLUtils.saveAsLibSVMFile(train3.repartition(4096), "/tmp/zrf/HIGGS-SPARSEEXT-Train")
    MLUtils.saveAsLibSVMFile(test3.repartition(1024), "/tmp/zrf/HIGGS-SPARSEEXT-Test")

    spark.stop()
  }
}

