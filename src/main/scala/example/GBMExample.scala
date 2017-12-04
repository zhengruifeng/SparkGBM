package example

import org.apache.spark.ml.gbm._
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
  * Example of rdd-based API
  * spark-submit --class example.GBMExample target/SparkGBM-0.0.1.jar 2>log
  */
object GBMExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("GBMExample")
      .getOrCreate()

    spark.sparkContext.setLogLevel("INFO")

    val rdd = spark.read.format("libsvm")
      .load("data/housing_scale")
      .select("label", "features")
      .rdd.map { row =>
      (1.0, row.getDouble(0), row.getAs[Vector](1))
    }

    val Array(train, test) = rdd.randomSplit(Array(0.8, 0.2), seed = 123456)

    /** User defined objective function */
    val obj = new ObjFunc {
      override def compute(label: Double,
                           score: Double): (Double, Double) = (score - label, 1.0)

      override def name: String = "Another Square"
    }

    /** User defined evaluation function for R2 */
    val r2Eval = new BatchEvalFunc {
      override def compute(data: RDD[(Double, Double, Double)]): Double = {
        /** ignore weight */
        new RegressionMetrics(data.map(t => (t._2, t._3))).r2
      }

      override def isLargerBetter: Boolean = true

      override def name: String = "R2 (no weight)"
    }

    /** User defined evaluation function for MAE */
    val maeEval = new SimpleEvalFunc {
      override def compute(label: Double,
                           score: Double): Double = (label - score).abs

      override def isLargerBetter: Boolean = false

      override def name: String = "Another MAE"
    }

    /** User defined callback function */
    val lrUpdater = new CallbackFunc {
      override def compute(spark: SparkSession,
                           boostConfig: BoostConfig,
                           model: GBMModel,
                           trainMetrics: Array[Map[String, Double]],
                           testMetrics: Array[Map[String, Double]]): Boolean = {
        /** learning rate decay */
        if (boostConfig.getStepSize > 0.01) {
          boostConfig.updateStepSize(boostConfig.getStepSize * 0.95)
        }

        println(s"Round ${model.numTrees}: train metrics: ${trainMetrics.last}")
        if (testMetrics.nonEmpty) {
          println(s"Round ${model.numTrees}: test metrics: ${testMetrics.last}")
        }
        false
      }

      override def name: String = "Learning Rate Updater"
    }

    val recoder = new MetricRecoder


    val gbm = new GBM
    gbm.setMaxIter(15)
      .setMaxDepth(5)
      .setStepSize(0.2)
      .setNumericalBinType("depth")
      .setObjectiveFunc(obj)
      .setEvaluateFunc(Array(r2Eval, maeEval))
      .setCallbackFunc(Array(lrUpdater, recoder))

    /** train with validation */
    val model = gbm.fit(train, test)

    recoder.testMetricsRecoder.zipWithIndex.foreach { case (metrics, iter) =>
      println(s"iter $iter, test metrics $metrics")
    }

    /** model save and load */
    val path = s"/tmp/SparkGBM/model-${System.currentTimeMillis}"
    model.save(path)
    val model2 = GBMModel.load(path)

    println(s"weights of trees: ${model.weights.mkString(",")}")

    /** label and score */
    val trainResult = train.map {
      case (_, label, features) =>
        (label, model.predict(features))
    }
    val trainR2 = new RegressionMetrics(trainResult).r2
    println(s"R2 on train data $trainR2")

    val testResult = test.map {
      case (_, label, features) =>
        (label, model.predict(features))
    }
    val testR2 = new RegressionMetrics(testResult).r2
    println(s"R2 on test data $testR2")

    spark.stop()
  }
}


