package org.apache.spark.ml.gbm

import org.apache.hadoop.fs.Path

import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.DefaultParamsWriter
import org.apache.spark.sql.SparkSession

/**
  * trait for callback function, will be called after each iteration
  */
trait CallbackFunc extends Logging with Serializable {

  /**
    * call back function
    *
    * @param boostConfig  boosting configuration, be careful to update it
    * @param model        snapshot of current model
    * @param metrics      metrics names and whether it is better for a larger value
    * @param trainMetrics training metric
    * @param testMetrics  validation metric
    * @return whether to stop training
    */
  def compute(boostConfig: BoostConfig,
              model: GBMModel,
              metrics: Map[String, Boolean],
              trainMetrics: Array[Map[String, Double]],
              testMetrics: Array[Map[String, Double]]): Boolean

  def name: String
}

/**
  * early stopping function
  *
  * @param iters the interval to stop training if one metric on validation data doesn't improve
  */
class EarlyStopFunc(val iters: Int) extends CallbackFunc {
  require(iters >= 1)

  def this() = this(10)

  override def compute(boostConfig: BoostConfig,
                       model: GBMModel,
                       metrics: Map[String, Boolean],
                       trainMetrics: Array[Map[String, Double]],
                       testMetrics: Array[Map[String, Double]]): Boolean = {
    var stop = false

    if (testMetrics.length > iters) {
      val len = iters + 1
      metrics.foreach { case (name, isLargerBetter) =>
        val values = testMetrics.takeRight(len).map(_ (name))
        val start = values.head
        val end = values.last

        if (isLargerBetter && start >= end) {
          logDebug(s"Fail to increase metric $name in the last $len iterations: ${values.mkString("(", ",", ")")}")
          stop = true
        } else if (!isLargerBetter && start <= end) {
          logDebug(s"Fail to decrease metric $name in the last $len iterations: ${values.mkString("(", ",", ")")}")
          stop = true
        }
      }
    }

    stop
  }

  override def name = "EarlyStop"
}


/**
  * model checkpoint function
  *
  * @param interval the interval between checkpoints
  * @param path     the path to save models
  */
class ModelCheckpointFunc(val interval: Int,
                          val path: String) extends CallbackFunc {

  override def compute(boostConfig: BoostConfig,
                       model: GBMModel,
                       metrics: Map[String, Boolean],
                       trainMetrics: Array[Map[String, Double]],
                       testMetrics: Array[Map[String, Double]]): Boolean = {
    if (model.numTrees % interval == 0) {
      val start = System.nanoTime()
      val currentPath = new Path(path, s"model-${model.numTrees}").toString
      GBMModel.save(model, currentPath)
      logDebug(s"Model checkpoint finish, duration ${(System.nanoTime() - start) / 1e9} seconds")
    }

    false
  }

  override def name = "ModelCheckpoint"
}

/**
  * model checkpoint function for GBMClassificationModel
  *
  * @param interval the interval between checkpoints
  * @param path     the path to save models
  * @param params   meta params to save
  */
class ClassificationModelCheckpointFunc(val interval: Int,
                                        val path: String,
                                        val params: Params) extends CallbackFunc {

  override def compute(boostConfig: BoostConfig,
                       model: GBMModel,
                       metrics: Map[String, Boolean],
                       trainMetrics: Array[Map[String, Double]],
                       testMetrics: Array[Map[String, Double]]): Boolean = {
    if (model.numTrees % interval == 0) {
      val start = System.nanoTime()

      val spark = SparkSession.builder().getOrCreate()

      val currentPath = new Path(path, s"model-${model.numTrees}").toString

      DefaultParamsWriter.saveMetadata(params, currentPath, spark.sparkContext, None)

      GBMModel.save(model, currentPath)

      val otherDF = spark.createDataFrame(Seq(
        ("type", "classification"),
        ("time", System.currentTimeMillis.toString))).toDF("key", "value")
      val otherPath = new Path(currentPath, "other").toString
      otherDF.write.parquet(otherPath)

      logDebug(s"Model checkpoint finish, duration ${(System.nanoTime() - start) / 1e9} seconds")
    }

    false
  }

  override def name = "ClassificationModelCheckpoint"
}


/**
  * model checkpoint function for GBMRegressionModel
  *
  * @param interval the interval between checkpoints
  * @param path     the path to save models
  * @param params   meta params to save
  */
class RegressionModelCheckpointFunc(val interval: Int,
                                    val path: String,
                                    val params: Params) extends CallbackFunc {

  override def compute(boostConfig: BoostConfig,
                       model: GBMModel,
                       metrics: Map[String, Boolean],
                       trainMetrics: Array[Map[String, Double]],
                       testMetrics: Array[Map[String, Double]]): Boolean = {
    if (model.numTrees % interval == 0) {
      val start = System.nanoTime()

      val spark = SparkSession.builder().getOrCreate()

      val currentPath = new Path(path, s"model-${model.numTrees}").toString

      DefaultParamsWriter.saveMetadata(params, currentPath, spark.sparkContext, None)

      GBMModel.save(model, currentPath)

      val otherDF = spark.createDataFrame(Seq(
        ("type", "regression"),
        ("time", System.currentTimeMillis.toString))).toDF("key", "value")
      val otherPath = new Path(currentPath, "other").toString
      otherDF.write.parquet(otherPath)

      logDebug(s"Model checkpoint finish, duration ${(System.nanoTime() - start) / 1e9} seconds")
    }

    false
  }

  override def name = "RegressionModelCheckpointFunc"
}
