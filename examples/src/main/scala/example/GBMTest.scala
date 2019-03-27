package example

import scala.io.Source
import scala.collection.mutable

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

import org.apache.spark.ml.classification._
import org.apache.spark.ml.param._
import org.apache.spark.ml.regression._
import org.apache.spark.sql.SparkSession


/**
  * spark-submit --class example.GBMTest --jars spark-gbm/target/spark-gbm-2.3.0.jar examples/target/examples-2.3.0.jar examples/src/main/resources/task.config 2>log
  */
object GBMTest {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("GBMTest")
      .getOrCreate

    spark.sparkContext.setLogLevel("DEBUG")
    spark.sparkContext.setCheckpointDir(s"/tmp/sparkGBM/${spark.sparkContext.applicationId}/checkpoints/")


    println(s"args: ${args.mkString(",")}")

    val config = loadConfig(args(0))


    val loadInputFile = config.inputType match {

      case "parquet" =>
        path: String =>
          spark.read.parquet(path)

      case "libsvm" =>
        path: String =>
          spark.read.format("libsvm").load(path)

      case tpe =>
        val numFeatures = tpe.substring(7).toInt
        path: String =>
          spark.read.format("libsvm")
            .option("numFeatures", numFeatures).load(path)
    }


    val trainDF = loadInputFile(config.trainPath)

    val testDF = if (config.testPath.nonEmpty) {
      loadInputFile(config.testPath)
    } else {
      null
    }


    config.task match {

      case "classification" =>
        val gbmc = new GBMClassifier
        updateLearner(gbmc, config.params)

        val model = if (config.testPath.nonEmpty) {
          gbmc.fit(trainDF, testDF)
        } else {
          gbmc.fit(trainDF)
        }

        model.write.overwrite().save(config.modelSavePath)


      case "regression" =>
        val gbmr = new GBMRegressor
        updateLearner(gbmr, config.params)

        val model = if (config.testPath.nonEmpty) {
          gbmr.fit(trainDF, testDF)
        } else {
          gbmr.fit(trainDF)
        }

        model.write.overwrite().save(config.modelSavePath)
    }

    spark.stop()
  }


  case class Config(task: String,
                    inputType: String,
                    trainPath: String,
                    testPath: String,
                    modelSavePath: String,
                    params: Map[String, String])


  def loadConfig(path: String): Config = {
    val params = mutable.OpenHashMap.empty[String, String]

    val fs = FileSystem.get(new Configuration())

    var task = "classification"
    var inputType = "parquet"
    var trainPath = ""
    var testPath = ""
    var modelSavePath = ""

    val lineIter = Source.fromInputStream(fs.open(new Path(path))).getLines()
    while (lineIter.hasNext) {
      val line = lineIter.next().trim
      if (line.nonEmpty && !line.startsWith("#")) {
        val Array(k, v) = line.split("=")
        k match {
          case "task" =>
            require(v.startsWith(""""""") && v.endsWith("""""""))
            task = v.substring(1, v.length - 1)

          case "inputType" =>
            require(v.startsWith(""""""") && v.endsWith("""""""))
            inputType = v.substring(1, v.length - 1)

          case "trainPath" =>
            require(v.startsWith(""""""") && v.endsWith("""""""))
            trainPath = v.substring(1, v.length - 1)

          case "testPath" =>
            require(v.startsWith(""""""") && v.endsWith("""""""))
            if (v.length > 2) {
              testPath = v.substring(1, v.length - 1)
            }

          case "modelSavePath" =>
            require(v.startsWith(""""""") && v.endsWith("""""""))
            modelSavePath = v.substring(1, v.length - 1)

          case _ => params.update(k, v)
        }
      }
    }

    require(task == "classification" || task == "regression")
    require(inputType == "parquet" || inputType == "libsvm" ||
      (inputType.startsWith("libsvm@") && inputType.substring(7).toInt > 0))
    require(trainPath.nonEmpty)
    require(modelSavePath.nonEmpty)

    Config(task, inputType, trainPath, testPath, modelSavePath, params.toMap)
  }


  def updateLearner(learner: Params,
                    params: Map[String, String]): Unit = {
    val paramIter = params.iterator
    while (paramIter.hasNext) {
      val (k, v) = paramIter.next()
      val param = learner.getParam(k)
      learner.set(param, param.jsonDecode(v))
    }
  }
}
