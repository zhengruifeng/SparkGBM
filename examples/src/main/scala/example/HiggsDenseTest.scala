package example

import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

/**
  * spark-submit --class example.HiggsDenseTest --master yarn-client --driver-memory 8G --executor-memory 2G --num-executors 32 --jars gbm/target/gbm-0.0.1.jar examples/target/examples-0.0.1.jar 2>log
  */
object HiggsDenseTest {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("HiggsDenseTest")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    sc.setLogLevel("INFO")
    sc.setCheckpointDir("/tmp/zrf/spark-checkpoint")

    val modelCheckpointPath = s"/tmp/zrf/spark-modelcheckpoint-${System.nanoTime}"

    val train = MLUtils.loadLibSVMFile(sc, "/tmp/zrf/HIGGS-DENSEEXT-Train", 1028)
      .map(l => (l.label, l.features.asML))
      .toDF("label", "features")

    val test = MLUtils.loadLibSVMFile(sc, "/tmp/zrf/HIGGS-DENSEEXT-Test", 1028)
      .map(l => (l.label, l.features.asML))
      .toDF("label", "features")

    val evaluator = new MulticlassClassificationEvaluator()
    evaluator.setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val gbmcStart = System.nanoTime

    val gbmc = new GBMClassifier
    gbmc.setBoostType("gbtree")
      .setStepSize(0.1)
      .setMaxIter(20)
      .setMaxDepth(5)
      .setMaxLeaves(128)
      .setMaxBins(128)
      .setMinNodeHess(1.0)
      .setSubSample(0.8)
      .setColSampleByTree(1.0)
      .setColSampleByLevel(1.0)
      .setRegAlpha(0.1)
      .setRegLambda(1.0)
      .setObjectiveFunc("logistic")
      .setEvaluateFunc(Array.empty)
      .setFloatType("float")
      .setCheckpointInterval(10)
      .setModelCheckpointInterval(10)
      .setModelCheckpointPath(modelCheckpointPath)
      .setPredictionCol("prediction")
      .setReduceParallelism(-2)

    val gbmcModel = gbmc.fit(train)

    val gbmcEnd = System.nanoTime

    val gbmcF1 = evaluator.evaluate(gbmcModel.transform(test))
    println(s"GBM finished, duration: ${(gbmcEnd - gbmcStart) / 1e9} seconds, F1 on test data: $gbmcF1")

    val gbtcStart = System.nanoTime

    val gbtc = new GBTClassifier
    gbtc.setStepSize(0.1)
      .setMaxIter(20)
      .setMaxDepth(5)
      .setSubsamplingRate(0.8)
      .setMaxBins(128)
      .setMinInstancesPerNode(1)
      .setCheckpointInterval(10)
      .setCacheNodeIds(true)
      .setPredictionCol("prediction")

    val gbtcModel = gbtc.fit(train)

    val gbtcEnd = System.nanoTime

    val gbtcF1 = evaluator.evaluate(gbtcModel.transform(test))
    println(s"GBT finished, duration: ${(gbtcEnd - gbtcStart) / 1e9} seconds, F1 on test data: $gbtcF1")

    spark.stop()
  }
}
