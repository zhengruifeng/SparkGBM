package example

import org.apache.spark.ml.classification.GBMClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

/**
  * spark-submit --class example.HiggsTest --master yarn-client --driver-memory 8G --executor-memory 2G --num-executors 32 SparkGBM-0.0.1.jar 2>log
  */
object HiggsTest {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("HiggsTest")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    sc.setLogLevel("WARN")
    sc.setCheckpointDir("/tmp/zrf/spark-checkpoint")

    val modelCheckpointPath = s"/tmp/zrf/spark-modelcheckpoint-${System.nanoTime}"

    val train = MLUtils.loadLibSVMFile(sc, "/tmp/zrf/HIGGS-EXT-Train", 1028)
      .map { l => (l.label, l.features.asML) }
      .toDF("label", "features")

    val test = MLUtils.loadLibSVMFile(sc, "/tmp/zrf/HIGGS-EXT-Test", 1028)
      .map { l => (l.label, l.features.asML) }
      .toDF("label", "features")

    val gbmc = new GBMClassifier
    gbmc.setBoostType("gbtree")
      .setStepSize(0.1)
      .setMaxIter(50)
      .setMaxDepth(5)
      .setMaxLeaves(128)
      .setMaxBins(128)
      .setSubSample(0.8)
      .setColSampleByTree(0.8)
      .setColSampleByLevel(0.8)
      .setRegAlpha(0.1)
      .setRegLambda(1.0)
      .setObjectiveFunc("logistic")
      .setEvaluateFunc(Array.empty)
      .setFloatType("float")
      .setCheckpointInterval(10)
      .setModelCheckpointInterval(10)
      .setModelCheckpointPath(modelCheckpointPath)

    val model = gbmc.fit(train)

    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val auc = evaluator.evaluate(model.transform(test))
    println(s"AUC on test data $auc")

    spark.stop()
  }
}
