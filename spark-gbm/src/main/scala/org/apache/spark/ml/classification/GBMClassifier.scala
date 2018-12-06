package org.apache.spark.ml.classification

import scala.collection.mutable

import org.apache.hadoop.fs.Path

import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.gbm.func._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasThreshold
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.storage.StorageLevel


trait GBMClassificationParams extends GBMParams with HasThreshold {

  setDefault(threshold -> 0.5)

  /**
    * Learning objective function.
    *
    * @group expertParam
    */
  final val objectiveFunc: Param[String] =
    new Param[String](this, "objectiveFunc", "Learning objective function",
      (obj: String) => GBMClassifier.supportedObjs.contains(obj))

  def getObjectiveFunc: String = $(objectiveFunc)

  setDefault(objectiveFunc -> GBMClassifier.LogisticObj)

  /**
    * Metric functions.
    *
    * @group expertParam
    */
  final val evaluateFunc: StringArrayParam =
    new StringArrayParam(this, "evaluateFunc", "Metric functions",
      (evals: Array[String]) => evals.forall(GBMClassifier.supportedEvals.contains))

  def getEvaluateFunc: Array[String] = $(evaluateFunc)

  setDefault(evaluateFunc -> Array(GBMClassifier.LogLossEval, GBMClassifier.AUROCEval, GBMClassifier.ErrorEval))
}


class GBMClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, GBMClassifier, GBMClassificationModel]
    with GBMClassificationParams with DefaultParamsWritable with Logging {

  def this() = this(Identifiable.randomUID("gbmc"))

  def setThreshold(value: Double): this.type = set(threshold, value)

  def setLeafCol(value: String): this.type = set(leafCol, value)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setStepSize(value: Double): this.type = set(stepSize, value)

  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  def setMaxBins(value: Int): this.type = set(maxBins, value)

  def setMaxLeaves(value: Int): this.type = set(maxLeaves, value)

  def setBaseScore(value: Array[Double]): this.type = set(baseScore, value)

  def setMinNodeHess(value: Double): this.type = set(minNodeHess, value)

  def setRegAlpha(value: Double): this.type = set(regAlpha, value)

  def setRegLambda(value: Double): this.type = set(regLambda, value)

  def setCatCols(value: Array[Int]): this.type = set(catCols, value)

  def setRankCols(value: Array[Int]): this.type = set(rankCols, value)

  def setSubSampleRate(value: Double): this.type = set(subSampleRate, value)

  def setSubSampleRateByLevel(value: Double): this.type = set(subSampleRateByLevel, value)

  def setColSampleRateByTree(value: Double): this.type = set(colSampleRateByTree, value)

  def setColSampleRateByLevel(value: Double): this.type = set(colSampleRateByLevel, value)

  def setMinGain(value: Double): this.type = set(minGain, value)

  def setStorageLevel1(value: String): this.type = set(storageLevel1, value)

  def setStorageLevel2(value: String): this.type = set(storageLevel2, value)

  def setStorageLevel3(value: String): this.type = set(storageLevel3, value)

  def setObjectiveFunc(value: String): this.type = set(objectiveFunc, value)

  def setEvaluateFunc(value: Array[String]): this.type = set(evaluateFunc, value)

  def setParallelismType(value: String): this.type = set(parallelismType, value)

  def setGreedierSearch(value: Boolean): this.type = set(greedierSearch, value)

  def setBoostType(value: String): this.type = set(boostType, value)

  def setDropRate(value: Double): this.type = set(dropRate, value)

  def setDropSkip(value: Double): this.type = set(dropSkip, value)

  def setMinDrop(value: Int): this.type = set(minDrop, value)

  def setMaxDrop(value: Int): this.type = set(maxDrop, value)

  def setTopRate(value: Double): this.type = set(topRate, value)

  def setOtherRate(value: Double): this.type = set(otherRate, value)

  def setInitialModelPath(value: String): this.type = set(initialModelPath, value)

  def setMaxBruteBins(value: Int): this.type = set(maxBruteBins, value)

  def setEarlyStopIters(value: Int): this.type = set(earlyStopIters, value)

  def setModelCheckpointInterval(value: Int): this.type = set(modelCheckpointInterval, value)

  def setModelCheckpointPath(value: String): this.type = set(modelCheckpointPath, value)

  def setNumericalBinType(value: String): this.type = set(numericalBinType, value)

  def setEnableOneHot(value: Boolean): this.type = set(enableOneHot, value)

  def setFirstTrees(value: Int): this.type = set(firstTrees, value)

  def setFloatType(value: String): this.type = set(floatType, value)

  def setZeroAsMissing(value: Boolean): this.type = set(zeroAsMissing, value)

  def setReduceParallelism(value: Double): this.type = set(reduceParallelism, value)

  def setSubSampleType(value: String): this.type = set(subSampleType, value)

  def setHistogramComputationType(value: String): this.type = set(histogramComputationType, value)

  def setBaseModelParallelism(value: Int): this.type = set(baseModelParallelism, value)

  def setBlockSize(value: Int): this.type = set(blockSize, value)

  def setImportanceType(value: String): this.type = set(importanceType, value)

  override def fit(dataset: Dataset[_]): GBMClassificationModel = {
    fit(dataset, None)
  }

  def fit(dataset: Dataset[_],
          testDataset: Dataset[_]): GBMClassificationModel = {
    fit(dataset, Some(testDataset))
  }

  private[ml] def fit(dataset: Dataset[_],
                      testDataset: Option[Dataset[_]]): GBMClassificationModel = instrumented { instr =>
    transformSchema(dataset.schema, logging = true)

    if ($(parallelismType) == "feature") {
      require(dataset.sparkSession.sparkContext.getCheckpointDir.nonEmpty)
    }
    require($(maxDrop) >= $(minDrop))

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, params: _*)

    val w = if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      col($(weightCol)).cast(DoubleType)
    } else {
      lit(1.0)
    }

    val labelWeights = dataset
      .groupBy(col($(labelCol)).cast(IntegerType)).agg(sum(w))
      .collect().map { row => (row.getInt(0), row.getDouble(1)) }
      .toMap

    val numClasses = labelWeights.size
    require(labelWeights.keys.forall(i => i >= 0 && i < numClasses))

    val objFunc: ObjFunc =
      $(objectiveFunc) match {
        case GBMClassifier.LogisticObj =>
          require(numClasses == 2)
          new LogisticObj

        case GBMClassifier.SoftmaxObj =>
          new SoftmaxObj(numClasses)
      }

    val evalFunc: Array[EvalFunc] =
      $(evaluateFunc).map {
        case GBMClassifier.LogLossEval =>
          new LogLossEval
        case GBMClassifier.AUROCEval =>
          new AUROCEval
        case GBMClassifier.AUPRCEval =>
          new AUPRCEval
        case GBMClassifier.ErrorEval =>
          new ErrorEval($(threshold))
      }

    val callBackFunc = mutable.ArrayBuffer.empty[CallbackFunc]
    if ($(earlyStopIters) >= 1) {
      callBackFunc.append(new EarlyStop($(earlyStopIters)))
    }
    if ($(modelCheckpointInterval) >= 1 && $(modelCheckpointPath).nonEmpty) {
      val mockModel = copyValues(new GBMClassificationModel(uid, null, numClasses).setParent(this))
      callBackFunc.append(new ClassificationModelCheckpoint($(modelCheckpointInterval), $(modelCheckpointPath), mockModel))
    }

    val initialModel =
      if (isDefined(initialModelPath) && $(initialModelPath).nonEmpty) {
        val model = GBMClassificationModel.load($(initialModelPath))
        if (model.getObjectiveFunc != $(objectiveFunc)) {
          logWarning(s"The objective function conflicts with that in initial model," +
            s" objective of initial model ${model.getObjectiveFunc} will be ignored.")
        }
        Some(model.model)
      } else {
        None
      }

    val gbm = new GBM
    gbm.setSparkSession(dataset.sparkSession)
      .setMaxIter($(maxIter))
      .setMaxDepth($(maxDepth))
      .setMaxLeaves($(maxLeaves))
      .setMaxBins($(maxBins))
      .setMinGain($(minGain))
      .setMinNodeHess($(minNodeHess))
      .setBaseScore($(baseScore))
      .setStepSize($(stepSize))
      .setRegAlpha($(regAlpha))
      .setRegLambda($(regLambda))
      .setObjFunc(objFunc)
      .setEvalFunc(evalFunc)
      .setCallbackFunc(callBackFunc.toArray)
      .setCatCols($(catCols).toSet)
      .setRankCols($(rankCols).toSet)
      .setSubSampleRate($(subSampleRate))
      .setSubSampleRateByLevel($(subSampleRateByLevel))
      .setColSampleRateByTree($(colSampleRateByTree))
      .setColSampleRateByLevel($(colSampleRateByLevel))
      .setCheckpointInterval($(checkpointInterval))
      .setStorageLevel1(StorageLevel.fromString($(storageLevel1)))
      .setStorageLevel2(StorageLevel.fromString($(storageLevel2)))
      .setStorageLevel3(StorageLevel.fromString($(storageLevel3)))
      .setAggregationDepth($(aggregationDepth))
      .setSeed($(seed))
      .setParallelismType($(parallelismType))
      .setGreedierSearch($(greedierSearch))
      .setBoostType($(boostType))
      .setDropRate($(dropRate))
      .setDropSkip($(dropSkip))
      .setMinDrop($(minDrop))
      .setMaxDrop($(maxDrop))
      .setTopRate($(topRate))
      .setOtherRate($(otherRate))
      .setMaxBruteBins($(maxBruteBins))
      .setNumericalBinType($(numericalBinType))
      .setZeroAsMissing($(zeroAsMissing))
      .setReduceParallelism($(reduceParallelism))
      .setSubSampleType($(subSampleType))
      .setHistogramComputationType($(histogramComputationType))
      .setBaseModelParallelism($(baseModelParallelism))
      .setBlockSize($(blockSize))
      .setInitialModel(initialModel)

    val transLabel = if ($(objectiveFunc) == GBMClassifier.LogisticObj) {
      label: Double => Array(label)
    } else {
      label: Double =>
        require(label == label.toInt)
        val array = Array.ofDim[Double](numClasses)
        array(label.toInt) = 1.0F
        array
    }

    val data = dataset.select(w, col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd
      .map { row => (row.getDouble(0), transLabel(row.getDouble(1)), row.getAs[Vector](2)) }

    val test = testDataset.map { data =>
      data.select(w, col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd
        .map { row => (row.getDouble(0), transLabel(row.getDouble(1)), row.getAs[Vector](2)) }
    }

    val gbmModel = gbm.fit(data, test)

    val model = new GBMClassificationModel(uid, gbmModel, numClasses)
    instr.logSuccess()
    copyValues(model.setParent(this))
  }

  override def copy(extra: ParamMap): GBMClassifier = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): GBMClassificationModel = {
    throw new NotImplementedError(s"train is not implemented for ${this.getClass}.")
  }
}

object GBMClassifier extends DefaultParamsReadable[GBMClassifier] {
  override def load(path: String): GBMClassifier = super.load(path)

  /** String name for LogisticObj */
  private[classification] val LogisticObj: String = "logistic"

  /** String name for SoftmaxObj */
  private[classification] val SoftmaxObj: String = "softmax"

  /** Set of objective functions that GBMClassifier supports */
  private[classification] val supportedObjs = Set(LogisticObj, SoftmaxObj)

  /** String name for LogLossEval */
  private[classification] val LogLossEval: String = "logloss"

  /** String name for AUROCEval */
  private[classification] val AUROCEval: String = "auroc"

  /** String name for AUPRCEval */
  private[classification] val AUPRCEval: String = "auprc"

  /** String name for classification error */
  private[classification] val ErrorEval: String = "error"

  /** Set of evaluate functions that GBMClassifier supports */
  private[classification] val supportedEvals = Set(LogLossEval, AUROCEval, AUPRCEval, ErrorEval)
}


class GBMClassificationModel(override val uid: String, val model: GBMModel, val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, GBMClassificationModel]
    with GBMClassificationParams with MLWritable with Serializable {

  def setThreshold(value: Double): this.type = set(threshold, value)

  def setLeafCol(value: String): this.type = set(leafCol, value)

  def setEnableOneHot(value: Boolean): this.type = set(enableOneHot, value)

  def setFirstTrees(value: Int): this.type = {
    require(value <= model.numTrees)
    set(firstTrees, value)
  }

  override def numFeatures: Int = model.numFeatures

  def numTrees: Int = model.numTrees

  def numNodes: Array[Int] = model.numNodes

  def numLeaves: Array[Int] = model.numLeaves

  def weights: Array[Double] = model.weights

  def depths: Array[Int] = model.depths

  override def write: MLWriter = {
    new GBMClassificationModel.GBMClassificationModelWriter(this)
  }

  override def copy(extra: ParamMap): GBMClassificationModel = {
    copyValues(new GBMClassificationModel(uid, model, numClasses), extra).setParent(parent)
  }

  @transient private lazy val computeRaw = $(objectiveFunc) match {
    case "logistic" =>
      features: Vector =>
        val raw = model.predictRaw(features, $(firstTrees))(0)
        Vectors.dense(-raw, raw)

    case "softmax" =>
      features: Vector =>
        val raw = model.predict(features, $(firstTrees))
        Vectors.dense(raw)
  }

  @transient private lazy val computeScore = $(objectiveFunc) match {
    case "logistic" =>
      raw: Vector =>
        require(raw.size == 2)
        raw match {
          case dv: DenseVector =>
            val pos = model.obj.transform(Array(dv(1))).head
            dv.values(0) = 1 - pos
            dv.values(1) = pos
            dv

          case sv: SparseVector =>
            throw new RuntimeException("Unexpected error in GBMClassificationModel:" +
              " raw2probabilityInPlace encountered SparseVector")
        }

    case "softmax" =>
      raw: Vector =>
        require(raw.size == numClasses)
        raw match {
          case dv: DenseVector =>
            val score = model.obj.transform(dv.values)
            System.arraycopy(score, 0, dv.values, 0, numClasses)
            dv

          case sv: SparseVector =>
            throw new RuntimeException("Unexpected error in GBMClassificationModel:" +
              " raw2probabilityInPlace encountered SparseVector")
        }
  }

  override protected def predictRaw(features: Vector): Vector = {
    computeRaw(features)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    computeScore(rawPrediction)
  }

  def featureImportances: Vector = {
    model.computeImportances($(importanceType), $(firstTrees))
  }

  def leaf(features: Vector): Vector = {
    model.leaf(features, $(enableOneHot), $(firstTrees))
  }

  def leaf(dataset: Dataset[_]): DataFrame = {
    if ($(leafCol).nonEmpty) {
      val leafUDF = udf { features: Any =>
        leaf(features.asInstanceOf[Vector])
      }
      dataset.withColumn($(leafCol), leafUDF(col($(featuresCol))))
    } else {
      this.logWarning(s"$uid: GBMRegressionModel.leaf() was called as NOOP" +
        " since no output columns were set.")
      dataset.toDF
    }
  }
}


object GBMClassificationModel extends MLReadable[GBMClassificationModel] {

  override def read: MLReader[GBMClassificationModel] = new GBMClassificationModelReader

  private[GBMClassificationModel] class GBMClassificationModelWriter(instance: GBMClassificationModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sparkSession.sparkContext, None)

      GBMModel.save(sparkSession, instance.model, path)

      val otherDF = sparkSession.createDataFrame(Seq(
        ("type", "classification"),
        ("numClasses", instance.numClasses.toString),
        ("time", System.nanoTime().toString))).toDF("key", "value")
      val otherPath = new Path(path, "other").toString
      otherDF.write.parquet(otherPath)
    }
  }

  private class GBMClassificationModelReader extends MLReader[GBMClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMClassificationModel].getName

    override def load(path: String): GBMClassificationModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val otherPath = new Path(path, "other").toString
      val otherDF = sparkSession.read.parquet(otherPath)
      var numClasses = -1
      otherDF.select("key", "value").collect()
        .foreach { row =>
          val key = row.getString(0)
          val value = row.getString(1)
          key match {
            case "type" =>
              require(value == "classification")

            case "numClasses" =>
              numClasses = value.toInt
              require(numClasses > 1)

            case "time" =>
          }
        }

      val gbModel = GBMModel.load(path)

      val model = new GBMClassificationModel(metadata.uid, gbModel, numClasses)
      metadata.getAndSetParams(model)
      model
    }
  }

}
