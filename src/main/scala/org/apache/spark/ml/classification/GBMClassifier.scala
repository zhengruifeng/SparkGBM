package org.apache.spark.ml.classification

import scala.collection.mutable.ArrayBuffer

import org.apache.hadoop.fs.Path

import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasThreshold
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Dataset}
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

  setDefault(evaluateFunc -> Array(GBMClassifier.LogLossEval, GBMClassifier.AUCEval, GBMClassifier.ErrorEval))
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

  def setBaseScore(value: Double): this.type = set(baseScore, value)

  def setMinNodeHess(value: Double): this.type = set(minNodeHess, value)

  def setRegAlpha(value: Double): this.type = set(regAlpha, value)

  def setRegLambda(value: Double): this.type = set(regLambda, value)

  def setCatCols(value: Array[Int]): this.type = set(catCols, value)

  def setRankCols(value: Array[Int]): this.type = set(rankCols, value)

  def setSubSample(value: Double): this.type = set(subSample, value)

  def setColSampleByTree(value: Double): this.type = set(colSampleByTree, value)

  def setColSampleByLevel(value: Double): this.type = set(colSampleByLevel, value)

  def setMinGain(value: Double): this.type = set(minGain, value)

  def setStorageLevel(value: String): this.type = set(storageLevel, value)

  def setObjectiveFunc(value: String): this.type = set(objectiveFunc, value)

  def setEvaluateFunc(value: Array[String]): this.type = set(evaluateFunc, value)

  def setBoostType(value: String): this.type = set(boostType, value)

  def setDropRate(value: Double): this.type = set(dropRate, value)

  def setDropSkip(value: Double): this.type = set(dropSkip, value)

  def setMinDrop(value: Int): this.type = set(minDrop, value)

  def setMaxDrop(value: Int): this.type = set(maxDrop, value)

  def setInitialModelPath(value: String): this.type = set(initialModelPath, value)

  def setMaxBruteBins(value: Int): this.type = set(maxBruteBins, value)

  def setEarlyStopIters(value: Int): this.type = set(earlyStopIters, value)

  def setModelCheckpointInterval(value: Int): this.type = set(modelCheckpointInterval, value)

  def setModelCheckpointPath(value: String): this.type = set(modelCheckpointPath, value)

  def setNumericalBinType(value: String): this.type = set(numericalBinType, value)

  def setEnableOneHot(value: Boolean): this.type = set(enableOneHot, value)

  def setFirstTrees(value: Int): this.type = set(firstTrees, value)

  def setFloatType(value: String): this.type = set(floatType, value)

  override def fit(dataset: Dataset[_]): GBMClassificationModel = {
    fit(dataset, None)
  }

  def fit(dataset: Dataset[_],
          testDataset: Dataset[_]): GBMClassificationModel = {
    fit(dataset, Some(testDataset))
  }

  private[ml] def fit(dataset: Dataset[_],
                      testDataset: Option[Dataset[_]]): GBMClassificationModel = {
    require($(maxDrop) >= $(minDrop))

    transformSchema(dataset.schema, logging = true)

    val instr = Instrumentation.create(this, dataset)
    instr.logParams(params: _*)

    val w = if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      col($(weightCol)).cast(DoubleType)
    } else {
      lit(1.0)
    }

    val data = dataset.select(w, col($(labelCol)).cast(DoubleType), col($(featuresCol)))
      .rdd.map { row =>
      (row.getDouble(0), row.getDouble(1), row.getAs[Vector](2))
    }

    val test = testDataset.map { data =>
      data.select(w, col($(labelCol)).cast(DoubleType), col($(featuresCol)))
        .rdd.map { row =>
        (row.getDouble(0), row.getDouble(1), row.getAs[Vector](2))
      }
    }

    val objFunc: LogisticObj =
      $(objectiveFunc) match {
        case GBMClassifier.LogisticObj =>
          new LogisticObj
      }

    val evalFunc: Array[EvalFunc] =
      $(evaluateFunc).map {
        case GBMClassifier.LogLossEval =>
          new LogLossEval
        case GBMClassifier.AUCEval =>
          new AUCEval
        case GBMClassifier.ErrorEval =>
          new ErrorEval($(threshold))
      }

    val callBackFunc = ArrayBuffer[CallbackFunc]()
    if ($(earlyStopIters) >= 1) {
      callBackFunc.append(new EarlyStopFunc($(earlyStopIters)))
    }
    if ($(modelCheckpointInterval) >= 1 && $(modelCheckpointPath).nonEmpty) {
      val mockModel = copyValues(new GBMClassificationModel(uid, null).setParent(this))
      callBackFunc.append(new ClassificationModelCheckpointFunc($(modelCheckpointInterval), $(modelCheckpointPath), mockModel))
    }

    val initialModel =
      if (isDefined(initialModelPath) && $(initialModelPath).nonEmpty) {
        Some(GBMClassificationModel.load($(initialModelPath)).model)
      } else {
        None
      }

    val gbm = new GBM
    gbm.setMaxIter($(maxIter))
      .setMaxDepth($(maxDepth))
      .setMaxLeaves($(maxLeaves))
      .setMaxBins($(maxBins))
      .setMinGain($(minGain))
      .setMinNodeHess($(minNodeHess))
      .setBaseScore($(baseScore))
      .setStepSize($(stepSize))
      .setRegAlpha($(regAlpha))
      .setRegLambda($(regLambda))
      .setObjectiveFunc(objFunc)
      .setEvaluateFunc(evalFunc)
      .setCallbackFunc(callBackFunc.toArray)
      .setCatCols($(catCols).toSet)
      .setRankCols($(rankCols).toSet)
      .setSubSample($(subSample))
      .setColSampleByTree($(colSampleByTree))
      .setColSampleByLevel($(colSampleByLevel))
      .setCheckpointInterval($(checkpointInterval))
      .setStorageLevel(StorageLevel.fromString($(storageLevel)))
      .setAggregationDepth($(aggregationDepth))
      .setSeed($(seed))
      .setBoostType($(boostType))
      .setDropRate($(dropRate))
      .setDropSkip($(dropSkip))
      .setMinDrop($(minDrop))
      .setMaxDrop($(maxDrop))
      .setMaxBruteBins($(maxBruteBins))
      .setNumericalBinType($(numericalBinType))
      .setInitialModel(initialModel)

    val gbmModel = gbm.fit(data, test)

    val model = new GBMClassificationModel(uid, gbmModel)
    instr.logSuccess(model)
    copyValues(model.setParent(this))
  }

  override def copy(extra: ParamMap): GBMClassifier = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): GBMClassificationModel = {
    throw new NotImplementedError(s"train is not implemented for ${this.getClass}.")
    null
  }
}

object GBMClassifier extends DefaultParamsReadable[GBMClassifier] {
  override def load(path: String): GBMClassifier = super.load(path)

  /** String name for LogisticObj */
  private[classification] val LogisticObj: String = "logistic"

  /** Set of objective functions that GBMClassifier supports */
  private[classification] val supportedObjs = Set(LogisticObj)

  /** String name for LogLossEval */
  private[classification] val LogLossEval: String = "logloss"

  /** String name for AUCEval */
  private[classification] val AUCEval: String = "auc"

  /** String name for classification error */
  private[classification] val ErrorEval: String = "error"

  /** Set of evaluate functions that GBMClassifier supports */
  private[classification] val supportedEvals = Set(LogLossEval, AUCEval, ErrorEval)
}


class GBMClassificationModel(override val uid: String, val model: GBMModel)
  extends ProbabilisticClassificationModel[Vector, GBMClassificationModel]
    with GBMClassificationParams with MLWritable with Serializable {

  def setThreshold(value: Double): this.type = set(threshold, value)

  def setLeafCol(value: String): this.type = set(leafCol, value)

  def setEnableOneHot(value: Boolean): this.type = set(enableOneHot, value)

  def setFirstTrees(value: Int): this.type = {
    require(value <= model.numTrees)
    set(firstTrees, value)
  }

  override def numFeatures: Int = model.numCols

  override def numClasses = 2

  def numTrees: Int = model.numTrees

  def numNodes: Array[Long] = model.numNodes

  def numLeaves: Array[Long] = model.numLeaves

  def weights: Array[Double] = model.weights

  def depths: Array[Int] = model.depths

  override def write: MLWriter = {
    new GBMClassificationModel.GBMClassificationModelWriter(this)
  }

  override def copy(extra: ParamMap): GBMClassificationModel = {
    copyValues(new GBMClassificationModel(uid, model), extra).setParent(parent)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): DenseVector = {
    rawPrediction match {
      case dv: DenseVector =>
        dv.values(1) = 1.0 / (1.0 + math.exp(-dv.values(1)))
        dv.values(0) = 1.0 - dv.values(1)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in GBTClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    val score = model.predict(features, $(firstTrees))
    Vectors.dense(-score, score)
  }

  def featureImportances: Vector = {
    val n = $(firstTrees)
    if (n == -1 || n == model.numTrees) {
      /** precomputed feature importance */
      model.importance
    } else {
      logWarning(s"Compute feature importances with first $n trees")
      model.computeImportance(n)
    }
  }

  def leaf(features: Vector): Vector = {
    model.leaf(features, $(enableOneHot), $(firstTrees))
  }

  def leaf(dataset: Dataset[_]): DataFrame = {
    if ($(leafCol).nonEmpty) {
      val leafUDF = udf { (features: Any) =>
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

      GBMModel.save(instance.model, path)

      val otherDF = sparkSession.createDataFrame(Seq(
        ("type", "classification"),
        ("time", System.nanoTime.toString))).toDF("key", "value")
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
      otherDF.select("key", "value").collect()
        .foreach { row =>
          val key = row.getString(0)
          val value = row.getString(1)
          key match {
            case "type" =>
              require(value == "classification")
            case "time" =>
          }
        }

      val gbModel = GBMModel.load(path)

      val model = new GBMClassificationModel(metadata.uid, gbModel)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}
