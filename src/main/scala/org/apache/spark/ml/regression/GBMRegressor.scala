package org.apache.spark.ml.regression

import scala.collection.mutable

import org.apache.hadoop.fs.Path

import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.storage.StorageLevel


trait GBMRegressionParams extends GBMParams {

  /**
    * Learning objective function.
    *
    * @group expertParam
    */
  final val objectiveFunc: Param[String] =
    new Param[String](this, "objectiveFunc", "Learning objective function",
      (obj: String) => GBMRegressor.supportedObjs.contains(obj))

  def getObjectiveFunc: String = $(objectiveFunc)

  setDefault(objectiveFunc -> GBMRegressor.SquareObj)

  /**
    * Metric functions.
    *
    * @group expertParam
    */
  final val evaluateFunc: StringArrayParam =
    new StringArrayParam(this, "evaluateFunc", "Metric functions",
      (evals: Array[String]) => evals.forall(GBMRegressor.supportedEvals.contains))

  def getEvaluateFunc: Array[String] = $(evaluateFunc)

  setDefault(evaluateFunc -> Array(GBMRegressor.RMSEEval, GBMRegressor.MSEEval, GBMRegressor.MAEEval, GBMRegressor.R2Eval))

  /**
    * Delta in huber loss for robust regression
    * (default = 1.0)
    */
  final val huberDelta: DoubleParam =
    new DoubleParam(this, "huberDelta", "Delta in huber loss for robust regression",
      ParamValidators.gt(0.0))

  def getHuberDelta: Double = $(huberDelta)

  setDefault(huberDelta -> 1.0)
}


class GBMRegressor(override val uid: String) extends
  Predictor[Vector, GBMRegressor, GBMRegressionModel]
  with GBMRegressionParams with DefaultParamsWritable with Logging {

  def this() = this(Identifiable.randomUID("gbmr"))

  def setHuberDelta(value: Double): this.type = set(huberDelta, value)

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

  def setZeroAsMissing(value: Boolean): this.type = set(zeroAsMissing, value)

  def setSparsityThreshold(value: Double): this.type = set(sparsityThreshold, value)

  def setParallelism(value: Int): this.type = set(parallelism, value)

  override def fit(dataset: Dataset[_]): GBMRegressionModel = {
    fit(dataset, None)
  }

  def fit(dataset: Dataset[_],
          testDataset: Dataset[_]): GBMRegressionModel = {
    fit(dataset, Some(testDataset))
  }

  private[ml] def fit(dataset: Dataset[_],
                      testDataset: Option[Dataset[_]]): GBMRegressionModel = {
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

    val objFunc: ObjFunc =
      $(objectiveFunc) match {
        case GBMRegressor.SquareObj =>
          new SquareObj
        case GBMRegressor.HuberObj =>
          new HuberObj($(huberDelta))
      }

    val evalFunc: Array[EvalFunc] =
      $(evaluateFunc).map {
        case GBMRegressor.MAEEval =>
          new MAEEval
        case GBMRegressor.MSEEval =>
          new MSEEval
        case GBMRegressor.RMSEEval =>
          new RMSEEval
        case GBMRegressor.R2Eval =>
          new R2Eval
      }.map(_.asInstanceOf[EvalFunc])

    val callBackFunc = mutable.ArrayBuffer.empty[CallbackFunc]
    if ($(earlyStopIters) >= 1) {
      callBackFunc.append(new EarlyStop($(earlyStopIters)))
    }
    if ($(modelCheckpointInterval) >= 1 && $(modelCheckpointPath).nonEmpty) {
      val mockModel = copyValues(new GBMRegressionModel(uid, null).setParent(this))
      callBackFunc.append(new RegressionModelCheckpoint($(modelCheckpointInterval), $(modelCheckpointPath), mockModel))
    }

    val initialModel =
      if (isDefined(initialModelPath) && $(initialModelPath).nonEmpty) {
        val model = GBMRegressionModel.load($(initialModelPath))
        if (model.getObjectiveFunc != $(objectiveFunc)) {
          logWarning(s"The objective function conflicts with that in initial model," +
            s" objective of initial model ${model.getObjectiveFunc} will be ignored.")
        }
        Some(model.model)
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
      .setZeroAsMissing($(zeroAsMissing))
      .setSparsityThreshold($(sparsityThreshold))
      .setParallelism($(parallelism))
      .setInitialModel(initialModel)

    val gbmModel = gbm.fit(data, test)

    val model = new GBMRegressionModel(uid, gbmModel)
    instr.logSuccess(model)
    copyValues(model.setParent(this))
  }

  override protected def train(dataset: Dataset[_]): GBMRegressionModel = {
    throw new NotImplementedError(s"train is not implemented for ${this.getClass}.")
    null
  }

  override def copy(extra: ParamMap): GBMRegressor = defaultCopy(extra)
}

object GBMRegressor extends DefaultParamsReadable[GBMRegressor] {
  override def load(path: String): GBMRegressor = super.load(path)

  /** String name for SquareObj */
  private[regression] val SquareObj: String = "square"

  /** String name for HuberObj */
  private[regression] val HuberObj: String = "huber"

  /** Set of objective functions that GBMRegressor supports */
  private[regression] val supportedObjs = Set(SquareObj, HuberObj)

  /** String name for RMSEEval */
  private[regression] val RMSEEval: String = "rmse"

  /** String name for MSEEval */
  private[regression] val MSEEval: String = "mse"

  /** String name for MAEEval */
  private[regression] val MAEEval: String = "mae"

  /** String name for R2Eval */
  private[regression] val R2Eval: String = "r2"

  /** Set of evaluate functions that GBMRegressor supports */
  private[regression] val supportedEvals = Set(RMSEEval, MSEEval, MAEEval, R2Eval)
}

class GBMRegressionModel(override val uid: String, val model: GBMModel)
  extends PredictionModel[Vector, GBMRegressionModel]
    with GBMRegressionParams with MLWritable with Serializable {

  def setLeafCol(value: String): this.type = set(leafCol, value)

  def setEnableOneHot(value: Boolean): this.type = set(enableOneHot, value)

  def setFirstTrees(value: Int): this.type = {
    require(value <= model.numTrees)
    set(firstTrees, value)
  }

  override def numFeatures: Int = model.numCols

  def numTrees: Int = model.numTrees

  def numNodes: Array[Long] = model.numNodes

  def numLeaves: Array[Long] = model.numLeaves

  def weights: Array[Double] = model.weights

  def depths: Array[Int] = model.depths

  override def write: MLWriter = {
    new GBMRegressionModel.GBMRegressionModelWriter(this)
  }

  override def copy(extra: ParamMap): GBMRegressionModel = {
    copyValues(new GBMRegressionModel(uid, model), extra).setParent(parent)
  }

  override def predict(features: Vector): Double = {
    model.predict(features, $(firstTrees))
  }

  def featureImportances: Vector = {
    val n = $(firstTrees)
    if (n == -1 || n == model.numTrees) {
      // precomputed feature importance
      model.importance
    } else {
      logInfo(s"Compute feature importances with first $n trees")
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


object GBMRegressionModel extends MLReadable[GBMRegressionModel] {

  override def read: MLReader[GBMRegressionModel] = new GBMRegressionModelReader

  private[GBMRegressionModel] class GBMRegressionModelWriter(instance: GBMRegressionModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sparkSession.sparkContext, None)

      GBMModel.save(sparkSession, instance.model, path)

      val otherDF = sparkSession.createDataFrame(Seq(
        ("type", "regression"),
        ("time", System.nanoTime.toString))).toDF("key", "value")
      val otherPath = new Path(path, "other").toString
      otherDF.write.parquet(otherPath)
    }
  }

  private class GBMRegressionModelReader extends MLReader[GBMRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMRegressionModel].getName

    override def load(path: String): GBMRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val otherPath = new Path(path, "other").toString
      val otherDF = sparkSession.read.parquet(otherPath)
      otherDF.select("key", "value").collect()
        .foreach { row =>
          val key = row.getString(0)
          val value = row.getString(1)
          key match {
            case "type" =>
              require(value == "regression")
            case "time" =>
          }
        }

      val gbModel = GBMModel.load(path)

      val model = new GBMRegressionModel(metadata.uid, gbModel)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}
