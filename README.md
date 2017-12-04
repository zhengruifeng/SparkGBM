# SparkGBM
SparkGBM is an implementation of Gradient Boosting Machine atop Apache Spark.
It is designed to be scalable and efficient with the following advantages:

1, Compatible with current ML/MLlib pipeline

2, Purely writen in Scala/Spark, no other dependancy

3, Faster training speed compared with `ml.GBT`


## Features
Thanks to XGBoost and LightGBM, SparkGBM draws on the valuable experience of them to aim to be an efficient framework:

From XGBoost we introduced:

1, Second order approximation of objective function

2, L1/L2 regulation of weights to prevent overfitting

3, Column subsampling by tree and by level

4, Sparsity-awareness

From LightGBM we introduced:

1, Histogram subtraction to halve communication overhead

2, Automatic data compression to reduce memory footprint


## API

### High-level DataFrame-based APIs

**GBMClassifier** for binary classification: 
[Source](https://github.com/zhengruifeng/SparkGBM/blob/master/src/main/scala/org/apache/spark/ml/classification/GBMClassifier.scala)
[Example](https://github.com/zhengruifeng/SparkGBM/blob/master/src/main/scala/example/GBMClassifierExample.scala)

    import org.apache.spark.ml.classification._
    
    val gbmc = new GBMClassifier
    
    gbmc.setBoostType("gbtree")                     // "dart" -> DART, "gbtree" -> gradient boosting
        .setObjectiveFunc("logistic")               // "logistic" -> logloss
        .setEvaluateFunc(Array("auc", "logloss"))   // "auc", "logloss", "error"
        .setMaxIter(10)                             // maximum number of iterations
        .setModelCheckpointInterval(4)              // model checkpoint interval
        .setModelCheckpointPath(path)               // model checkpoint directory
    
    // training without validation
    val model = gbmc.fit(train)
    
    // load the snapshots saved during the training
    val modelSnapshot4 = GBMClassificationModel.load(s"$path/model-4")
    val modelSnapshot8 = GBMClassificationModel.load(s"$path/model-8")

    // model save and load
    model.save(savePath)
    val model2 = GBMClassificationModel.load(savePath)

**GBMRegressor** for regression: 
[Source](https://github.com/zhengruifeng/SparkGBM/blob/master/src/main/scala/org/apache/spark/ml/regression/GBMRegressor.scala)
[Example](https://github.com/zhengruifeng/SparkGBM/blob/master/src/main/scala/example/GBMRegressorExample.scala)


    import org.apache.spark.ml.regression._
    
    val gbmr = new GBMRegressor
    
    gbmr.setBoostType("dart")                   // "dart" -> DART, "gbtree" -> gradient boosting
        .setObjectiveFunc("square")             // "square" -> MSE, "huber" -> Pseudo-Huber loss
        .setEvaluateFunc(Array("rmse", "mae"))  // "rmse", "mse", "mae"
        .setMaxIter(10)                         // maximum number of iterations
        .setMaxDepth(7)                         // maximum depth
        .setMaxBins(32)                         // maximum number of bins
        .setNumericalBinType("width")           // "width" -> by interval-equal bins, "depth" -> by quantiles 
        .setMaxLeaves(100)                      // maximum number of leaves
        .setMinNodeHess(0.001)                  // minimum hessian needed in a node
        .setRegAlpha(0.1)                       // L1 regularization
        .setRegLambda(0.5)                      // L2 regularization
        .setDropRate(0.1)                       // dropout rate
        .setDropSkip(0.5)                       // probability of skipping drop
        .setInitialModelPath(path)              // path of initial model
        .setEarlyStopIters(10)                  // early stopping
    
    // training without validation, early stopping is ignored
    val model1 = gbmr.fit(train) 
    
    // training with validation
    val model2 = gbmr.fit(train, test)
    
    // using only 5 tree for the following feature importance computation, prediction and leaf transformation 
    model2.setFirstTrees(5)
    
    // feature importance
    model2.featureImportances
    
    // prediction
    model2.transform(test)
    
    // enable one-hot leaf transform
    model2.setEnableOneHot(true)
    model2.leaf(test)
   

### Low-level RDD-based APIs:

Besides all the functions in DataFrame-based APIs, RDD-based APIs also support user-defined **objective**, **evaluation** and **callback**.
[Source](https://github.com/zhengruifeng/SparkGBM/blob/master/src/main/scala/org/apache/spark/ml/gbm/GBM.scala)
[Example](https://github.com/zhengruifeng/SparkGBM/blob/master/src/main/scala/example/GBMExample.scala)

    import org.apache.spark.ml.gbm._
    
    // User defined objective function
    val obj = new ObjFunc {
      override def compute(label: Double,
                           score: Double): (Double, Double) = (score - label, 1.0)
      override def name: String = "Another Square"
    }

    // User defined evaluation function for R2
    val r2Eval = new BatchEvalFunc {
      override def compute(data: RDD[(Double, Double, Double)]): Double = {
        // ignore weight
        new RegressionMetrics(data.map(t => (t._2, t._3))).r2
      }

      override def isLargerBetter: Boolean = true

      override def name: String = "R2 (no weight)"
    }

    // User defined evaluation function for MAE
    val maeEval = new SimpleEvalFunc {
      override def compute(label: Double,
                           score: Double): Double = (label - score).abs

      override def isLargerBetter: Boolean = false

      override def name: String = "Another MAE"
    }

    // User defined callback function
    val lrUpdater = new CallbackFunc {
      override def compute(spark: SparkSession,
                           boostConfig: BoostConfig,
                           model: GBMModel,
                           trainMetrics: Array[Map[String, Double]],
                           testMetrics: Array[Map[String, Double]]): Boolean = {
        // learning rate decay
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


    val gbm = new GBM
    gbm.setMaxIter(20)
      .setMaxDepth(5)
      .setStepSize(0.2)
      .setNumericalBinType("depth")
      .setObjectiveFunc(obj)
      .setEvaluateFunc(Array(r2Eval, maeEval))
      .setCallbackFunc(Array(lrUpdater))

    // train with validation
    val model = gbm.fit(train, test)
    



## Building

    mvn clean package
    
## Note

Current master branch work for **Spark-2.2.0**
