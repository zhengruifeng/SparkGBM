package org.apache.spark.ml.gbm

import java.io._
import java.{util => ju}

import scala.collection.mutable

import org.apache.spark.ml.gbm.func.ObjFunc
import org.apache.spark.ml.gbm.util.Utils
import org.apache.spark.ml.linalg._
import org.apache.spark.sql._

/**
  * Model of GBM
  *
  * @param discretizer discretizer to convert raw features into bins
  * @param rawBase     raw bias
  * @param trees       array of trees
  * @param weights     weights of trees
  */
class GBMModel(val obj: ObjFunc,
               val discretizer: Discretizer,
               val rawBase: Array[Double],
               val trees: Array[TreeModel],
               val weights: Array[Double]) extends Serializable {
  require(rawBase.nonEmpty)
  require(rawBase.forall(v => !v.isNaN && !v.isInfinity))
  require(trees.length % rawBase.length == 0)
  require(trees.length == weights.length)
  require(weights.forall(w => !w.isNaN && !w.isInfinity))


  @transient lazy val baseScore: Array[Double] = obj.transform(rawBase)

  def numFeatures: Int = discretizer.colDiscretizers.length

  def numTrees: Int = trees.length

  def numLeaves: Array[Int] = trees.map(_.numLeaves)

  def numNodes: Array[Int] = trees.map(_.numNodes)

  def depths: Array[Int] = trees.map(_.depth)


  /** feature importance of the first trees */
  def computeImportances(importanceType: String,
                         firstTrees: Int): Vector = {
    require(firstTrees >= -1 && firstTrees <= numTrees)

    var n = firstTrees
    if (n == 0) {
      return Vectors.sparse(numFeatures, Seq.empty)
    }

    if (n == -1) {
      n = numTrees
    }


    val iter = Iterator.range(0, n)
      .flatMap { i => trees(i).root.internalNodeIterator }

    val gains = importanceType.toLowerCase(ju.Locale.ROOT) match {

      case GBM.AvgGain =>
        val gains = mutable.OpenHashMap.empty[Int, (Double, Int)]
        while (iter.hasNext) {
          val n = iter.next()
          val (g, c) = gains.getOrElse(n.colId, (0.0, 0))
          gains.update(n.colId, (g + n.gain, c + 1))
        }

        gains.map { case (colId, (gain, count)) =>
          (colId, gain / count)
        }.toArray


      case GBM.SumGain =>
        val gains = mutable.OpenHashMap.empty[Int, Double]
        while (iter.hasNext) {
          val n = iter.next()
          val g = gains.getOrElse(n.colId, 0.0)
          gains.update(n.colId, g + n.gain)
        }

        gains.toArray


      case GBM.NumSplits =>
        val counts = mutable.OpenHashMap.empty[Int, Int]
        while (iter.hasNext) {
          val n = iter.next()
          val c = counts.getOrElse(n.colId, 0)
          counts.update(n.colId, c + 1)
        }

        counts.mapValues(_.toDouble).toArray
    }

    val (indices, values) = gains.sortBy(_._1).unzip

    Vectors.sparse(numFeatures, indices, values)
      .compressed
  }


  def predict(features: Vector): Array[Double] = {
    predict(features, numTrees)
  }


  def predict(features: Vector,
              firstTrees: Int): Array[Double] = {
    val raw = predictRaw(features, firstTrees)
    obj.transform(raw)
  }

  def predictRaw(features: Vector): Array[Double] = {
    predictRaw(features, numTrees)
  }

  def predictRaw(features: Vector,
                 firstTrees: Int): Array[Double] = {
    require(features.size == numFeatures)
    require(firstTrees >= -1 && firstTrees <= numTrees)

    val bins = discretizer.transform[Int](features)

    var n = firstTrees
    if (n == -1) {
      n = numTrees
    }

    val raw = rawBase.clone()
    var i = 0
    while (i < n) {
      raw(i % rawBase.length) += trees(i).predict[Int](bins) * weights(i)
      i += 1
    }

    raw
  }


  def leaf(features: Vector): Vector = {
    leaf(features, false)
  }


  def leaf(features: Vector,
           oneHot: Boolean): Vector = {
    leaf(features, oneHot, numTrees)
  }


  /** leaf transformation with first #firstTrees trees
    * if oneHot is enable, transform input into a sparse one-hot encoded vector */
  def leaf(features: Vector,
           oneHot: Boolean,
           firstTrees: Int): Vector = {
    require(features.size == numFeatures)
    require(firstTrees >= -1 && firstTrees <= numTrees)

    val bins = discretizer.transform[Int](features)

    var n = firstTrees
    if (n == -1) {
      n = numTrees
    }

    if (oneHot) {
      val indices = Array.ofDim[Int](n)
      var step = 0
      var i = 0
      while (i < n) {
        val index = trees(i).index(bins)
        indices(i) = step + index
        step += numLeaves(i)
        i += 1
      }

      val values = Array.fill(n)(1.0)
      Vectors.sparse(step, indices, values)

    } else {
      val indices = Array.ofDim[Double](n)
      var i = 0
      while (i < n) {
        indices(i) = trees(i).index(bins)
        i += 1
      }

      Vectors.dense(indices)
        .compressed
    }
  }

  def save(path: String): Unit = {
    val spark = SparkSession.builder.getOrCreate()
    GBMModel.save(spark, this, path)
  }
}


object GBMModel {

  /** save GBMModel to a path */
  private[ml] def save(spark: SparkSession,
                       model: GBMModel,
                       path: String): Unit = {
    val names = Array("obj", "discretizerCol", "discretizerExtra", "trees", "extra")
    val dataframes = toDF(spark, model)
    Utils.saveDataFrames(dataframes, names, path)
  }


  /** load GBMModel from a path */
  def load(path: String): GBMModel = {
    val spark = SparkSession.builder().getOrCreate()
    val names = Array("obj", "discretizerCol", "discretizerExtra", "trees", "extra")
    val dataframes = Utils.loadDataFrames(spark, names, path)
    fromDF(dataframes)
  }


  /** helper function to convert GBMModel to dataframes */
  private[gbm] def toDF(spark: SparkSession,
                        model: GBMModel): Array[DataFrame] = {
    val Array(disColDF, disExtraDF) = Discretizer.toDF(spark, model.discretizer)


    val bos = new ByteArrayOutputStream
    val oos = new ObjectOutputStream(bos)
    oos.writeObject(model.obj)
    oos.flush()
    val objBytes = bos.toByteArray
    val objDF = spark.createDataFrame(
      Seq(("obj", objBytes)))
      .toDF("key", "value")

    val treesDatum = model.trees.zipWithIndex.flatMap {
      case (tree, index) =>
        val (nodeData, _) = NodeData.createData(tree.root, 0)
        nodeData.map((_, index))
    }
    val treesDF = spark.createDataFrame(treesDatum)
      .toDF("node", "treeIndex")

    val extraDF = spark.createDataFrame(
      Seq(("weights", model.weights),
        ("rawBase", model.rawBase)))
      .toDF("key", "value")

    Array(objDF, disColDF, disExtraDF, treesDF, extraDF)
  }


  /** helper function to convert dataframes back to GBMModel */
  private[gbm] def fromDF(dataframes: Array[DataFrame]): GBMModel = {
    val Array(objDF, disColDF, disExtraDF, treesDF, extraDF) = dataframes

    val spark = objDF.sparkSession
    import spark.implicits._

    val discretizer = Discretizer.fromDF(Array(disColDF, disExtraDF))

    val objBytes = objDF.first().get(1).asInstanceOf[Array[Byte]]
    val bis = new ByteArrayInputStream(objBytes)
    val ois = new ObjectInputStream(bis)
    val obj = ois.readObject().asInstanceOf[ObjFunc]

    val (indice, trees) =
      treesDF.select("treeIndex", "node").as[(Int, NodeData)].rdd
        .groupByKey().map { case (index, nodes) =>
        val root = NodeData.createNode(nodes.toArray)
        (index, new TreeModel(root))
      }.collect().sortBy(_._1).unzip

    require(indice.length == indice.distinct.length)
    require(indice.length == indice.fold(0)(math.max) + 1)

    var rawBase = Array.emptyDoubleArray
    var weights = Array.emptyDoubleArray
    extraDF.select("key", "value").collect()
      .foreach { row =>
        val key = row.getString(0)
        val value = row.getSeq[Double](1).toArray

        key match {
          case "weights" => weights = value
          case "rawBase" => rawBase = value
        }
      }
    require(rawBase.forall(v => !v.isNaN && !v.isInfinity))

    new GBMModel(obj, discretizer, rawBase, trees, weights)
  }
}