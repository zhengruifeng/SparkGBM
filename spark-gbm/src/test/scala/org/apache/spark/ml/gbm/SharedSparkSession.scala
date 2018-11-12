package org.apache.spark.ml.gbm

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

trait SharedSparkSession {

  var spark: SparkSession = null
  var sc: SparkContext = null

  def beforeAll(): Unit = {
    spark = SparkSession.builder().getOrCreate()
    sc = spark.sparkContext
  }

  def afterAll(): Unit = {
    spark.stop()
    spark = null
    sc = null
  }
}
