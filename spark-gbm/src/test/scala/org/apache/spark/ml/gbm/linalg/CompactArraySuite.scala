package org.apache.spark.ml.gbm.linalg

import org.scalatest.FunSuite

class CompactArraySuite extends FunSuite {

  import CompactArray._

  test("basic") {
    val values = Array(1.0, 3.0, 4.0, 5.0, 1.0)
    val array = build(values)

    assert(array.size === values.length)
    assert(array.iterator.toArray === values)
  }


  test("all values are the same") {
    val values = Array.fill(10)(1.0)
    val array = build(values)

    assert(array.values === Array(1.0))
    assert(array.times === Array(10))
    assert(array.size === values.length)
    assert(array.iterator.toArray === values)
  }


  test("empty input") {
    val array = build(Seq.empty[Int])

    assert(array.isEmpty)
    assert(array.iterator.isEmpty)
  }
}


