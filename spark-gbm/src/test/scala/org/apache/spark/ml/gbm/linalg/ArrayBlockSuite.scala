package org.apache.spark.ml.gbm.linalg

import org.scalatest.FunSuite

class ArrayBlockSuite extends FunSuite {

  import ArrayBlock._

  test("basic") {
    val arrays = Array(Array(0), Array(1, 2, 3), Array.emptyIntArray, Array(4, 5, 6))
    val block = build(arrays)

    assert(block.size === 4)
    assert(block.flag === 0)
    assert(block.iterator.toArray === arrays)
  }


  test("all arrays are of same length") {
    val arrays = Array(Array(0, -1, -2), Array(1, 2, 3), Array(7, 8, 9), Array(4, 5, 6))
    val block = build(arrays)

    assert(block.size === 4)
    assert(block.flag === 3)
    assert(block.steps === Array())
    assert(block.iterator.toArray === arrays)
  }


  test("all arrays are the same") {
    val arrays = Array(Array(1, 2, 3), Array(1, 2, 3), Array(1, 2, 3), Array(1, 2, 3))
    val block = build(arrays)

    assert(block.size === 4)
    assert(block.flag === -4)
    assert(block.values === Array(1, 2, 3))
    assert(block.steps === Array())
    assert(block.iterator.toArray === arrays)
  }


  test("fill") {
    val block = fill(4, Array(1, 2, 3))

    assert(block.size === 4)
    assert(block.flag === -4)
    assert(block.values === Array(1, 2, 3))
    assert(block.steps === Array())
    assert(block.iterator.toArray === Array(Array(1, 2, 3), Array(1, 2, 3), Array(1, 2, 3), Array(1, 2, 3)))
  }


  test("empty input") {
    val block = build(Seq.empty[Array[Int]])

    assert(block.isEmpty)
    assert(block.iterator.isEmpty)
  }
}


