package org.apache.spark.ml.gbm

import org.scalatest.FunSuite

class SplitSuite extends FunSuite {

  test("SeqSplit") {
    val split = SeqSplit(0, false, 5, true, 1.0, Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))

    assert(split.left)

    assert(split.leftWeight === 0.1)
    assert(split.leftGrad === 0.2)
    assert(split.leftHess === 0.3)
    assert(split.rightWeight === 0.4)
    assert(split.rightGrad === 0.5)
    assert(split.rightHess === 0.6)

    assert(!split.goLeft(Array(0L)))
    assert(split.goLeft(Array(1)))
    assert(split.goLeft(Array(5)))
    assert(!split.goLeft(Array(6)))

    val split2 = split.reverse

    assert(!split2.left)

    assert(split2.leftWeight === 0.4)
    assert(split2.leftGrad === 0.5)
    assert(split2.leftHess === 0.6)
    assert(split2.rightWeight === 0.1)
    assert(split2.rightGrad === 0.2)
    assert(split2.rightHess === 0.3)

    assert(split2.goLeft(Array(0L)))
    assert(!split2.goLeft(Array(1)))
    assert(!split2.goLeft(Array(5)))
    assert(split2.goLeft(Array(6)))
  }


  test("SetSplit") {
    val split = SetSplit(0, false, Array(2, 6, 9), true, 1.0, Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))

    assert(split.left)

    assert(split.leftWeight === 0.1)
    assert(split.leftGrad === 0.2)
    assert(split.leftHess === 0.3)
    assert(split.rightWeight === 0.4)
    assert(split.rightGrad === 0.5)
    assert(split.rightHess === 0.6)

    assert(!split.goLeft(Array(0L)))
    assert(!split.goLeft(Array(1)))
    assert(split.goLeft(Array(2)))
    assert(split.goLeft(Array(6)))
    assert(!split.goLeft(Array(7)))

    val split2 = split.reverse

    assert(!split2.left)

    assert(split2.leftWeight === 0.4)
    assert(split2.leftGrad === 0.5)
    assert(split2.leftHess === 0.6)
    assert(split2.rightWeight === 0.1)
    assert(split2.rightGrad === 0.2)
    assert(split2.rightHess === 0.3)

    assert(split2.goLeft(Array(0L)))
    assert(split2.goLeft(Array(1)))
    assert(!split2.goLeft(Array(2)))
    assert(!split2.goLeft(Array(6)))
    assert(split2.goLeft(Array(7)))
  }


  test("splitSeq") {


  }



  test("splitSetHeuristic") {


  }


  test("splitSetBrute") {


  }

}