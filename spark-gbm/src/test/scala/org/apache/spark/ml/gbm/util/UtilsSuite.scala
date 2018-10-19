package org.apache.spark.ml.gbm.util

import org.scalatest.{BeforeAndAfter, FunSuite}

class UtilsSuite extends FunSuite with BeforeAndAfter {

  import Utils._

  test("arrayEquiv") {
    assert(arrayEquiv[Int](null, null) === true)
    assert(arrayEquiv(Array(1), null) === false)
    assert(arrayEquiv(Array(1, 2), Array(1, 2)) === true)
    assert(arrayEquiv(Array(1, 2), Array(1)) === false)
    assert(arrayEquiv(Array(1, 2), Array(1, 1)) === false)
  }


  test("validateOrdering") {
    assert(validateOrdering(Seq.empty[Int].iterator).isEmpty)
    assert(validateOrdering(Iterator(1)).size === 1)
    assert(validateOrdering(Iterator(1, 2, 3)).size === 3)
    assert(validateOrdering(Iterator(1, 2, 2), true, false).size === 3)
    assert(validateOrdering(Iterator(3, 2, 1), false).size === 3)
    assert(validateOrdering(Iterator(2, 2, 1), false, false).size === 3)

    intercept[IllegalArgumentException] {
      validateOrdering(Iterator(1, 2, 2)).size
    }

    intercept[IllegalArgumentException] {
      validateOrdering(Iterator(1, 2, 1), true, false).size
    }

    intercept[IllegalArgumentException] {
      validateOrdering(Iterator(2, 2, 1), false).size
    }

    intercept[IllegalArgumentException] {
      validateOrdering(Iterator(1, 2, 1), false, false).size
    }
  }

}
