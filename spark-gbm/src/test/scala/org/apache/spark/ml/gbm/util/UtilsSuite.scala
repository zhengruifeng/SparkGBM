package org.apache.spark.ml.gbm.util

import org.scalatest.FunSuite
import scala.collection.mutable

import org.apache.spark.ml.linalg._

class UtilsSuite extends FunSuite {

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


  test("validateKeyOrdering") {
    assert(validateKeyOrdering(Seq.empty[(Int, Long)].iterator).isEmpty)
    assert(validateKeyOrdering(Iterator((1, 2L))).size === 1)
    assert(validateKeyOrdering(Iterator((1, 2L), (2, -3L), (3, 5L))).size === 3)
    assert(validateKeyOrdering(Iterator((1, 2L), (2, -3L), (2, -3L)), true, false).size === 3)
    assert(validateKeyOrdering(Iterator((3, 5L), (2, -3L), (1, 2L)), false).size === 3)
    assert(validateKeyOrdering(Iterator((2, -3L), (2, -3L), (1, 2L)), false, false).size === 3)

    intercept[IllegalArgumentException] {
      validateKeyOrdering(Iterator((1, 2L), (2, -3L), (2, -3L))).size
    }

    intercept[IllegalArgumentException] {
      validateKeyOrdering(Iterator((1, 2L), (2, -3L), (1, 2L)), true, false).size
    }

    intercept[IllegalArgumentException] {
      validateKeyOrdering(Iterator((2, -3L), (2, -3L), (1, 2L)), false).size
    }

    intercept[IllegalArgumentException] {
      validateKeyOrdering(Iterator((1, 2L), (2, -3L), (1, 2L)), false, false).size
    }
  }


  test("outerJoinSortedIters") {
    assert(outerJoinSortedIters(Seq.empty[(Int, Long)].iterator, Seq.empty[(Int, Short)].iterator).isEmpty)

    assert(outerJoinSortedIters(Iterator((1, 2L), (2, -3L)), Seq.empty[(Int, Boolean)].iterator).toArray
      === Array((1, Some(2L), None), (2, Some(-3L), None)))

    assert(outerJoinSortedIters(Iterator((1, 2L), (2, -3L)), Iterator((0, 9L), (1, -2L), (3, -3L))).toArray
      === Array((0, None, Some(9L)), (1, Some(2L), Some(-2L)), (2, Some(-3L), None), (3, None, Some(-3L))))
  }


  test("innerJoinSortedIters") {
    assert(innerJoinSortedIters(Seq.empty[(Int, Long)].iterator, Seq.empty[(Int, Boolean)].iterator).isEmpty)

    assert(innerJoinSortedIters(Iterator((1, 2L), (2, -3L)), Seq.empty[(Int, Boolean)].iterator).isEmpty)

    assert(innerJoinSortedIters(Iterator((1, 2L), (2, -3L)), Iterator((0, 9L), (1, -2L), (3, -3L))).toArray
      === Array((1, 2L, -2L)))
  }


  test("reduceByKey") {
    assert(reduceByKey[Int, Long](Seq.empty[(Int, Long)].iterator, _ + _).isEmpty)

    assert(reduceByKey[Int, Long](Iterator((0, 1L), (0, 2L), (2, 3L), (0, 6L), (0, 1L)), _ + _).toArray
      === Array((0, 3L), (2, 3L), (0, 7L)))
  }


  test("aggregateByKey") {
    assert(aggregateByKey[Int, Long, mutable.Set[Long]](Seq.empty[(Int, Long)].iterator,
      () => mutable.TreeSet.empty[Long], _ += _).isEmpty)

    assert(aggregateByKey[Int, Long, mutable.Set[Long]](Iterator((0, 1L), (0, 2L), (2, 3L), (0, 6L), (0, 1L)),
      () => mutable.TreeSet.empty[Long], _ += _).toArray
      === Array((0, Set(1L, 2L)), (2, Set(3L)), (0, Set(1L, 6L))))
  }


  test("partialAggregate") {
    val buff = mutable.ArrayBuffer.empty[Int]

    assert(partialAggregate[Int, mutable.ArrayBuffer[Int], Array[Int]](Seq.empty[Int].iterator,
      () => {
        buff.clear()
        buff
      },
      _ += _,
      (buff: mutable.ArrayBuffer[Int], index: Long, partIndex: Long) => buff.size >= 3,
      (buff: mutable.ArrayBuffer[Int]) => Iterator(buff.toArray)).isEmpty)

    buff.clear()

    assert(partialAggregate[Int, mutable.ArrayBuffer[Int], Array[Int]](Iterator.range(0, 9),
      () => {
        buff.clear()
        buff
      },
      _ += _,
      (buff: mutable.ArrayBuffer[Int], index: Long, partIndex: Long) => buff.size >= 3,
      (buff: mutable.ArrayBuffer[Int]) => Iterator(buff.toArray)).toArray
      === Array.range(0, 9).grouped(3).toArray)
  }


  test("getTotalIter") {
    assert(getTotalIter(Vectors.dense(Array(1.0, 2.0))).toArray === Array((0, 1.0), (1, 2.0)))

    assert(getTotalIter(Vectors.sparse(5, Array.emptyIntArray, Array.emptyDoubleArray)).toArray
      === Array((0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0)))

    assert(getTotalIter(Vectors.sparse(5, Array(1), Array(2.0))).toArray
      === Array((0, 0.0), (1, 2.0), (2, 0.0), (3, 0.0), (4, 0.0)))
  }


  test("getActiveIter") {
    assert(getActiveIter(Vectors.dense(Array(1.0, 0.0, 2.0))).toArray === Array((0, 1.0), (2, 2.0)))

    assert(getActiveIter(Vectors.sparse(5, Array.emptyIntArray, Array.emptyDoubleArray)).isEmpty)

    assert(getActiveIter(Vectors.sparse(5, Array(1, 3), Array(2.0, 0.0))).toArray
      === Array((1, 2.0)))
  }
}
