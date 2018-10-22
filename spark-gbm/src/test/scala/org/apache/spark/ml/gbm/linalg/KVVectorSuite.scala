package org.apache.spark.ml.gbm.linalg

import org.scalatest.FunSuite

class KVVectorSuite extends FunSuite {

  import KVVector._

  test("basic") {
    val v1 = dense[Short, Int](Array(1, 3, 0, 5))
    val v2 = sparse[Short, Int](5, Array(2.toShort, 3.toShort, 4.toShort), Array(1, 0, -1))
    val v3 = empty[Short, Int]

    assert(v1.isDense)
    assert(v1.nonEmpty)
    assert(v1.size === 4)
    assert(v1.nnz === 3)
    assert(v1(0) === 1)
    assert(v1(1) === 3)
    assert(v1(2) === 0)
    assert(v1(3) === 5)
    assert(v1.toArray === Array(1, 3, 0, 5))
    assert(v1.iterator.toArray === Array((0.toShort, 1), (1.toShort, 3), (2.toShort, 0), (3.toShort, 5)))
    assert(v1.activeIterator.toArray === Array((0.toShort, 1), (1.toShort, 3), (3.toShort, 5)))

    assert(v2.isSparse)
    assert(v2.nonEmpty)
    assert(v2.size === 5)
    assert(v2.nnz === 2)
    assert(v2(0) === 0)
    assert(v2(1) === 0)
    assert(v2(2) === 1)
    assert(v2(3) === 0)
    assert(v2(4) === -1)
    assert(v2.toArray === Array(0, 0, 1, 0, -1))
    assert(v2.iterator.toArray === Array((0.toShort, 0), (1.toShort, 0), (2.toShort, 1), (3.toShort, 0), (4.toShort, -1)))
    assert(v2.activeIterator.toArray === Array((2.toShort, 1), (4.toShort, -1)))

    assert(v3.isEmpty)
    assert(v3.size === 0)
    assert(v3.iterator.isEmpty)
    assert(v3.activeIterator.isEmpty)
  }


  test("negate") {
    val v1 = dense[Short, Int](Array(1, 3, 0, 5))
    val v2 = sparse[Short, Int](5, Array(2.toShort, 3.toShort, 4.toShort), Array(1, 0, -1))
    val v3 = empty[Short, Int]

    assert(v1.negate.toArray === Array(-1, -3, 0, -5))
    assert(v2.negate.toArray === Array(0, 0, -1, 0, 1))
    assert(v3.negate.iterator.isEmpty)
  }


  test("plus/minus") {
    val v1 = dense[Short, Int](Array(1, 3, 0, 5))
    val v2 = sparse[Short, Int](5, Array(2.toShort, 3.toShort, 4.toShort), Array(1, 0, -1))
    val v3 = empty[Short, Int]

    assert(v1.copy.plus(0.toShort, 1).toArray === Array(2, 3, 0, 5))
    assert(v1.copy.plus(7.toShort, 10).toArray === Array(1, 3, 0, 5, 0, 0, 0, 10))
    assert(v1.copy.minus(0.toShort, 1).toArray === Array(0, 3, 0, 5))
    assert(v1.copy.minus(7.toShort, 10).toArray === Array(1, 3, 0, 5, 0, 0, 0, -10))

    assert(v2.copy.plus(0.toShort, 1).toArray === Array(1, 0, 1, 0, -1))
    assert(v2.copy.plus(2.toShort, 1).toArray === Array(0, 0, 2, 0, -1))
    assert(v2.copy.plus(7.toShort, 10).toArray === Array(0, 0, 1, 0, -1, 0, 0, 10))
    assert(v2.copy.minus(0.toShort, 1).toArray === Array(-1, 0, 1, 0, -1))
    assert(v2.copy.minus(2.toShort, 1).toArray === Array(0, 0, 0, 0, -1))
    assert(v2.copy.minus(7.toShort, 10).toArray === Array(0, 0, 1, 0, -1, 0, 0, -10))

    assert(v3.copy.plus(0.toShort, 1).toArray === Array(1))
    assert(v3.copy.plus(3.toShort, 1).toArray === Array(0, 0, 0, 1))
    assert(v3.copy.minus(0.toShort, 1).toArray === Array(-1))
    assert(v3.copy.minus(3.toShort, 1).toArray === Array(0, 0, 0, -1))

    assert(v1.copy.plus(dense[Short, Int](Array(-1, -3, 1))).toArray === Array(0, 0, 1, 5))
    assert(v1.copy.plus(dense[Short, Int](Array(-1, -3, 1, 0, 1))).toArray === Array(0, 0, 1, 5, 1))
    assert(v1.copy.plus(sparse[Short, Int](3, Array(1.toShort), Array(2))).toArray === Array(1, 5, 0, 5))
    assert(v1.copy.plus(sparse[Short, Int](7, Array(1.toShort), Array(2))).toArray === Array(1, 5, 0, 5, 0, 0, 0))

    assert(v2.copy.plus(dense[Short, Int](Array(-1, -3, 1))).toArray === Array(-1, -3, 2, 0, -1))
    assert(v2.copy.plus(dense[Short, Int](Array(-1, -3, 1, 0, 1, 4))).toArray === Array(-1, -3, 2, 0, 0, 4))
    assert(v2.copy.plus(sparse[Short, Int](3, Array(1.toShort), Array(2))).toArray === Array(0, 2, 1, 0, -1))
    assert(v2.copy.plus(sparse[Short, Int](7, Array(1.toShort), Array(2))).toArray === Array(0, 2, 1, 0, -1, 0, 0))

    assert(v3.copy.plus(dense[Short, Int](Array(-1, -3, 1))).toArray === Array(-1, -3, 1))
    assert(v3.copy.plus(sparse[Short, Int](3, Array(1.toShort), Array(2))).toArray === Array(0, 2, 0))
  }


  test("slice") {
    val v1 = dense[Short, Int](Array(1, 3, 0, 5, 9, 7))
    val v2 = sparse[Short, Int](5, Array(2.toShort, 3.toShort, 4.toShort), Array(1, 0, -1))

    assert(v1.slice(Array(0, 3, 4)).toArray === Array(1, 5, 9))
    assert(v2.slice(Array(0, 3, 4)).toArray === Array(0, 0, -1))
  }


  test("compress") {
    val v1 = dense[Short, Int](Array(1, 3, 0, 5, 9, 7))
    val v2 = dense[Short, Int](Array(1, 0, 3, 0, 0, 0, 5, 9, 7, 0, 0, 0, 0))
    val v3 = sparse[Short, Int](5, Array(2.toShort, 3.toShort, 4.toShort), Array(1, 2, -1))
    val v4 = sparse[Short, Int](15, Array(2.toShort, 3.toShort, 4.toShort), Array(1, 2, -1))

    assert(v1.compress.isDense)
    assert(v1.compress.toArray === v1.toArray)

    assert(v2.compress.isSparse)
    assert(v2.compress.toArray === v2.toArray)

    assert(v3.compress.isDense)
    assert(v3.compress.toArray === v3.toArray)

    assert(v4.compress.isSparse)
    assert(v4.compress.toArray === v4.toArray)
  }
}

