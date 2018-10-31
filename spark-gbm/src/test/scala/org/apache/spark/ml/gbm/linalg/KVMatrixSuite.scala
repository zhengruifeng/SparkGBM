package org.apache.spark.ml.gbm.linalg

import org.scalatest.FunSuite

class KVMatrixSuite extends FunSuite {

  import KVMatrix._


  test("basic") {
    val vecs = Seq(
      KVVector.sparse[Int, Long](5, Array(3), Array(1L)),
      KVVector.dense[Int, Long](Array(0L, 1L, 2L, 3L, 4L)),
      KVVector.dense[Int, Long](Array(0L, -1L, -2L, -3L, -4L)),
      KVVector.sparse[Int, Long](5, Array(2), Array(-1L)))
    val matrix = build(vecs)

    assert(matrix.size === 4)
    assert(matrix.iterator.map(_.isDense).toArray === Array(false, true, true, false))
    assert(matrix.iterator.map(_.toArray).toArray ===
      Array(Array(0L, 0L, 0L, 1L, 0L),
        Array(0L, 1L, 2L, 3L, 4L),
        Array(0L, -1L, -2L, -3L, -4L),
        Array(0L, 0L, -1L, 0L, 0L)))
  }


  test("all vectors are dense") {
    val vecs = Seq(
      KVVector.dense[Int, Long](Array(0L, 0L, 0L, 1L, 0L)),
      KVVector.dense[Int, Long](Array(0L, 1L, 2L, 3L, 4L)),
      KVVector.dense[Int, Long](Array(0L, -1L, -2L, -3L, -4L)),
      KVVector.dense[Int, Long](Array(0L, 0L, -1L, 0L, 0L)))
    val matrix = build(vecs)

    assert(matrix.size === 4)
    assert(matrix.iterator.map(_.isDense).toArray === Array(true, true, true, true))
    assert(matrix.iterator.map(_.toArray).toArray ===
      Array(Array(0L, 0L, 0L, 1L, 0L),
        Array(0L, 1L, 2L, 3L, 4L),
        Array(0L, -1L, -2L, -3L, -4L),
        Array(0L, 0L, -1L, 0L, 0L)))
  }


  test("all vectors are empty") {
    val iter = Iterator.range(0, 10).map(_ => KVVector.empty[Int, Long])
    val matrix = build(iter)

    assert(matrix.size === 10)
    assert(matrix.iterator.forall(_.isEmpty))
  }


  test("empty input") {
    val seq = Seq.empty[KVVector[Int, Long]]
    val matrix = build(seq)

    assert(matrix.isEmpty)
  }


  test("slice") {
    val vecs = Seq(
      KVVector.sparse[Int, Long](5, Array(3), Array(1L)),
      KVVector.dense[Int, Long](Array(0L, 1L, 2L, 3L, 4L)),
      KVVector.dense[Int, Long](Array(0L, -1L, -2L, -3L, -4L)),
      KVVector.sparse[Int, Long](5, Array(2), Array(-1L)))
    val matrix = sliceAndBuild(Array(2, 4), vecs)

    assert(matrix.iterator.map(_.toArray).toArray ===
      Array(Array(0L, 0L, 0L, 0L, 0L),
        Array(0L, 0L, 2L, 0L, 4L),
        Array(0L, 0L, -2L, 0L, -4L),
        Array(0L, 0L, -1L, 0L, 0L)))
  }
}

