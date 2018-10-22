package org.apache.spark.ml.gbm.util

import java.{util => ju}

import scala.util.Random

import org.apache.spark.unsafe.hash.Murmur3_x86_32


/**
  * Indicator that indicate whether a set contains a value.
  * This can be used in:
  * 1, indicate whether a tree contains column in column-sampling (ByTree or/and ByLevel)
  * 2, indicate whether a tree contains a row or block in sub-sampling
  */
private[gbm] trait Selector extends Serializable {

  def size: Int

  def contains[T, C](setId: T, value: C)
                    (implicit int: Integral[T], inc: Integral[C]): Boolean
}


private[gbm] object Selector extends Serializable {

  /**
    * Initialize a new selector based on given parameters.
    * Note: Trees in a same base model should share the same selector.
    */
  def create(rate: Double,
             cardinality: Long,
             distinct: Int,
             replica: Int,
             seed: Long): Selector = {

    if (rate == 1) {
      TrueSelector(distinct * replica)

    } else if (cardinality * rate > 32) {
      val rng = new Random(seed)
      val maximum = (Int.MaxValue * rate).ceil.toInt

      val seeds = Array.range(0, distinct).flatMap { i =>
        val s = rng.nextInt
        Iterator.fill(replica)(s)
      }

      HashSelector(maximum, seeds)

    } else {
      // When size of selected elements is small, it is hard for hashing to perform robust sampling,
      // we then switch to `SetSelector` for exactly sampling.
      val rng = new Random(seed)
      val numSelected = (cardinality * rate).ceil.toInt

      val sets = Array.range(0, distinct).flatMap { i =>
        val selected = rng.shuffle(Seq.range(0, cardinality))
          .take(numSelected).toArray.sorted
        Iterator.fill(replica)(selected)
      }

      SetSelector(sets)
    }
  }

  /**
    * Merge several selectors into one, will skip redundant `TrueSelector`.
    */
  def union(selectors: Selector*): Selector = {
    require(selectors.nonEmpty)
    require(selectors.map(_.size).distinct.size == 1)

    val nonTrues = selectors.flatMap {
      case s: TrueSelector => Iterator.empty
      case s => Iterator.single(s)
    }

    if (nonTrues.nonEmpty) {
      UnionSelector(nonTrues)
    } else {
      selectors.head
    }
  }
}


private[gbm] case class TrueSelector(size: Int) extends Selector {

  override def contains[T, C](setId: T, value: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    val t = int.toInt(setId)
    require(t >= 0 && t < size)
    true
  }

  override def toString: String = {
    s"TrueSelector(size: $size)"
  }
}


private[gbm] case class HashSelector(maximum: Int,
                                     seeds: Array[Int]) extends Selector {
  require(maximum >= 0)

  override def size: Int = seeds.length

  override def contains[T, C](setId: T, value: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    Murmur3_x86_32.hashLong(inc.toLong(value), seeds(int.toInt(setId))).abs < maximum
  }

  override def toString: String = {
    s"HashSelector(maximum: $maximum, seeds: ${seeds.mkString("[", ",", "]")})"
  }
}


private[gbm] case class SetSelector(sets: Array[Array[Long]]) extends Selector {
  require(sets.nonEmpty)
  require(sets.forall(set => Utils.validateOrdering[Long](set.iterator).size > 0))

  override def size: Int = sets.length

  override def contains[T, C](setId: T, value: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    ju.Arrays.binarySearch(sets(int.toInt(setId)), inc.toLong(value)) >= 0
  }

  override def toString: String = {
    s"SetSelector(sets: ${sets.mkString("{", ",", "}")})"
  }
}


private[gbm] case class UnionSelector(selectors: Seq[Selector]) extends Selector {
  require(selectors.nonEmpty)

  override def size: Int = selectors.head.size

  override def contains[T, C](setId: T, value: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    selectors.forall(_.contains[T, C](setId, value))
  }

  override def toString: String = {
    s"UnionSelector(selectors: ${selectors.mkString("[", ",", "]")})"
  }
}


