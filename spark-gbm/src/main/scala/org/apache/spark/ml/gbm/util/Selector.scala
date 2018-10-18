package org.apache.spark.ml.gbm.util

import java.{util => ju}

import scala.util.Random

import org.apache.spark.unsafe.hash.Murmur3_x86_32


/**
  * Indicator that indicate whether:
  * 1, a tree contains a column in column-sampling (ByTree or/and ByLevel)
  * 2, or, a tree contains a row or block in sub-sampling
  */
private[gbm] trait Selector extends Serializable {

  def size: Int

  def contains[T, C](treeId: T, index: C)
                    (implicit int: Integral[T], inc: Integral[C]): Boolean
}


private[gbm] object Selector extends Serializable {

  /**
    * Initialize a new selector based on given parameters.
    * Note: Trees in a same base model should share the same selector.
    */
  def create(sampleRate: Double,
             numKeys: Long,
             numBaseModels: Int,
             rawSize: Int,
             seed: Long): Selector = {

    if (sampleRate == 1) {
      TrueSelector(numBaseModels * rawSize)

    } else if (numKeys * sampleRate > 32) {
      val rng = new Random(seed)
      val maximum = (Int.MaxValue * sampleRate).ceil.toInt

      val seeds = Array.range(0, numBaseModels).flatMap { i =>
        val s = rng.nextInt
        Iterator.fill(rawSize)(s)
      }

      HashSelector(maximum, seeds)

    } else {
      // When size of selected columns is small, it is hard for hashing to perform robust sampling,
      // we then switch to `SetSelector` for exactly sampling.
      val rng = new Random(seed)
      val numSelected = (numKeys * sampleRate).ceil.toInt

      val sets = Array.range(0, numBaseModels).flatMap { i =>
        val selected = rng.shuffle(Seq.range(0, numKeys))
          .take(numSelected).toArray.sorted
        Iterator.fill(rawSize)(selected)
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

  override def contains[T, C](treeId: T, index: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    val t = int.toInt(treeId)
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

  override def contains[T, C](treeId: T, index: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    Murmur3_x86_32.hashLong(inc.toLong(index), seeds(int.toInt(treeId))).abs < maximum
  }

  override def toString: String = {
    s"HashSelector(maximum: $maximum, seeds: ${seeds.mkString("[", ",", "]")})"
  }
}


private[gbm] case class SetSelector(sets: Array[Array[Long]]) extends Selector {
  require(sets.nonEmpty)
  require(sets.forall(set => Utils.validateOrdering[Long](set.iterator).size > 0))

  override def size: Int = sets.length

  override def contains[T, C](treeId: T, index: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    ju.Arrays.binarySearch(sets(int.toInt(treeId)), inc.toLong(index)) >= 0
  }

  override def toString: String = {
    s"SetSelector(sets: ${sets.mkString("{", ",", "}")})"
  }
}


private[gbm] case class UnionSelector(selectors: Seq[Selector]) extends Selector {
  require(selectors.nonEmpty)

  override def size: Int = selectors.head.size

  override def contains[T, C](treeId: T, index: C)
                             (implicit int: Integral[T], inc: Integral[C]): Boolean = {
    selectors.forall(_.contains[T, C](treeId, index))
  }

  override def toString: String = {
    s"UnionSelector(selectors: ${selectors.mkString("[", ",", "]")})"
  }
}


