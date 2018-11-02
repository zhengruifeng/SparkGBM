package org.apache.spark.ml.gbm.util

import java.{util => ju}

import scala.reflect.ClassTag
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

  def containsById[K, V](setId: K, value: V)
                        (implicit ink: Integral[K], inv: Integral[V]): Boolean

  def contains[K, V](setIds: Array[K], value: V)
                    (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    setIds.exists { setId => containsById[K, V](setId, value) }
  }

  def contains[V](value: V)
                 (implicit inv: Integral[V]): Boolean = {
    contains[Int, V](Array.range(0, size), value)
  }

  def index[K, V](setIds: Array[K], value: V)
                 (implicit ck: ClassTag[K], ink: Integral[K],
                  inv: Integral[V]): Array[K] = {
    setIds.filter { setId => containsById[K, V](setId, value) }
  }

  def index[K, V](value: V)
                 (implicit ck: ClassTag[K], ink: Integral[K],
                  inv: Integral[V]): Array[K] = {
    index[Int, V](Array.range(0, size), value)
      .map(ink.fromInt)
  }
}


private[gbm] object Selector extends Serializable {

  /**
    * Initialize a new selector based on given parameters.
    * Note: Trees in a same base model should share the same selector.
    */
  def create(rate: Double,
             cardinality: Long,
             numDistinctSets: Int,
             numReplicated: Int,
             seed: Long): Selector = {
    require(rate >= 0 && rate <= 1)
    require(cardinality > 0)
    require(numDistinctSets > 0)
    require(numReplicated > 0)

    if (rate == 1) {
      TrueSelector(numDistinctSets * numReplicated)

    } else if (cardinality * rate * numDistinctSets * numReplicated <= 4096) {
      // When size of selected elements is small, it is hard for hashing to perform robust sampling,
      // we then switch to `SetSelector` for exactly sampling.
      val rng = new Random(seed)
      val numSelected = (cardinality * rate).ceil.toInt

      val sets = Array.range(0, numDistinctSets)
        .flatMap { _ =>
          val selected = rng.shuffle(Seq.range(0, cardinality))
            .take(numSelected).toArray.sorted
          Iterator.fill(numReplicated)(selected)
        }

      SetSelector(sets)

    } else {
      val rng = new Random(seed)
      val maximum = (Int.MaxValue * rate).ceil.toInt

      val seeds = Array.range(0, numDistinctSets)
        .flatMap { _ =>
          val s = rng.nextInt
          Iterator.fill(numReplicated)(s)
        }

      HashSelector(maximum, seeds)
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

  override def containsById[K, V](setId: K, value: V)
                                 (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    val t = ink.toInt(setId)
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

  override def containsById[K, V](setId: K, value: V)
                                 (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    Murmur3_x86_32.hashLong(inv.toLong(value), seeds(ink.toInt(setId))).abs < maximum
  }

  override def contains[K, V](setIds: Array[K], value: V)
                             (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    if (setIds.nonEmpty) {
      val value_ = inv.toLong(value)

      var seed = seeds(ink.toInt(setIds.head))
      var ret = Murmur3_x86_32.hashLong(value_, seed).abs < maximum

      var i = 1
      while (i < setIds.length && !ret) {
        val seed_ = seeds(ink.toInt(setIds(i)))
        if (seed_ != seed) {
          seed = seed_
          ret = Murmur3_x86_32.hashLong(value_, seed).abs < maximum
        }
        i += 1
      }

      ret

    } else {
      false
    }
  }

  override def index[K, V](setIds: Array[K], value: V)
                          (implicit ck: ClassTag[K], ink: Integral[K],
                           inv: Integral[V]): Array[K] = {

    if (setIds.nonEmpty) {
      val value_ = inv.toLong(value)

      var seed = seeds(ink.toInt(setIds.head))
      var ret = Murmur3_x86_32.hashLong(value_, seed).abs < maximum

      val rets = Array.ofDim[Boolean](setIds.length)
      rets(0) = ret

      var i = 1
      while (i < setIds.length) {
        val seed_ = seeds(ink.toInt(setIds(i)))
        if (seed_ != seed) {
          seed = seed_
          ret = Murmur3_x86_32.hashLong(value_, seed).abs < maximum
        }
        rets(i) = ret
        i += 1
      }

      setIds.zip(rets).filter(_._2).map(_._1)

    } else {
      Array.empty
    }
  }

  override def toString: String = {
    s"HashSelector(maximum: $maximum, seeds: ${seeds.mkString("[", ",", "]")})"
  }
}


private[gbm] case class SetSelector(sets: Array[Array[Long]]) extends Selector {
  require(sets.nonEmpty)
  require(sets.forall(set => Utils.validateOrdering[Long](set.iterator).size > 0))

  override def size: Int = sets.length

  override def containsById[K, V](setId: K, value: V)
                                 (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    ju.Arrays.binarySearch(sets(ink.toInt(setId)), inv.toLong(value)) >= 0
  }

  override def toString: String = {
    s"SetSelector(sets: ${sets.map(_.mkString("(", ",", ")")).mkString("[", ",", "]")})"
  }
}


private[gbm] case class UnionSelector(selectors: Seq[Selector]) extends Selector {
  require(selectors.nonEmpty)

  override def size: Int = selectors.head.size

  override def containsById[K, V](setId: K, value: V)
                                 (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    selectors.forall(_.containsById[K, V](setId, value))
  }

  override def toString: String = {
    s"UnionSelector(selectors: ${selectors.mkString("[", ",", "]")})"
  }
}


