package org.apache.spark.ml.gbm.util

import java.{util => ju}

import scala.collection.mutable
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
             numDistinct: Int,
             numReplicas: Int,
             seed: Long): Selector = {
    require(rate >= 0 && rate <= 1)
    require(cardinality > 0)
    require(numDistinct > 0)
    require(numReplicas > 0)

    if (rate == 1) {
      TrueSelector(numDistinct * numReplicas)

    } else if (cardinality > (1 << 16) || cardinality * rate * numDistinct > (1 << 12)) {
      val rng = new Random(seed)
      val maximum = (Int.MaxValue * rate).ceil.toInt
      val seeds = Array.tabulate(numDistinct)(_ => rng.nextInt)

      HashSelector(maximum, seeds, numReplicas)

    } else {
      // When size of selected elements is small, it is hard for hashing to perform robust sampling,
      // we then switch to `SetSelector` for exactly sampling.
      val rng = new Random(seed)
      val numSelected = (cardinality * rate).ceil.toInt
      val sets = Array.tabulate(numDistinct)(_ => rng.shuffle(Seq.range(0, cardinality)).take(numSelected).toArray.sorted)

      SetSelector(sets, numReplicas)
    }
  }


  /**
    * Merge several selectors into one, will skip redundant `TrueSelector`.
    */
  def union(selectors: Selector*): Selector = {
    require(selectors.nonEmpty)
    require(selectors.map(_.size).distinct.size == 1, s"selectors: ${selectors.mkString(",")}")

    val nonTrues = selectors.flatMap {
      case s: TrueSelector => Iterator.empty
      case s => Iterator.single(s)
    }.toArray

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
                                     seeds: Array[Int],
                                     numReplicas: Int) extends Selector {
  require(maximum >= 0)
  require(seeds.nonEmpty)
  require(numReplicas > 0)

  override def size: Int = seeds.length * numReplicas

  override def containsById[K, V](setId: K, value: V)
                                 (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    val distinctId = ink.toInt(setId) / numReplicas
    Murmur3_x86_32.hashLong(inv.toLong(value), seeds(distinctId)).abs < maximum
  }

  override def contains[K, V](setIds: Array[K], value: V)
                             (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    if (setIds.nonEmpty) {
      val value2 = inv.toLong(value)

      setIds.map { setId => ink.toInt(setId) / numReplicas }
        .distinct
        .exists { distinctId => Murmur3_x86_32.hashLong(value2, seeds(distinctId)).abs < maximum }

    } else {
      false
    }
  }

  override def index[K, V](setIds: Array[K], value: V)
                          (implicit ck: ClassTag[K], ink: Integral[K],
                           inv: Integral[V]): Array[K] = {

    if (setIds.nonEmpty) {
      val value2 = inv.toLong(value)

      val builder = mutable.ArrayBuilder.make[K]

      var prevDistinctId = -1
      var prevResult = false

      var i = 0
      while (i < setIds.length) {
        val setId = setIds(i)
        val distinctId = ink.toInt(setId) / numReplicas

        if (distinctId != prevDistinctId) {
          prevDistinctId = distinctId
          prevResult = Murmur3_x86_32.hashLong(value2, seeds(distinctId)).abs < maximum
        }

        if (prevResult) {
          builder += setId
        }

        i += 1
      }

      builder.result()

    } else {
      Array.empty
    }
  }

  override def toString: String = {
    s"HashSelector(maximum: $maximum, seeds: ${seeds.mkString("[", ",", "]")}, numReplicas: $numReplicas)"
  }
}


private[gbm] case class SetSelector(sets: Array[Array[Long]],
                                    numReplicas: Int) extends Selector {
  require(sets.nonEmpty)
  require(sets.forall(set => Utils.validateOrdering[Long](set.iterator).size > 0))
  require(numReplicas > 0)

  override def size: Int = sets.length * numReplicas

  override def containsById[K, V](setId: K, value: V)
                                 (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    val distinctId = ink.toInt(setId) / numReplicas
    ju.Arrays.binarySearch(sets(distinctId), inv.toLong(value)) >= 0
  }

  override def contains[K, V](setIds: Array[K], value: V)
                             (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    if (setIds.nonEmpty) {
      val value2 = inv.toLong(value)

      setIds.map { setId => ink.toInt(setId) / numReplicas }
        .distinct
        .exists { distinctId => ju.Arrays.binarySearch(sets(distinctId), inv.toLong(value)) >= 0 }

    } else {
      false
    }
  }

  override def index[K, V](setIds: Array[K], value: V)
                          (implicit ck: ClassTag[K], ink: Integral[K],
                           inv: Integral[V]): Array[K] = {

    if (setIds.nonEmpty) {
      val value2 = inv.toLong(value)

      val builder = mutable.ArrayBuilder.make[K]

      var prevDistinctId = -1
      var prevResult = false

      var i = 0
      while (i < setIds.length) {
        val setId = setIds(i)
        val distinctId = ink.toInt(setId) / numReplicas

        if (distinctId != prevDistinctId) {
          prevDistinctId = distinctId
          prevResult = ju.Arrays.binarySearch(sets(distinctId), value2) >= 0
        }

        if (prevResult) {
          builder += setId
        }

        i += 1
      }

      builder.result()

    } else {
      Array.empty
    }
  }

  override def toString: String = {
    s"SetSelector(sets: ${sets.map(_.mkString("(", ",", ")")).mkString("[", ",", "]")}, numReplicas: $numReplicas)"
  }
}


private[gbm] case class UnionSelector(selectors: Array[Selector]) extends Selector {
  require(selectors.nonEmpty)

  override def size: Int = selectors.head.size

  override def containsById[K, V](setId: K, value: V)
                                 (implicit ink: Integral[K], inv: Integral[V]): Boolean = {
    selectors.forall(_.containsById[K, V](setId, value))
  }

  override def index[K, V](setIds: Array[K], value: V)
                          (implicit ck: ClassTag[K], ink: Integral[K],
                           inv: Integral[V]): Array[K] = {

    if (setIds.nonEmpty) {
      var result = setIds
      var i = 0
      while (i < selectors.length && result.nonEmpty) {
        result = selectors(i).index[K, V](result, value)
        i += 1
      }
      result

    } else {
      Array.empty
    }
  }

  override def toString: String = {
    s"UnionSelector(selectors: ${selectors.mkString("[", ",", "]")})"
  }
}


