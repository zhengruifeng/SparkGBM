package org.apache.spark.ml.gbm

import scala.collection.mutable

abstract class Split extends Serializable {

  def featureId: Int

  def missingGoLeft: Boolean

  def goLeft(bins: Array[Int]): Boolean

  def leftWeight: Double

  def rightWeight: Double

  def gain: Double
}

class SeqSplit(val featureId: Int,
               val missingGoLeft: Boolean,
               val gain: Double,
               val leftWeight: Double,
               val rightWeight: Double,
               val threshold: Int) extends Split {

  override def goLeft(bins: Array[Int]): Boolean = {
    val bin = bins(featureId)
    if (bin == 0) {
      missingGoLeft
    } else {
      bin <= threshold
    }
  }
}


class SetSplit(val featureId: Int,
               val missingGoLeft: Boolean,
               val gain: Double,
               val leftWeight: Double,
               val rightWeight: Double,
               val leftSet: Array[Int]) extends Split {

  override def goLeft(bins: Array[Int]): Boolean = {
    val bin = bins(featureId)
    if (bin == 0) {
      missingGoLeft
    } else {
      java.util.Arrays.binarySearch(leftSet, bin) >= 0
    }
  }
}


private[gbm] object Split {

  /**
    * find the best split
    *
    * @param featureId   feature index
    * @param hist        histogram
    * @param boostConfig boosting config info
    * @param treeConfig  tree config info
    * @return best split if any
    */
  def split(featureId: Int,
            hist: Array[Double],
            boostConfig: BoostConfig,
            treeConfig: TreeConfig): Option[Split] = {
    require(hist.length % 2 == 0)

    if (hist.length <= 2) {
      return None
    }

    val gradSeq = Array.range(0, hist.length, 2).map(hist)
    val hessSeq = Array.range(1, hist.length, 2).map(hist)

    val nnz = gradSeq.zip(hessSeq)
      .count { case (g, h) =>
        g != 0 || h != 0
      }

    if (nnz <= 1) {
      return None
    }

    val split = if (treeConfig.isSeq(featureId)) {
      splitSeq(featureId, gradSeq, hessSeq, boostConfig)
    } else if (nnz <= boostConfig.getMaxBruteBins) {
      splitSetBrute(featureId, gradSeq, hessSeq, boostConfig)
    } else {
      splitSetHeuristic(featureId, gradSeq, hessSeq, boostConfig)
    }

    if (split.isEmpty) {
      return None
    }

    val values = Array(split.get.gain, split.get.leftWeight, split.get.rightWeight)
    if (validate(values)) {
      split
    } else {
      None
    }
  }

  /**
    * validate values for numerical stability
    *
    * @param values numbers
    * @return true is all numbers are ok
    */
  def validate(values: Array[Double]): Boolean = {
    !values.exists(v => v.isInfinity || v.isNaN)
  }

  /**
    * sequentially search the best split, with specially dealing with missing value
    *
    * @param featureId   feature index
    * @param gradSeq     grad array
    * @param hessSeq     hess array
    * @param boostConfig boosting config info
    * @return best split if any
    */
  def splitSeq(featureId: Int,
               gradSeq: Array[Double],
               hessSeq: Array[Double],
               boostConfig: BoostConfig): Option[SeqSplit] = {
    /** missing go left */
    /** find best split on indices of [i0, i1, i2, i3, i4] */
    val search1 = seqSearch(gradSeq, hessSeq, boostConfig)

    val search2 = if (gradSeq.head == 0 && hessSeq.head == 0) {
      /** if hist of missing value is zero, do not need to place missing value to the right side */
      None
    } else {
      /** missing go right */
      /** find best split on indices of [i1, i2, i3, i4, i0] */
      seqSearch(gradSeq.tail :+ gradSeq.head, hessSeq.tail :+ hessSeq.head, boostConfig)
    }

    (search1, search2) match {
      case (Some((cut1, gain1, weightLeft1, weightRight1)), Some((cut2, gain2, weightLeft2, weightRight2))) =>
        if (gain1 >= gain2) {
          Some(new SeqSplit(featureId, true, gain1, weightLeft1, weightRight1, cut1))
        } else {
          /** adjust the cut of split2 */
          /** cut = 2, [i1, i2, i3 | i4, i0] -> cut = 3 */
          Some(new SeqSplit(featureId, false, gain2, weightLeft2, weightRight2, cut2 + 1))
        }

      case (Some((cut, gain, weightLeft, weightRight)), None) =>
        Some(new SeqSplit(featureId, true, gain, weightLeft, weightRight, cut))

      case (None, Some((cut, gain, weightLeft, weightRight))) =>
        Some(new SeqSplit(featureId, false, gain, weightLeft, weightRight, cut + 1))

      case _ => None
    }
  }


  /**
    * Heuristically find the best set split
    *
    * @param featureId   feature index
    * @param gradSeq     grad array
    * @param hessSeq     hess array
    * @param boostConfig boosting config info
    * @return best split if any
    */
  def splitSetHeuristic(featureId: Int,
                        gradSeq: Array[Double],
                        hessSeq: Array[Double],
                        boostConfig: BoostConfig): Option[SetSplit] = {
    /** sort the hist according to the relevance of gain */
    /** [g0, g1, g2, g3], [h0, h1, h2, h3] -> [g1, g3, g0, g2], [h1, h3, h0, h2] */
    val epsion = boostConfig.getRegLambda / gradSeq.length
    val (sortedGradSeq, sortedHessSeq, sortedIndices) =
      gradSeq.zip(hessSeq).zipWithIndex
        .map { case ((grad, hess), i) =>
          (grad, hess, i)
        }.sortBy { case (grad, hess, i) =>
        grad / (hess + epsion)
      }.unzip3

    val search = seqSearch(sortedGradSeq, sortedHessSeq, boostConfig)

    if (search.isEmpty) {
      return None
    }

    val (cut, gain, weight1, weight2) = search.get
    val indices1 = sortedIndices.take(cut + 1)

    val split = createSetSplit(featureId, gradSeq, hessSeq, gain, indices1, weight1, weight2)
    Some(split)
  }

  /**
    * Search the best set split by brute force
    *
    * @param featureId   feature index
    * @param gradSeq     grad array
    * @param hessSeq     hess array
    * @param boostConfig boosting config info
    * @return best split if any
    */
  def splitSetBrute(featureId: Int,
                    gradSeq: Array[Double],
                    hessSeq: Array[Double],
                    boostConfig: BoostConfig): Option[SetSplit] = {
    val gradSum = gradSeq.sum
    val hessSum = hessSeq.sum

    val (_, baseScore) = computeScore(gradSum, hessSum, boostConfig)
    if (!validate(Array(baseScore))) {
      return None
    }

    /** ignore the indices with zero hist */
    /** [g0, g1, g2, g3, g4, g5, g6], [h0, h1, h2, h3, h4, h5, h6], [i0, i1, i2, i3, i4, i5, i6] -> */
    /** [g1, g2, g4, g6], [h1, h2, h4, h6], [i1, i2, i4, i6] */
    val nzIndices = gradSeq.zip(hessSeq).zipWithIndex
      .filter { case ((grad, hess), i) =>
        grad != 0 || hess != 0
      }.map(_._2)

    val len = nzIndices.length

    val bestSet1 = mutable.Set[Int]()
    val set1 = mutable.Set[Int]()
    var bestScore = Double.MinValue

    var grad1 = 0.0
    var grad2 = 0.0

    var hess1 = 0.0
    var hess2 = 0.0

    var bestWeight1 = Double.NaN
    var bestWeight2 = Double.NaN

    /** the first element in nnz hist is always unselected in set1 */
    val k = 1L << (len - 1)
    var num = 1L

    val char1 = "1".head
    while (num < k) {
      /** len = 4, num = 3, binStr = "11" -> "0011" & [i1, i2, i4, i6] = [i4, i6] */
      val binStr = num.toBinaryString

      set1.clear()
      grad1 = 0.0
      hess1 = 0.0

      val pad = len - binStr.length
      var i = 0
      while (i < binStr.length) {
        /** len = 4, num = 3, binStr = "11", pad = 2, i = [0, 1] -> nzIndices[2, 3] -> [i4, i6] */
        if (binStr(i) == char1) {
          val index = nzIndices(pad + i)
          grad1 += gradSeq(index)
          hess1 += hessSeq(index)
          set1.add(index)
        }
        i += 1
      }

      grad2 = gradSum - grad1
      hess2 = hessSum - hess1

      if (hess1 >= boostConfig.getMinNodeHess && hess2 >= boostConfig.getMinNodeHess) {

        val (weight1, score1) = computeScore(grad1, hess1, boostConfig)
        val (weight2, score2) = computeScore(grad2, hess2, boostConfig)

        if (validate(Array(weight1, score1, weight2, score2))) {
          val score = score1 + score2
          if (score > bestScore) {
            bestSet1.clear()
            set1.foreach(bestSet1.add)
            bestScore = score
            bestWeight1 = weight1
            bestWeight2 = weight2
          }
        }
      }

      num += 1
    }

    if (!validate(Array(bestScore, bestWeight1, bestWeight2))) {
      return None
    }

    val gain = bestScore - baseScore
    if (gain < boostConfig.getMinGain) {
      return None
    }

    val indices1 = bestSet1.toArray
    val split = createSetSplit(featureId, gradSeq, hessSeq, gain, indices1, bestWeight1, bestWeight2)
    Some(split)
  }


  /**
    * Sequentially search the best split
    *
    * @param gradSeq     grad array
    * @param hessSeq     hess array
    * @param boostConfig boosting config info
    * @return best split if any
    */
  def seqSearch(gradSeq: Array[Double],
                hessSeq: Array[Double],
                boostConfig: BoostConfig): Option[(Int, Double, Double, Double)] = {
    val gradSum = gradSeq.sum
    val hessSum = hessSeq.sum

    val (_, baseScore) = computeScore(gradSum, hessSum, boostConfig)
    if (!validate(Array(baseScore))) {
      return None
    }

    var bestCut = -1
    var bestScore = Double.MinValue

    var gradLeft = 0.0
    var gradRight = 0.0

    var hessLeft = 0.0
    var hessRight = 0.0

    var bestWeightLeft = Double.NaN
    var bestWeightRight = Double.NaN

    (0 until gradSeq.length - 1).foreach { i =>
      if (gradSeq(i) != 0 || hessSeq(i) != 0) {
        gradLeft += gradSeq(i)
        gradRight = gradSum - gradLeft
        hessLeft += hessSeq(i)
        hessRight = hessSum - hessLeft

        if (hessLeft >= boostConfig.getMinNodeHess && hessRight >= boostConfig.getMinNodeHess) {

          val (weightLeft, scoreLeft) = computeScore(gradLeft, hessLeft, boostConfig)
          val (weightRight, scoreRight) = computeScore(gradRight, hessRight, boostConfig)

          if (validate(Array(weightLeft, scoreLeft, weightRight, scoreRight))) {
            val score = scoreLeft + scoreRight
            if (score > bestScore) {
              bestCut = i
              bestScore = score
              bestWeightLeft = weightLeft
              bestWeightRight = weightRight
            }
          }

        }

      }
    }

    if (!validate(Array(bestScore, bestWeightLeft, bestWeightRight))) {
      return None
    }

    val gain = bestScore - baseScore
    if (bestCut >= 0 && gain >= boostConfig.getMinGain) {
      Some((bestCut, gain, bestWeightLeft, bestWeightRight))
    } else {
      None
    }
  }

  /**
    * Compute the weight and score, given the sum of hist.
    *
    * @param gradSum     sum of grad
    * @param hessSum     sum of hess
    * @param boostConfig boosting config info containing the regulization parameters
    * @return weight and score
    */
  def computeScore(gradSum: Double,
                   hessSum: Double,
                   boostConfig: BoostConfig): (Double, Double) = {
    if (boostConfig.getRegAlpha == 0) {
      val weight = -gradSum / (hessSum + boostConfig.getRegLambda)
      val loss = (hessSum + boostConfig.getRegLambda) * weight * weight / 2 + gradSum * weight
      (weight, -loss)

    } else {
      val weight = if (gradSum + boostConfig.getRegAlpha < 0) {
        -(boostConfig.getRegAlpha + gradSum) / (hessSum + boostConfig.getRegLambda)
      } else if (gradSum - boostConfig.getRegAlpha > 0) {
        (boostConfig.getRegAlpha - gradSum) / (hessSum + boostConfig.getRegLambda)
      } else {
        0.0
      }
      val loss = (hessSum + boostConfig.getRegLambda) * weight * weight / 2 + gradSum * weight +
        boostConfig.getRegAlpha * weight.abs
      (weight, -loss)
    }
  }

  /**
    * Given a valid set split, choose the form of SetSplit by the size of set.
    *
    * @param featureId feature index
    * @param gradSeq   grad array
    * @param hessSeq   hess array
    * @param gain      gain
    * @param indices1  indices of raw set1
    * @param weight1   weight assigned to set1
    * @param weight2   weight assigned to set2
    * @return a SetSplit
    */
  def createSetSplit(featureId: Int,
                     gradSeq: Array[Double],
                     hessSeq: Array[Double],
                     gain: Double,
                     indices1: Array[Int],
                     weight1: Double,
                     weight2: Double): SetSplit = {
    require(indices1.max < gradSeq.length)

    /** ignore zero hist */
    val set1 = mutable.Set[Int]()
    indices1.foreach { i =>
      if (gradSeq(i) != 0 || hessSeq(i) != 0) {
        set1.add(i)
      }
    }

    val set2 = mutable.Set[Int]()
    gradSeq.indices.foreach { i =>
      if ((gradSeq(i) != 0 || hessSeq(i) != 0) && !set1.contains(i)) {
        set2.add(i)
      }
    }

    /** remove index of missing value */
    val missingInSet1 = set1.contains(0)
    val missingInSet2 = set2.contains(0)
    if (missingInSet1) {
      set1.remove(0)
    } else if (missingInSet2) {
      set2.remove(0)
    }

    /** choose the smaller set */
    if (set1.size <= set2.size) {
      new SetSplit(featureId, missingInSet1, gain, weight1, weight2, set1.toArray.sorted)
    } else {
      new SetSplit(featureId, missingInSet2, gain, weight2, weight1, set2.toArray.sorted)
    }
  }
}

