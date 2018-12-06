package org.apache.spark.ml.gbm.impl

import java.{lang => jl}
import java.{util => ju}

import scala.annotation.varargs
import scala.collection.mutable
import scala.{specialized => spec}

import org.apache.spark.internal.Logging
import org.apache.spark.ml.gbm._


private[gbm] trait Split extends Serializable {

  def colId: Int

  def left: Boolean

  protected def goByBin[@spec(Byte, Short, Int) B](bin: B)
                                                  (implicit inb: Integral[B]): Boolean

  protected def goLeftByBin[@spec(Byte, Short, Int) B](bin: B)
                                                      (implicit inb: Integral[B]): Boolean = {
    if (left) {
      goByBin[B](bin)
    } else {
      !goByBin[B](bin)
    }
  }

  def goLeft[@spec(Byte, Short, Int) B](bins: Int => B)
                                       (implicit inb: Integral[B]): Boolean = {
    goLeftByBin[B](bins(colId))
  }

  def gain: Float

  def stats: Array[Float]

  def leftWeight: Float = if (left) {
    stats(0)
  } else {
    stats(3)
  }

  def leftGrad: Float = if (left) {
    stats(1)
  } else {
    stats(4)
  }

  def leftHess: Float = if (left) {
    stats(2)
  } else {
    stats(5)
  }

  def rightWeight: Float = if (left) {
    stats(3)
  } else {
    stats(0)
  }

  def rightGrad: Float = if (left) {
    stats(4)
  } else {
    stats(1)
  }

  def rightHess: Float = if (left) {
    stats(5)
  } else {
    stats(2)
  }

  def reverse: Split
}


private[gbm] case class SeqSplit(colId: Int,
                                 missingGo: Boolean,
                                 threshold: Int,
                                 left: Boolean,
                                 gain: Float,
                                 stats: Array[Float]) extends Split {
  require(stats.length == 6)

  override def goByBin[@spec(Byte, Short, Int) B](bin: B)
                                                 (implicit inb: Integral[B]): Boolean = {
    val b = inb.toInt(bin)
    if (b == 0) {
      missingGo
    } else {
      b <= threshold
    }
  }

  override def reverse: Split = {
    SeqSplit(colId, missingGo, threshold, !left, gain, stats)
  }
}


private[gbm] case class SetSplit(colId: Int,
                                 missingGo: Boolean,
                                 set: Array[Int],
                                 left: Boolean,
                                 gain: Float,
                                 stats: Array[Float]) extends Split {
  require(stats.length == 6)

  override def goByBin[@spec(Byte, Short, Int) B](bin: B)
                                                 (implicit inb: Integral[B]): Boolean = {
    val b = inb.toInt(bin)
    if (b == 0) {
      missingGo
    } else {
      ju.Arrays.binarySearch(set, b) >= 0
    }
  }

  override def reverse: Split = {
    SetSplit(colId, missingGo, set, !left, gain, stats)
  }
}


private[gbm] object Split extends Logging {

  /**
    * find the best split
    *
    * @param colId     feature index
    * @param hist      histogram
    * @param boostConf boosting config info
    * @param baseConf  tree config info
    * @return best split if any
    */
  def split[@spec(Float, Double) H](colId: Int,
                                    hist: Array[H],
                                    boostConf: BoostConfig,
                                    baseConf: BaseConfig)
                                   (implicit nuh: Numeric[H]): Option[Split] = {
    require(hist.length % 2 == 0)

    if (hist.length <= 2) {
      return None
    }

    val len = hist.length >> 1

    val gradSeq = Array.ofDim[Float](len)
    val hessSeq = Array.ofDim[Float](len)

    var gradAbsSum = 0.0
    var hessAbsSum = 0.0
    var nnz = 0

    var i = 0
    while (i < len) {
      val idx = i << 1
      gradSeq(i) = nuh.toFloat(hist(idx))
      hessSeq(i) = nuh.toFloat(hist(idx + 1))

      if (gradSeq(i) != 0 || hessSeq(i) != 0) {
        gradAbsSum += gradSeq(i).abs
        hessAbsSum += hessSeq(i).abs
        nnz += 1
      }

      i += 1
    }


    // hists of zero-bin are always computed by minus
    // for numerical stability, we ignore small values
    if ((gradSeq.head != 0 || hessSeq.head != 0) &&
      gradSeq.head.abs < gradAbsSum * 1e-4 &&
      hessSeq.head.abs < hessAbsSum * 1e-4) {

      gradSeq(0) = 0.0F
      hessSeq(0) = 0.0F
      nnz -= 1
    }

    if (nnz <= 1) {
      return None
    }

    val split = if (boostConf.isSeq(colId)) {
      splitSeq(colId, gradSeq, hessSeq, boostConf)
    } else if (nnz <= boostConf.getMaxBruteBins) {
      splitSetBrute(colId, gradSeq, hessSeq, boostConf)
    } else {
      splitSetHeuristic(colId, gradSeq, hessSeq, boostConf)
    }

    if (split.isEmpty ||
      !validate(split.get.stats: _*) ||
      !validate(split.get.gain)) {
      return None
    }

    val s = split.get
    // make sure that right leaves have smaller hess
    // to reduce the histogram computation in the next iteration
    if (s.leftHess < s.rightHess) {
      Some(s.reverse)
    } else {
      Some(s)
    }
  }


  /**
    * validate values for numerical stability
    *
    * @param values numbers
    * @return true is all numbers are ok
    */
  @varargs def validate(values: Float*): Boolean = {
    values.forall(v => !v.isNaN && !v.isInfinity)
  }


  /**
    * sequentially search the best split, with specially dealing with missing value
    *
    * @param colId     feature index
    * @param gradSeq   grad array
    * @param hessSeq   hess array
    * @param boostConf boosting config info
    * @return best split if any
    *         Note: input histogram arrays maybe updated in-place!
    */
  def splitSeq(colId: Int,
               gradSeq: Array[Float],
               hessSeq: Array[Float],
               boostConf: BoostConfig): Option[SeqSplit] = {
    // missing go left
    // find best split on indices of [i0, i1, i2, i3, i4]
    val search1 = seqSearch(gradSeq, hessSeq, boostConf)

    val search2 = if (gradSeq.head == 0 && hessSeq.head == 0) {
      // if hist of missing value is zero
      // do not need to place missing value to the right side
      None
    } else {

      // missing go right
      // find best split on indices of [i1, i2, i3, i4, i0]
      // Note: the histogram array here is updated in-place!
      val g0 = gradSeq.head
      System.arraycopy(gradSeq, 1, gradSeq, 0, gradSeq.length - 1)
      gradSeq(gradSeq.length - 1) = g0

      val h0 = hessSeq.head
      System.arraycopy(hessSeq, 1, hessSeq, 0, hessSeq.length - 1)
      hessSeq(hessSeq.length - 1) = h0

      seqSearch(gradSeq, hessSeq, boostConf)
    }

    (search1, search2) match {
      case (Some((cut1, gain1, stats1)), Some((cut2, gain2, stats2))) =>
        if (gain1 >= gain2) {
          Some(SeqSplit(colId, true, cut1, true, gain1, stats1))
        } else {
          // adjust the cut of split2
          // cut = 2, [i1, i2, i3 | i4, i0] -> cut = 3
          Some(SeqSplit(colId, false, cut2 + 1, true, gain2, stats2))
        }

      case (Some((cut, gain, stats)), None) =>
        Some(SeqSplit(colId, true, cut, true, gain, stats))

      case (None, Some((cut, gain, stats))) =>
        Some(SeqSplit(colId, false, cut + 1, true, gain, stats))

      case _ => None
    }
  }


  /**
    * Heuristically find the best set split
    *
    * @param colId     feature index
    * @param gradSeq   grad array
    * @param hessSeq   hess array
    * @param boostConf boosting config info
    * @return best split if any
    */
  def splitSetHeuristic(colId: Int,
                        gradSeq: Array[Float],
                        hessSeq: Array[Float],
                        boostConf: BoostConfig): Option[SetSplit] = {
    // sort the hist according to the relevance of gain
    // [g0, g1, g2, g3], [h0, h1, h2, h3] -> [g1, g3, g0, g2], [h1, h3, h0, h2]
    val epsion = boostConf.getRegLambda / gradSeq.length
    val (sortedGradSeq, sortedHessSeq, sortedIndices) =
      gradSeq.zip(hessSeq).zipWithIndex
        .map { case ((grad, hess), i) =>
          (grad, hess, i)
        }.sortBy { case (grad, hess, i) =>
        grad / (hess + epsion)
      }.unzip3

    val search = seqSearch(sortedGradSeq, sortedHessSeq, boostConf)

    if (search.isEmpty) {
      return None
    }

    val (cut, gain, stats) = search.get
    val indices1 = sortedIndices.take(cut + 1)

    val split = createSetSplit(colId, gradSeq, hessSeq, gain, indices1, stats)
    Some(split)
  }


  /**
    * Search the best set split by brute force
    *
    * @param colId     feature index
    * @param gradSeq   grad array
    * @param hessSeq   hess array
    * @param boostConf boosting config info
    * @return best split if any
    */
  def splitSetBrute(colId: Int,
                    gradSeq: Array[Float],
                    hessSeq: Array[Float],
                    boostConf: BoostConfig): Option[SetSplit] = {
    val gradSum = gradSeq.sum
    val hessSum = hessSeq.sum

    val (_, baseScore) = computeScore(gradSum, hessSum, boostConf)
    if (!validate(baseScore)) {
      return None
    }

    // ignore the indices with zero hist
    // [g0, g1, g2, g3, g4, g5, g6], [h0, h1, h2, h3, h4, h5, h6], [i0, i1, i2, i3, i4, i5, i6] ->
    // [g1, g2, g4, g6], [h1, h2, h4, h6], [i1, i2, i4, i6]
    val nzIndices = gradSeq.zip(hessSeq).zipWithIndex
      .filter { case ((grad, hess), _) =>
        grad != 0 || hess != 0
      }.map(_._2)

    val nnz = nzIndices.length
    require(nnz <= 64)


    // Here, we use a trick to efficiently traverse all candidates.
    // We not traverse the candidate sets, instead we use Gray codes of sets.
    // the initial `num` is zero, and its gray code is also zero,
    // which means no indices is selected, so the initial values of
    // `grad1` and `hess1` are zero.
    var bestGray = 0L
    var prevGray = 0L

    var bestScore = Float.MinValue

    var grad1 = 0.0F
    var grad2 = 0.0F

    var hess1 = 0.0F
    var hess2 = 0.0F

    val stats = Array.fill(6)(Float.NaN)

    // the first element in non-zero hist is always unselected in set1
    val k = 1L << (nnz - 1)
    var num = 1L

    while (num < k) {
      // gray code of `num`
      val gray = (num >> 1) ^ num

      // changed bit between current gray code and the previous one
      val i = jl.Long.numberOfTrailingZeros(gray ^ prevGray)

      val nzIndex = nzIndices(i)

      if (gray > prevGray) {
        grad1 += gradSeq(nzIndex)
        hess1 += hessSeq(nzIndex)
      } else {
        grad1 -= gradSeq(nzIndex)
        hess1 -= hessSeq(nzIndex)
      }

      grad2 = gradSum - grad1
      hess2 = hessSum - hess1

      if (hess1 >= boostConf.getMinNodeHess && hess2 >= boostConf.getMinNodeHess) {

        val (weight1, score1) = computeScore(grad1, hess1, boostConf)
        val (weight2, score2) = computeScore(grad2, hess2, boostConf)

        if (validate(weight1, score1, weight2, score2)) {
          val score = score1 + score2
          if (score > bestScore) {
            bestGray = gray
            bestScore = score

            stats(0) = weight1
            stats(1) = grad1
            stats(2) = hess1

            stats(3) = weight2
            stats(4) = grad2
            stats(5) = hess2
          }
        }
      }

      prevGray = gray
      num += 1
    }


    if (bestGray == 0L) {
      return None
    }

    if (!validate(stats: _*) || !validate(bestScore)) {
      return None
    }

    val gain = bestScore - baseScore
    if (gain < boostConf.getMinGain) {
      return None
    }


    // decode Gray code to indices set
    val builder = mutable.ArrayBuilder.make[Int]
    var i = 0
    while (i < nnz - 1) {
      val mask = 1L << i
      if ((mask & bestGray) != 0L) {
        builder += i
      }
      i += 1
    }
    val indices1 = builder.result()

    val split = createSetSplit(colId, gradSeq, hessSeq, gain, indices1, stats)
    Some(split)
  }


  /**
    * Sequentially search the best split
    *
    * @param gradSeq   grad array
    * @param hessSeq   hess array
    * @param boostConf boosting config info
    * @return best split containing (cut, gain, Array(weightL, gradL, hessL, weightR, gradR, hessR)), if any
    */
  def seqSearch(gradSeq: Array[Float],
                hessSeq: Array[Float],
                boostConf: BoostConfig): Option[(Int, Float, Array[Float])] = {
    val gradSum = gradSeq.sum
    val hessSum = hessSeq.sum

    val (_, baseScore) = computeScore(gradSum, hessSum, boostConf)
    if (!validate(baseScore)) {
      return None
    }

    var bestCut = -1
    var bestScore = Float.MinValue

    // weightLeft, weightRight, gradLeft, gradRight, hessLeft, hessRight
    val stats = Array.fill(6)(Float.NaN)

    var gradLeft = 0.0F
    var gradRight = 0.0F

    var hessLeft = 0.0F
    var hessRight = 0.0F

    var i = 0
    while (i < gradSeq.length - 1) {
      if (gradSeq(i) != 0 || hessSeq(i) != 0) {
        gradLeft += gradSeq(i)
        gradRight = gradSum - gradLeft
        hessLeft += hessSeq(i)
        hessRight = hessSum - hessLeft

        if (hessLeft >= boostConf.getMinNodeHess && hessRight >= boostConf.getMinNodeHess) {

          val (weightLeft, scoreLeft) = computeScore(gradLeft, hessLeft, boostConf)
          val (weightRight, scoreRight) = computeScore(gradRight, hessRight, boostConf)

          if (validate(weightLeft, scoreLeft, weightRight, scoreRight)) {
            val score = scoreLeft + scoreRight
            if (score > bestScore) {
              bestCut = i
              bestScore = score

              stats(0) = weightLeft
              stats(1) = gradLeft
              stats(2) = hessLeft

              stats(3) = weightRight
              stats(4) = gradRight
              stats(5) = hessRight
            }
          }

        }
      }

      i += 1
    }

    if (!validate(stats: _*) || !validate(bestScore)) {
      return None
    }

    val gain = bestScore - baseScore
    if (bestCut >= 0 && gain >= boostConf.getMinGain) {
      Some((bestCut, gain, stats))
    } else {
      None
    }
  }


  /**
    * Compute the weight and score, given the sum of hist.
    *
    * @param gradSum   sum of grad
    * @param hessSum   sum of hess
    * @param boostConf boosting config info containing the regulization parameters
    * @return weight and score
    */
  def computeScore(gradSum: Float,
                   hessSum: Float,
                   boostConf: BoostConfig): (Float, Float) = {
    if (boostConf.getRegAlpha == 0) {
      val weight = -gradSum / (hessSum + boostConf.getRegLambda)
      val loss = (hessSum + boostConf.getRegLambda) * weight * weight / 2 + gradSum * weight
      (weight.toFloat, -loss.toFloat)

    } else {
      val weight = if (gradSum + boostConf.getRegAlpha < 0) {
        -(boostConf.getRegAlpha + gradSum) / (hessSum + boostConf.getRegLambda)
      } else if (gradSum - boostConf.getRegAlpha > 0) {
        (boostConf.getRegAlpha - gradSum) / (hessSum + boostConf.getRegLambda)
      } else {
        0.0
      }
      val loss = (hessSum + boostConf.getRegLambda) * weight * weight / 2 + gradSum * weight +
        boostConf.getRegAlpha * weight.abs
      (weight.toFloat, -loss.toFloat)
    }
  }


  /**
    * Given a valid set split, choose the form of SetSplit to minimize the size of set.
    *
    * @param featureId feature index
    * @param gradSeq   grad array
    * @param hessSeq   hess array
    * @param gain      gain
    * @param indices1  indices of raw set1
    * @param stats     array containing (weight1, grad1, hess1, weight2, grad2, hess2)
    * @return a SetSplit
    */
  def createSetSplit(featureId: Int,
                     gradSeq: Array[Float],
                     hessSeq: Array[Float],
                     gain: Float,
                     indices1: Array[Int],
                     stats: Array[Float]): SetSplit = {
    require(indices1.max < gradSeq.length)
    require(stats.length == 6)

    // ignore zero hist
    val set1 = mutable.Set.empty[Int]
    var i = 0
    while (i < indices1.length) {
      if (gradSeq(i) != 0 || hessSeq(i) != 0) {
        set1.add(i)
      }
      i += 1
    }

    val set2 = mutable.Set.empty[Int]
    i = 0
    while (i < gradSeq.length) {
      if ((gradSeq(i) != 0 || hessSeq(i) != 0) && !set1.contains(i)) {
        set2.add(i)
      }
      i += 1
    }

    // remove index of missing value
    val missingInSet1 = set1.contains(0)
    val missingInSet2 = set2.contains(0)
    if (missingInSet1) {
      set1.remove(0)
    } else if (missingInSet2) {
      set2.remove(0)
    }

    // choose the smaller set
    if (set1.size <= set2.size) {
      SetSplit(featureId, missingInSet1, set1.toArray.sorted, true, gain, stats)
    } else {
      val newStats = stats.takeRight(3) ++ stats.take(3)
      SetSplit(featureId, missingInSet2, set2.toArray.sorted, true, gain, newStats)
    }
  }
}

