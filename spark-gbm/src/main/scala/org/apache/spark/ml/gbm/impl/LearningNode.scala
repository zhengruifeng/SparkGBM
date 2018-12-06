package org.apache.spark.ml.gbm.impl

import scala.{specialized => spec}


private[gbm] class LearningNode(val nodeId: Int,
                                var isLeaf: Boolean,
                                var prediction: Float,
                                var split: Option[Split],
                                var leftNode: Option[LearningNode],
                                var rightNode: Option[LearningNode]) extends Serializable {

  def index[@spec(Byte, Short, Int) B](bins: Array[B])
                                      (implicit inb: Integral[B]): Int = {
    if (isLeaf) {
      nodeId
    } else if (split.get.goLeft(bins)) {
      leftNode.get.index[B](bins)
    } else {
      rightNode.get.index[B](bins)
    }
  }

  def predict[@spec(Byte, Short, Int) B](bins: Array[B])
                                        (implicit inb: Integral[B]): Float = {
    if (isLeaf) {
      prediction
    } else if (split.get.goLeft(bins)) {
      leftNode.get.predict[B](bins)
    } else {
      rightNode.get.predict[B](bins)
    }
  }

  def nodeIterator: Iterator[LearningNode] = {
    Iterator.single(this) ++
      leftNode.map(_.nodeIterator).getOrElse(Iterator.empty) ++
      rightNode.map(_.nodeIterator).getOrElse(Iterator.empty)
  }

  def numDescendants: Int = {
    (leftNode ++ rightNode).map(_.numDescendants).sum
  }

  def subtreeDepth: Int = {
    if (isLeaf) {
      0
    } else {
      (leftNode ++ rightNode).map(_.subtreeDepth).fold(0)(math.max) + 1
    }
  }
}


private[gbm] object LearningNode {
  def create(nodeId: Int): LearningNode =
    new LearningNode(nodeId, true, 0.0F, None, None, None)
}

