package org.apache.spark.ml.gbm

import java.{util => ju}

import scala.collection.mutable
import scala.{specialized => spec}


class TreeModel(val root: Node) extends Serializable {

  def isEmpty: Boolean = {
    root match {
      case leaf: LeafNode => leaf.weight == 0
      case _ => false
    }
  }

  lazy val depth: Int = root.subtreeDepth

  lazy val numLeaves: Int = root.numLeaves

  lazy val numNodes: Int = root.numDescendants

  private[gbm] def predict[@spec(Byte, Short, Int) B](bins: Int => B)
                                                     (implicit inb: Integral[B]): Double = root.predict[B](bins)

  private[gbm] def index[@spec(Byte, Short, Int) B](bins: Int => B)
                                                   (implicit inb: Integral[B]): Int = root.index[B](bins)

  def computeImportance: Map[Int, Double] = {
    val gains = mutable.OpenHashMap.empty[Int, Double]

    root.nodeIterator.foreach {
      case n: InternalNode =>
        val g = gains.getOrElse(n.colId, 0.0)
        gains.update(n.colId, g + n.gain)

      case _ =>
    }

    gains.toMap
  }
}


private[gbm] object TreeModel {

  def createModel(root: LearningNode): TreeModel = {
    createModel(root, (n: Int) => n)
  }

  def createModel(root: LearningNode,
                  mapping: Int => Int): TreeModel = {
    val leafIds = root.nodeIterator.filter(_.isLeaf)
      .map(_.nodeId).toArray.sorted
    val node = createNode(root, mapping, leafIds)
    new TreeModel(node)
  }

  def createNode(node: LearningNode,
                 mapping: Int => Int,
                 leafIds: Array[Int]): Node = {
    if (node.isLeaf) {
      require(node.leftNode.isEmpty &&
        node.rightNode.isEmpty &&
        node.split.isEmpty)

      val leafId = ju.Arrays.binarySearch(leafIds, node.nodeId)
      new LeafNode(node.prediction, leafId)

    } else {
      require(node.split.nonEmpty &&
        node.leftNode.nonEmpty && node.rightNode.nonEmpty)

      val reindex = mapping(node.split.get.colId)

      node.split.get match {
        case SeqSplit(_, missingGo, threshold, left, gain, _) =>
          new InternalNode(reindex, true, missingGo,
            Array(threshold), left, gain,
            createNode(node.leftNode.get, mapping, leafIds),
            createNode(node.rightNode.get, mapping, leafIds))


        case SetSplit(_, missingGo, set, left, gain, _) =>
          new InternalNode(reindex, false, missingGo,
            set, left, gain,
            createNode(node.leftNode.get, mapping, leafIds),
            createNode(node.rightNode.get, mapping, leafIds))
      }
    }
  }
}
