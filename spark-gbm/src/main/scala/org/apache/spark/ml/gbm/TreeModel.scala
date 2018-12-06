package org.apache.spark.ml.gbm

import java.{util => ju}

import scala.collection.mutable
import scala.{specialized => spec}

import org.apache.spark.ml.gbm.impl._


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

  def computeImportances(importanceType: String): Map[Int, Double] = {
    val iter = root.internalNodeIterator

    importanceType.toLowerCase(ju.Locale.ROOT) match {

      case GBM.AvgGain =>
        val gains = mutable.OpenHashMap.empty[Int, (Double, Int)]
        while (iter.hasNext) {
          val n = iter.next()
          val (g, c) = gains.getOrElse(n.colId, (0.0, 0))
          gains.update(n.colId, (g + n.gain, c + 1))
        }

        gains.map { case (colId, (gain, count)) =>
          (colId, gain / count)
        }.toMap


      case GBM.SumGain =>
        val gains = mutable.OpenHashMap.empty[Int, Double]
        while (iter.hasNext) {
          val n = iter.next()
          val g = gains.getOrElse(n.colId, 0.0)
          gains.update(n.colId, g + n.gain)
        }

        gains.toMap


      case GBM.NumSplits =>
        val counts = mutable.OpenHashMap.empty[Int, Int]
        while (iter.hasNext) {
          val n = iter.next()
          val c = counts.getOrElse(n.colId, 0)
          counts.update(n.colId, c + 1)
        }

        counts.mapValues(_.toDouble).toMap
    }
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





trait Node extends Serializable {

  private[gbm] def index[@spec(Byte, Short, Int) B](bins: Int => B)
                                                   (implicit inb: Integral[B]): Int

  private[gbm] def predict[@spec(Byte, Short, Int) B](bins: Int => B)
                                                     (implicit inb: Integral[B]): Float

  def subtreeDepth: Int

  def nodeIterator: Iterator[Node]

  def internalNodeIterator: Iterator[InternalNode]

  def numDescendants: Int =
    nodeIterator.size

  def numLeaves: Int =
    nodeIterator.count(_.isInstanceOf[LeafNode])
}


class InternalNode(val colId: Int,
                   val isSeq: Boolean,
                   val missingGo: Boolean,
                   val data: Array[Int],
                   val left: Boolean,
                   val gain: Float,
                   val leftNode: Node,
                   val rightNode: Node) extends Node {
  if (isSeq) {
    require(data.length == 1)
  }

  private def goByBin[@spec(Byte, Short, Int) B](bin: B)
                                                (implicit inb: Integral[B]): Boolean = {
    val b = inb.toInt(bin)
    if (b == 0) {
      missingGo
    } else if (isSeq) {
      b <= data.head
    } else {
      java.util.Arrays.binarySearch(data, b) >= 0
    }
  }

  private def goLeftByBin[@spec(Byte, Short, Int) B](bin: B)
                                                    (implicit inb: Integral[B]): Boolean = {
    if (left) {
      goByBin[B](bin)
    } else {
      !goByBin[B](bin)
    }
  }

  private[gbm] def goLeft[@spec(Byte, Short, Int) B](bins: Int => B)
                                                    (implicit inb: Integral[B]): Boolean = {
    goLeftByBin[B](bins(colId))
  }

  private[gbm] override def index[@spec(Byte, Short, Int) B](bins: Int => B)
                                                            (implicit inb: Integral[B]): Int = {
    if (goLeft(bins)) {
      leftNode.index[B](bins)
    } else {
      rightNode.index[B](bins)
    }
  }

  private[gbm] override def predict[@spec(Byte, Short, Int) B](bins: Int => B)
                                                              (implicit inb: Integral[B]): Float = {
    if (goLeft(bins)) {
      leftNode.predict[B](bins)
    } else {
      rightNode.predict[B](bins)
    }
  }

  override def subtreeDepth: Int = {
    math.max(leftNode.subtreeDepth, rightNode.subtreeDepth) + 1
  }

  override def nodeIterator: Iterator[Node] = {
    Iterator(this) ++
      leftNode.nodeIterator ++
      rightNode.nodeIterator
  }

  override def internalNodeIterator: Iterator[InternalNode] = {
    Iterator(this) ++
      leftNode.internalNodeIterator ++
      rightNode.internalNodeIterator
  }
}

class LeafNode(val weight: Float,
               val leafId: Int) extends Node {

  override def subtreeDepth: Int = 0

  override def nodeIterator: Iterator[Node] = Iterator(this)

  override def internalNodeIterator: Iterator[InternalNode] = Iterator.empty

  private[gbm] override def index[@spec(Byte, Short, Int) B](bins: Int => B)
                                                            (implicit inb: Integral[B]): Int = leafId

  private[gbm] override def predict[@spec(Byte, Short, Int) B](bins: Int => B)
                                                              (implicit inb: Integral[B]): Float = weight
}


private[gbm] case class NodeData(id: Int,
                                 colId: Int,
                                 isSeq: Boolean,
                                 missingGo: Boolean,
                                 data: Array[Int],
                                 left: Boolean,
                                 gain: Float,
                                 leftNode: Int,
                                 rightNode: Int,
                                 weight: Float,
                                 leafId: Int)


private[gbm] object NodeData {

  def createData(node: Node, id: Int): (Seq[NodeData], Int) = {
    node match {
      case n: InternalNode =>
        val (leftNodeData, leftIdx) = createData(n.leftNode, id + 1)
        val (rightNodeData, rightIdx) = createData(n.rightNode, leftIdx + 1)
        val thisNodeData = NodeData(id, n.colId, n.isSeq, n.missingGo,
          n.data, n.left, n.gain, leftNodeData.head.id, rightNodeData.head.id,
          Float.NaN, -1)
        (thisNodeData +: (leftNodeData ++ rightNodeData), rightIdx)

      case n: LeafNode =>
        (Seq(NodeData(id, -1, false, false, Array.emptyIntArray, false,
          Float.NaN, -1, -1, n.weight, n.leafId)), id)
    }
  }

  /**
    * Given all data for all nodes in a tree, rebuild the tree.
    *
    * @param data Unsorted node data
    * @return Root node of reconstructed tree
    */
  def createNode(data: Array[NodeData]): Node = {
    // Load all nodes, sorted by ID.
    val nodes = data.sortBy(_.id)
    // Sanity checks; could remove
    assert(nodes.head.id == 0, s"Decision Tree load failed.  Expected smallest node ID to be 0," +
      s" but found ${nodes.head.id}")
    assert(nodes.last.id == nodes.length - 1, s"Decision Tree load failed.  Expected largest" +
      s" node ID to be ${nodes.length - 1}, but found ${nodes.last.id}")
    // We fill `finalNodes` in reverse order.  Since node IDs are assigned via a pre-order
    // traversal, this guarantees that child nodes will be built before parent nodes.
    val finalNodes = new Array[Node](nodes.length)
    nodes.reverseIterator.foreach { n =>
      val node = if (n.leftNode != -1) {
        val leftChild = finalNodes(n.leftNode)
        val rightChild = finalNodes(n.rightNode)
        new InternalNode(n.colId, n.isSeq, n.missingGo, n.data, n.left, n.gain, leftChild, rightChild)
      } else {
        new LeafNode(n.weight, n.leafId)
      }
      finalNodes(n.id) = node
    }
    // Return the root node
    finalNodes.head
  }
}

