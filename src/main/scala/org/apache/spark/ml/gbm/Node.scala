package org.apache.spark.ml.gbm

private[gbm] class LearningNode(val nodeId: Long,
                                var isLeaf: Boolean,
                                var prediction: Double,
                                var split: Option[Split],
                                var leftNode: Option[LearningNode],
                                var rightNode: Option[LearningNode]) extends Serializable {

  def index(bins: Array[Int]): Long = {
    if (isLeaf) {
      nodeId
    } else if (split.get.goLeft(bins)) {
      leftNode.get.index(bins)
    } else {
      rightNode.get.index(bins)
    }
  }

  def predict(bins: Array[Int]): Double = {
    if (isLeaf) {
      prediction
    } else if (split.get.goLeft(bins)) {
      leftNode.get.predict(bins)
    } else {
      rightNode.get.predict(bins)
    }
  }

  def nodeIterator: Iterator[LearningNode] = {
    Iterator.single(this) ++
      leftNode.map(_.nodeIterator).getOrElse(Iterator.empty) ++
      rightNode.map(_.nodeIterator).getOrElse(Iterator.empty)
  }

  def numDescendants: Long = {
    (leftNode ++ rightNode).map(_.numDescendants).sum
  }

  def subtreeDepth: Long = {
    if (isLeaf) {
      0L
    } else {
      (leftNode ++ rightNode).map(_.subtreeDepth).fold(0L)(math.max) + 1
    }
  }
}

private[gbm] object LearningNode {
  def create(nodeId: Long): LearningNode =
    new LearningNode(nodeId, true, Double.NaN, None, None, None)
}


trait Node extends Serializable {
  def index(bins: Array[Int]): Long

  def predict(bins: Array[Int]): Double

  def subtreeDepth: Int

  def nodeIterator: Iterator[Node]

  def numDescendants: Long = {
    var cnt = 0L
    nodeIterator.foreach(_ => cnt += 1)
    cnt
  }

  def numLeaves: Long = {
    var cnt = 0L
    nodeIterator.foreach {
      case _: LeafNode =>
        cnt += 1
      case _ =>
    }
    cnt
  }
}


class InternalNode(val featureId: Int,
                   val isSeq: Boolean,
                   val missingGoLeft: Boolean,
                   val data: Array[Int],
                   val gain: Double,
                   val leftNode: Node,
                   val rightNode: Node) extends Node {
  if (isSeq) {
    require(data.length == 1)
  }

  def goLeft(bins: Array[Int]): Boolean = {
    val bin = bins(featureId)
    if (bin == 0) {
      missingGoLeft
    } else if (isSeq) {
      bin <= data.head
    } else {
      java.util.Arrays.binarySearch(data, bin) >= 0
    }
  }

  def index(bins: Array[Int]): Long = {
    if (goLeft(bins)) {
      leftNode.index(bins)
    } else {
      rightNode.index(bins)
    }
  }

  def predict(bins: Array[Int]): Double = {
    if (goLeft(bins)) {
      leftNode.predict(bins)
    } else {
      rightNode.predict(bins)
    }
  }

  def subtreeDepth: Int = {
    math.max(leftNode.subtreeDepth, rightNode.subtreeDepth) + 1
  }

  def nodeIterator: Iterator[Node] = {
    Iterator(this) ++
      leftNode.nodeIterator ++
      rightNode.nodeIterator
  }
}

class LeafNode(val weight: Double,
               val leafId: Long) extends Node {

  def index(bins: Array[Int]): Long = leafId

  def predict(bins: Array[Int]): Double = weight

  def subtreeDepth: Int = 0

  def nodeIterator: Iterator[Node] = Iterator(this)
}


private[gbm] case class NodeData(id: Int,
                                 featureId: Int,
                                 isSeq: Boolean,
                                 missingGoLeft: Boolean,
                                 data: Array[Int],
                                 gain: Double,
                                 leftNode: Int,
                                 rightNode: Int,
                                 weight: Double,
                                 leafId: Long)


private[gbm] object NodeData {

  def createData(node: Node, id: Int): (Seq[NodeData], Int) = {
    node match {
      case n: InternalNode =>
        val (leftNodeData, leftIdx) = createData(n.leftNode, id + 1)
        val (rightNodeData, rightIdx) = createData(n.rightNode, leftIdx + 1)
        val thisNodeData = NodeData(id, n.featureId, n.isSeq, n.missingGoLeft, n.data, n.gain,
          leftNodeData.head.id, rightNodeData.head.id, Double.NaN, -1)
        (thisNodeData +: (leftNodeData ++ rightNodeData), rightIdx)

      case n: LeafNode =>
        (Seq(NodeData(id, -1, false, false, Array.emptyIntArray,
          Double.NaN, -1, -1, n.weight, n.leafId)), id)
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
    nodes.reverseIterator.foreach { n: NodeData =>
      val node = if (n.leftNode != -1) {
        val leftChild = finalNodes(n.leftNode)
        val rightChild = finalNodes(n.rightNode)
        new InternalNode(n.featureId, n.isSeq, n.missingGoLeft, n.data, n.gain, leftChild, rightChild)
      } else {
        new LeafNode(n.weight, n.leafId)
      }
      finalNodes(n.id) = node
    }
    // Return the root node
    finalNodes.head
  }
}