package org.apache.spark.ml.gbm.linalg

import java.{util => ju}

private[gbm] trait NumericExt[K] extends Serializable {

  def emptyArray: Array[K]

  def fromFloat(value: Float): K

  def fromDouble(value: Double): K

  def fromDouble(array: Array[Double]): Array[K]

  def toDouble(array: Array[K]): Array[Double]

  def toInt(array: Array[K]): Array[Int]

  def sqrt(value: K): K

  def search(array: Array[K], value: K): Int

  def size: Int
}

private object ByteNumericExt extends NumericExt[Byte] {

  override def emptyArray: Array[Byte] = Array.emptyByteArray

  override def fromFloat(value: Float): Byte = value.toByte

  override def fromDouble(value: Double): Byte = value.toByte

  override def fromDouble(array: Array[Double]): Array[Byte] = array.map(_.toByte)

  override def toDouble(array: Array[Byte]): Array[Double] = array.map(_.toDouble)

  override def toInt(array: Array[Byte]): Array[Int] = array.map(_.toInt)

  override def sqrt(value: Byte): Byte = math.sqrt(value).toByte

  override def search(array: Array[Byte], value: Byte): Int = ju.Arrays.binarySearch(array, value)

  override def size: Int = 1
}

private object ShortNumericExt extends NumericExt[Short] {

  override def emptyArray: Array[Short] = Array.emptyShortArray

  override def fromFloat(value: Float): Short = value.toShort

  override def fromDouble(value: Double): Short = value.toShort

  override def fromDouble(array: Array[Double]): Array[Short] = array.map(_.toShort)

  override def toDouble(array: Array[Short]): Array[Double] = array.map(_.toDouble)

  override def toInt(array: Array[Short]): Array[Int] = array.map(_.toInt)

  override def sqrt(value: Short): Short = math.sqrt(value).toByte

  override def search(array: Array[Short], value: Short): Int = ju.Arrays.binarySearch(array, value)

  override def size: Int = 2
}

private object IntNumericExt extends NumericExt[Int] {

  override def emptyArray: Array[Int] = Array.emptyIntArray

  override def fromFloat(value: Float): Int = value.toInt

  override def fromDouble(value: Double): Int = value.toInt

  override def fromDouble(array: Array[Double]): Array[Int] = array.map(_.toInt)

  override def toDouble(array: Array[Int]): Array[Double] = array.map(_.toDouble)

  override def toInt(array: Array[Int]): Array[Int] = array

  override def sqrt(value: Int): Int = math.sqrt(value).toInt

  override def search(array: Array[Int], value: Int): Int = ju.Arrays.binarySearch(array, value)

  override def size: Int = 4
}

private object LongNumericExt extends NumericExt[Long] {

  override def emptyArray: Array[Long] = Array.emptyLongArray

  override def fromFloat(value: Float): Long = value.toLong

  override def fromDouble(value: Double): Long = value.toLong

  override def fromDouble(array: Array[Double]): Array[Long] = array.map(_.toLong)

  override def toDouble(array: Array[Long]): Array[Double] = array.map(_.toDouble)

  override def toInt(array: Array[Long]): Array[Int] = array.map(_.toInt)

  override def sqrt(value: Long): Long = math.sqrt(value).toLong

  override def search(array: Array[Long], value: Long): Int = ju.Arrays.binarySearch(array, value)

  override def size: Int = 8
}

private object FloatNumericExt extends NumericExt[Float] {

  override def emptyArray: Array[Float] = Array.emptyFloatArray

  override def fromFloat(value: Float): Float = value

  override def fromDouble(value: Double): Float = value.toFloat

  override def fromDouble(array: Array[Double]): Array[Float] = array.map(_.toFloat)

  override def toDouble(array: Array[Float]): Array[Double] = array.map(_.toDouble)

  override def toInt(array: Array[Float]): Array[Int] = array.map(_.toInt)

  override def sqrt(value: Float): Float = math.sqrt(value).toFloat

  override def search(array: Array[Float], value: Float): Int = ju.Arrays.binarySearch(array, value)

  override def size: Int = 4
}

private object DoubleNumericExt extends NumericExt[Double] {

  override def emptyArray: Array[Double] = Array.emptyDoubleArray

  override def fromFloat(value: Float): Double = value.toDouble

  override def fromDouble(value: Double): Double = value

  override def fromDouble(array: Array[Double]): Array[Double] = array

  override def toDouble(array: Array[Double]): Array[Double] = array

  override def toInt(array: Array[Double]): Array[Int] = array.map(_.toInt)

  override def sqrt(value: Double): Double = math.sqrt(value)

  override def search(array: Array[Double], value: Double): Int = ju.Arrays.binarySearch(array, value)

  override def size: Int = 8
}


private[gbm] object NumericExt {

  implicit final val byteNumericExt: NumericExt[Byte] = ByteNumericExt

  implicit final val shortNumericExt: NumericExt[Short] = ShortNumericExt

  implicit final val intNumericExt: NumericExt[Int] = IntNumericExt

  implicit final val longNumericExt: NumericExt[Long] = LongNumericExt

  implicit final val floatNumericExt: NumericExt[Float] = FloatNumericExt

  implicit final val doubleNumericExt: NumericExt[Double] = DoubleNumericExt
}

