package com.arvin.ml

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import scala.math.random

object MainTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("SparkPi")
    //创建环境变量 设置本地模式 设置所要执行APP的名字
    val sc= new SparkContext(conf)

    val slices = if (args.length > 0)
      args(0).toInt else 2
    val n = math.min(10000000L * slices, Int.MaxValue).toInt
    //随机产生100000个数
    val count = sc.parallelize(1 until n, slices).map { i =>
      val x = random * 2 - 1
      val y = random * 2 - 1
      if (x * x + y * y < 1) 1 else 0
    }.reduce(_ + _)
    println("Pi is rough：" + 4.0 * count / n)
    sc.stop()

//    val spark = SparkSession.builder()
//      .master("local")
////      .enableHiveSupport()
//      .appName("Main Test")
//      .getOrCreate()
//
//    println(spark.conf)
  }
}
