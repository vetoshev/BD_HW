package org.apache.spark.ml.customlinreg

import org.apache.spark.sql.SparkSession

trait WithSpark {
  lazy val spark = WithSpark._spark
  lazy val sqlc = WithSpark._sqlc
}

object WithSpark {
  lazy val _spark = SparkSession.builder
  .appName("Test App")
  .master("local[*]")
  .getOrCreate()

  lazy val _sqlc = _spark.sqlContext
}