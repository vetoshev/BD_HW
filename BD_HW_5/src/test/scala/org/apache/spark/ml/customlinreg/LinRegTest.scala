package org.apache.spark.ml.customlinreg

import com.google.common.io.Files
import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers.should


class LinRegTest extends AnyFlatSpec with should.Matchers with WithSpark{
  val data_size = 100
  val value_vector = DenseVector[Double](1.5, 0.3, -0.7)
  val d = 1e-3
  val trash = 1e-6
  val feat = "x"
  val predict = "y"
  protected val predicted = "prediction"

  lazy val X = DenseMatrix.rand[Double](data_size, value_vector.size)
  lazy val y = X * value_vector + DenseVector.rand[Double](data_size) * 0.001
  lazy val df = generateDF(X, y)

  private def generateDF(X: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {
    import sqlc.implicits._
    lazy val data: DenseMatrix[Double] = DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    lazy val _df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq.toDF("x1", "x2", "x3", "y")

    lazy val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3")).setOutputCol("x")
    lazy val df = assembler.transform(_df).select("x", "y")
    df
  }

  private def validate(model: LinRegModel): Unit = {
    val dfWithPrediction = model.transform(df)
    val evaluator = new RegressionEvaluator()
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setMetricName("mse")
    val mse = evaluator.evaluate(dfWithPrediction)
    mse should be <= trash
  }

  "Estimator" should s"have params: $value_vector" in {
    val lr = new LinReg()
      .setInputCol(feat)
      .setOutputCol(predict)
    val model = lr.fit(df)
    val params = model.getWeights
    params(0) should be(value_vector(0) +- d)
    params(1) should be(value_vector(1) +- d)
    params(2) should be(value_vector(2) +- d)
  }

  "Model" should "have MSE < threshold" in {
    val lr = new LinReg().setInputCol(feat)
      .setOutputCol(predict)
      .setPredictionCol(predicted)
    val model = lr.fit(df)
    validate(model)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(
      Array(
        new LinReg()
          .setInputCol(feat)
          .setOutputCol(predict)
          .setPredictionCol(predicted)
      )
    )
    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = Pipeline
      .load(tmpFolder.getAbsolutePath)
      .fit(df)
      .stages(0)
      .asInstanceOf[LinRegModel]
    validate(reRead)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(
      Array(
        new LinReg()
          .setInputCol(feat)
          .setOutputCol(predict)
          .setPredictionCol(predicted)
      )
    )
    val model = pipeline.fit(df)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)
    validate(reRead.stages(0).asInstanceOf[LinRegModel])
  }
}
