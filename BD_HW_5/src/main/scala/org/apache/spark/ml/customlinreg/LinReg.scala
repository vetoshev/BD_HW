package org.apache.spark.ml.customlinreg

import breeze.linalg.{functions, sum, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader,
  DefaultParamsWritable, DefaultParamsWriter, Identifiable,
  MLReadable, MLReader, MLWritable, MLWriter, MetadataUtils, SchemaUtils}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.{DoubleType, StructType}

trait LinRegParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  val prediction: Param[String] = new Param[String](
    this, "predictionCol", "prediction column name")
  def setPredictionCol(value: String): this.type = set(prediction, value)
  def getPredictionCol: String = $(prediction)
  setDefault(prediction, "prediction")
  val lr = new DoubleParam(this, "learningRate", "learning rate")
  def setLearningRate(value: Double) : this.type = set(lr, value)
  setDefault(lr -> 1e-1)
  val tolerance = new DoubleParam(this, "tol", "allowed error when converged")
  def setTol(value: Double) : this.type = set(tolerance, value)
  setDefault(tolerance -> 1e-5)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getOutputCol, DoubleType)
    schema
  }
}

class LinReg(override val uid: String) extends Estimator[LinRegModel] with LinRegParams
  with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("linReg"))
  override def copy(extra: ParamMap): Estimator[LinRegModel] = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def fit(dataset: Dataset[_]): LinRegModel = {
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()
    val data = dataset.withColumn("ones", lit(1))
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array($(inputCol), "ones", $(outputCol)))
      .setOutputCol("dataset_with_ones")

    val train: Dataset[Vector] = vectorAssembler
      .transform(data).select(col="dataset_with_ones").as[Vector]
    val i = MetadataUtils.getNumFeatures(dataset, $(inputCol))
    var thetas = BreezeDenseVector.rand[Double](i + 1)
    var thetas_p = new BreezeDenseVector(data=Array.fill[Double](i + 1)(Double.PositiveInfinity))

    while (functions.euclideanDistance(thetas.toDenseVector, thetas_p.toDenseVector) > $(tolerance)) {
      val summary = train.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v.asBreeze(0 until thetas.size).toDenseVector
          val y = v.asBreeze(-1)
          val theta = 2.0 * (X * (sum(X * thetas) - y))
          summarizer.add(mllib.linalg.Vectors.fromBreeze(theta))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)
      thetas_p = thetas.copy
      thetas = thetas - summary.mean.asBreeze * $(lr)
    }
    copyValues(new LinRegModel(thetas).setParent(this))
  }
}

object LinReg extends DefaultParamsReadable[LinReg]

class LinRegModel (override val uid: String,
                               val thetas: BreezeDenseVector[Double])
  extends Model[LinRegModel] with LinRegParams with MLWritable{

  def this(thetas: BreezeDenseVector[Double]) =
    this(Identifiable.randomUID("linRegModel"), thetas)
  override def copy(extra: ParamMap): LinRegModel = defaultCopy(extra)
  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x : Vector) => {
                sum(x.asBreeze.toDenseVector * thetas(0 to thetas.size - 2)) + thetas(-1)
        }
      )
    }
    dataset.withColumn($(prediction), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
      override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      sqlContext.createDataFrame(Seq(Tuple1(Vectors.fromBreeze(thetas))))
        .write.parquet(path + "/weights")
    }
  }

  def getWeights: BreezeDenseVector[Double] = {
    thetas
  }
}

object LinRegModel extends MLReadable[LinRegModel] {
  override def read: MLReader[LinRegModel] = new MLReader[LinRegModel] {
    override def load(path: String): LinRegModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/weights")
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()
      val weights : BreezeDenseVector[Double] = vectors.select(vectors("_1")
        .as[Vector]).first().asBreeze.toDenseVector
      val model = new LinRegModel(weights)
      metadata.getAndSetParams(model)
      model
    }
  }
}