import org.apache.spark.sql._
import org.apache.spark.sql.functions._


class Config(){
  val input_data_path: String = "tripadvisor_hotel_reviews.csv"
  val text_col_name: String = "Review"
  val count_top_words: Int = 100
}

object task {
  def normalize_text(col_name: String)(df: DataFrame): DataFrame ={
    df.withColumn("clear_review", regexp_replace(lower(col(col_name)),
      "[^a-zA-Z0-9]+", " "))
  }

  def inverse_doc_freq(docs_count: Long, doc_freq: Long): Double ={
    math.log((docs_count.toDouble + 1)/(doc_freq.toDouble + 1))
  }


  def main(args: Array[String]): Unit = {
    val cfg = new Config()
    val spark = SparkSession.builder().master("local[*]").appName("made-demo").getOrCreate()
    import spark.implicits._

    val data = spark.read.option("header", "true").option("inferSchema", "true").csv(cfg.input_data_path)
      .transform(normalize_text(cfg.text_col_name)).select("clear_review").withColumnRenamed(
      "clear_review", "review").withColumn("id", monotonically_increasing_id())

    val words = data.select(col("id"), explode(split(col("review"), " ")).alias("word"))
      .where(length($"word") > 1)

    val tf = words.groupBy("id", "word").agg(count("word").alias("tf"))

    val docf = words.groupBy("word").agg(countDistinct("id").alias("df"))
      .orderBy(desc("df")).limit(cfg.count_top_words)

    val docs_count = data.count()
    val invdf = udf { df: Long => inverse_doc_freq(docs_count, df) }
    val idf = docf.withColumn("idf", invdf(col("df")))

    val tfidf = tf.join(idf, "word").withColumn("tfidf", col("tf") * col("idf"))
      .select(col("word"), col("id"), col("tfidf"))

    val res = tfidf.groupBy("id").pivot("word").sum("tfidf").orderBy("id").show()
  }
}