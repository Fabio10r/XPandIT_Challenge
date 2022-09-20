package xpandit

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.types.{DateType, DoubleType, IntegerType, LongType, StringType, StructType}
import org.apache.hadoop.shaded.org.eclipse.jetty.websocket.common.frames.DataFrame
import org.apache.spark.sql.functions.{avg, col, count, desc, regexp_replace, split, substring, substring_index, to_timestamp, udf, when}

import scala.Double.NaN
import scala.reflect.internal.util.TriState.True

/**
 * @author ${user.name}
 */
object App {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("SparkByExample")
      .getOrCreate()

    val schema_reviews = new StructType()
      .add("App", StringType, true)
      .add("Translated_Review", StringType, true)
      .add("Sentiment", StringType, true)
      .add("Sentiment_Polarity", DoubleType, true)
      .add("Sentiment_Subjectivity", DoubleType, true)

    val schema_apps = new StructType()
      .add("App", StringType, false) //ver
      .add("Category", StringType, false) //ver
      .add("Rating", DoubleType, false)
      .add("Reviews", LongType, true)
      .add("Size", StringType, true)
      .add("Installs", StringType, true)
      .add("Type", StringType, true)
      .add("Price", StringType, true)
      .add("Content Rating", StringType, true)
      .add("Genres", StringType, true)
      .add("Last Updated", StringType, true)
      .add("Current Ver", StringType, true)
      .add("Android Ver", StringType, true)

    val months = Map("01" -> "January", "02" -> "February", "03" -> "March", "04" -> "April", "05" -> "May", "06" -> "June",
      "07" -> "July", "08" -> "August", "10" -> "October", "11" -> "November", "12" -> "December")


    val df_reviews = spark.read.option("header", true)
      .schema(schema_reviews)
      .csv("resources/googleplaystore_user_reviews.csv")

    val df_apps = spark.read.option("header", true)
      .schema(schema_apps)
      .csv("resources/googleplaystore.csv")

    //part1
    val df_1 = df_reviews.na.fill(0, Array("Sentiment_Polarity"))
      .groupBy("App")
      .agg(avg("Sentiment_Polarity").as("Average_Sentiment_Polarity"))

    //part2
    val df_2 = df_apps
      .na.fill(0, Array("Price"))
      .filter(col("Rating").notEqual(NaN))
      .select("*")
      .where(col("Rating") >= 4.0)
      .orderBy(desc("Rating"))
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .options(Map("header" -> "true", "delimiter" -> "§"))
      .save("resources/part2")

    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val file_2 = fs.globStatus(new Path("resources/part2/part*"))(0).getPath().getName()
    fs.rename(new Path("resources/part2/" + file_2), new Path("resources/best_apps.csv"))
    fs.delete(new Path("resources/.best_apps.csv.crc"), true)
    fs.delete(new Path("resources/part2"), true)

    //part3

    def covertToTimestamp(col: String): String = {
      val result = ""
      val arr = col.toString().replace(",", "").split(" ")
      if (arr.length == 3) {
        val first = arr(0)
        var day = arr(1)
        if ((arr(1).toInt > 0) && (arr(1).toInt < 10)) {
          day = "0" + arr(1)
        }
        for ((k, v) <- months) if (v.equals(first)) {
          return day + "-" + k + "-" + arr(2) + " 00:00:00.000"
        }
      }
      return null
    }

    val covertToTimestampUDF = udf(covertToTimestamp _)

    val df_3 = df_apps
      .dropDuplicates()
      //Reviews column
      .na.fill(0, Array("Reviews"))
      //size column
      .withColumn("Size", when(col("Size").notEqual("Varies with device"),
        substring_index(col("Size"), "M", 1)).otherwise("0"))
      .withColumn("Size", col("Size").cast("double"))
      //Price column
      .withColumn("Price", when(col("Price").notEqual("0"),
        substring_index(col("Price"), "$", -1)).otherwise("0"))
      .withColumn("Price", col("Price").cast("double") * 0.9)
      //Genre column
      .withColumn("Genres", split(col("Genres"), "&"))
      //Last Updated Column
      .withColumn("Last Updated", to_timestamp(covertToTimestampUDF(col("Last Updated")), "dd-MM-yyyy HH:mm:ss.SSS"))
      .withColumnRenamed("Category", "Categories")


    //part4
    val df_1_prel = df_1.dropDuplicates().withColumnRenamed("App", "App2")

    val df_4 = df_3.as("df3")
      .join(df_1_prel.as("df1"), col("df3.App") === col("df1.App2"), "inner")
      .drop("App2")
      .withColumnRenamed("Content Rating", "Content_Rating")
      .withColumnRenamed("Last Updated", "Last_Updated")
      .withColumnRenamed("Current Ver", "Current_Version")
      .withColumnRenamed("Android Ver", "Minimum_Android_Version")
      .coalesce(1)
      .write
      .option("compression", "gzip")
      .parquet("resources/part4")

    val file_4 = fs.globStatus(new Path("resources/part4/part*"))(0).getPath().getName()
    fs.rename(new Path("resources/part4/" + file_4), new Path("resources/googleplaystore_cleaned.parquet"))
    fs.delete(new Path("resources/.googleplaystore_cleaned.parquet.crc"), true)
    fs.delete(new Path("resources/part4"), true)


    //part5
    val df_5 = df_3.as("df3")
      .join(df_1_prel.as("df1"), col("df3.App") === col("df1.App2"), "inner")
      .drop("App2")
      .groupBy(col("App"))
      .agg(count("App").as("Count"), avg("Rating").as("Average_Rating"),
        avg("Average_Sentiment_Polarity").as("Average_Sentiment_Polarity"))
      .write
      .option("compression", "gzip")
      .parquet("resources/part5")

    val file_5 = fs.globStatus(new Path("resources/part5/part*"))(0).getPath().getName()
    fs.rename(new Path("resources/part5/" + file_5), new Path("resources/googleplaystore_metrics.parquet"))
    fs.delete(new Path("resources/.googleplaystore_metrics.parquet.crc"), true)
    fs.delete(new Path("resources/part5"), true)
  }
}
