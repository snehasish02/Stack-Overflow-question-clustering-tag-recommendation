import bq_helper
from bq_helper import BigQueryHelper
import os

# Spark imports
import findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Get Data from GC
google_creds = "s3://cloud-stack-overflow/gc.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds
bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
QUERY_1  = "SELECT ID,TITLE FROM `bigquery-public-data.stackoverflow.posts_questions` WHERE ID = 57804"
QUERY = "SELECT q.id, q.title, q.body, q.tags, a.body as answers, a.score FROM `bigquery-public-data.stackoverflow.posts_questions` AS q INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a ON q.id = a.parent_id WHERE q.tags LIKE '%python%' LIMIT 100"

df = bq_assistant.query_to_pandas(QUERY)

file_save_path = "s3://cloud-stack-overflow/test_output"
# file_save_path = os.getcwd()+"/dataset/big_query.parquet"
data_frame = spark.createDataFrame(df)
data_frame.write.format('parquet').mode("overwrite").save(file_save_path)

