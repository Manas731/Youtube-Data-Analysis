# Databricks notebook source
# MAGIC %sh
# MAGIC pip install --upgrade google-api-python-client
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
from googleapiclient.discovery import build
import pandas as pd

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder \
    .appName("YouTube Data Fetch") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020") \
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
    .getOrCreate()

# COMMAND ----------

# Your API key
DEVELOPER_KEY = "AIzaSyB52x8uxLfameKrof1MVW1qSlk77NCQV_A"
api_service_name = "youtube"
api_version = "v3"

# COMMAND ----------

# Build the YouTube API client
youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Request to get comments for a specific video
request = youtube.commentThreads().list(
    part="snippet",
    videoId="Ltnhz3YfJGY",
    maxResults=100
)
response = request.execute()

# COMMAND ----------

# Extract comments into a list
comments = []
for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['updatedAt'],
        comment['likeCount'],
        comment['textDisplay']
    ])


# COMMAND ----------

# Convert the comments list to a Pandas DataFrame
pdf = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])

# Convert Pandas DataFrame to PySpark DataFrame
df = spark.createDataFrame(pdf)

# COMMAND ----------

df = df.dropna()

# COMMAND ----------

# Example: Convert like_count to integer
from pyspark.sql.functions import col, regexp_replace
df = df.withColumn("like_count", col("like_count").cast("integer"))

# COMMAND ----------

df = df.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z0-9\s]", ""))

# COMMAND ----------

df.show(101)

# COMMAND ----------

# Define the output path
#output_path = "/databricks/driver/metastore_db/tmp/cleaned_output2.csv"

# Write the DataFrame to HDFS
#df.write.csv(hdfs_path, header=True)

# COMMAND ----------

pdf = df.toPandas()
output_path = "/databricks/driver/metastore_db/tmp/cleaned_output2.csv"
# Save the DataFrame to CSV
pdf.to_csv(output_path, index=False)


# COMMAND ----------

ls

# COMMAND ----------

cd metastore_db/

# COMMAND ----------

ls

# COMMAND ----------

cd tmp

# COMMAND ----------

ls

# COMMAND ----------

cat cleaned_output2.csv

# COMMAND ----------


