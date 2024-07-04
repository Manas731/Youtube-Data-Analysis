# Databricks notebook source
# DBTITLE 1,System Requirements
# MAGIC %sh
# MAGIC pip install --upgrade google-api-python-client
# MAGIC pip install textblob
# MAGIC pip install streamlit
# MAGIC pip install wordcloud

# COMMAND ----------

# DBTITLE 1,Data Fetch
import streamlit as st
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
# Your API key
DEVELOPER_KEY = "AIzaSyB52x8uxLfameKrof1MVW1qSlk77NCQV_A"
api_service_name = "youtube"
api_version = "v3"

# Build the YouTube API client
youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Request to get comments for a specific video
request = youtube.commentThreads().list(
    part="snippet",
    videoId="Ltnhz3YfJGY",
    maxResults=100
)
response = request.execute()

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

# Convert the comments list to a Pandas DataFrame
pdf = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
# Convert Pandas DataFrame to PySpark DataFrame
df = spark.createDataFrame(pdf)


# COMMAND ----------

# DBTITLE 1,Data Cleaning
df = df.dropna()
# Example: Convert like_count to integer
from pyspark.sql.functions import col, regexp_replace
df = df.withColumn("like_count", col("like_count").cast("integer"))

df = df.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z0-9\s]", ""))

df.show()

# COMMAND ----------

# DBTITLE 1,Saved to DBFS as csv
pdf = df.toPandas()
output_path = "/databricks/driver/metastore_db/tmp/cleaned_output.csv"
# Save the DataFrame to CSV
pdf.to_csv(output_path, index=False)


# COMMAND ----------

# Define the output path
#output_path = "/databricks/driver/metastore_db/tmp/cleaned_output2.csv"

# Write the DataFrame to HDFS
#df.write.csv(hdfs_path, header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Sentiment Analysis

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from textblob import TextBlob
import re

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()

# Define a UDF for TextBlob sentiment analysis
def get_sentiment(text):

    # Remove special characters
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Register the UDF
sentiment_udf = udf(get_sentiment, FloatType())

# Add a new column with sentiment scores
df = df.withColumn("sentiment", sentiment_udf(df.text))

# Show the DataFrame with the sentiment column
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Loading Data to StreamLit

# COMMAND ----------

# Display the DataFrame in Streamlit
padf = df.toPandas()
st.title("YouTube Comments Sentiment Analysis")
st.write("This dashboard shows the sentiment analysis of YouTube comments.")
st.dataframe(padf)
padf.head()

# COMMAND ----------

# DBTITLE 1,Visualize Data
import matplotlib.pyplot as plt
#bar chart
# Sentiment distribution
sentiment_counts = padf['sentiment'].value_counts()
st.bar_chart(sentiment_counts)







# COMMAND ----------

#Word Count
#from wordcloud import WordCloud

# Combine all text
#text = ' '.join(padf['text'].tolist())

# Generate word cloud
#wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display word cloud
#plt.figure(figsize=(10, 5))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')
#st.pyplot(plt)

# COMMAND ----------

padf['published_at'] = pd.to_datetime(padf['published_at'])
padf.set_index('published_at', inplace=True)

st.line_chart(padf['like_count'])


# COMMAND ----------

# MAGIC %md
# MAGIC filtering and interaction

# COMMAND ----------

#Filter by Author
authors = padf['author'].unique().tolist()
selected_author = st.selectbox('Select an author', authors)

filtered_df = padf[padf['author'] == selected_author]
st.dataframe(filtered_df)

#Filter by Date Range
start_date = st.date_input('Start date', padf.index.min().date())
end_date = st.date_input('End date', padf.index.max().date())

filtered_df = padf[(padf.index.date >= start_date) & (padf.index.date <= end_date)]
st.dataframe(filtered_df)



# COMMAND ----------

# MAGIC %md
# MAGIC Sentiment Analysis Visualization

# COMMAND ----------

import seaborn as sns

# Histogram of Sentiment Scores
st.subheader("Sentiment Score Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(padf['sentiment'], bins=20, kde=True)
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
st.pyplot(plt)

# Scatter Plot of Likes vs Sentiment
st.subheader("Likes vs Sentiment")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=padf, x='like_count', y='sentiment')
plt.title('Likes vs Sentiment')
plt.xlabel('Number of Likes')
plt.ylabel('Sentiment Score')
st.pyplot(plt)

# COMMAND ----------

# MAGIC %md
# MAGIC Applying Naive Bayes

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

# Convert sentiment to binary classification (0 for negative, 1 for non-negative)
df = df.withColumn("label", (col("sentiment") >= 0).cast("integer"))

# Tokenize the text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df = tokenizer.transform(df)

# Vectorize the words
vectorizer = CountVectorizer(inputCol="words", outputCol="features")
vectorizer_model = vectorizer.fit(df)
df = vectorizer_model.transform(df)

# Select features and label for training
df = df.select("features", "label")

# Split the data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=12345)

# Train the Naive Bayes model
nb = NaiveBayes(modelType="multinomial")
model = nb.fit(train_df)

# Make predictions
predictions = model.transform(test_df)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Test set accuracy = " + str(accuracy))

# Show some predictions
predictions.select("features", "label", "prediction").show()




# COMMAND ----------

model.save("/databricks/driver/metastore_db/tmp/naive-bayes-model")


# COMMAND ----------

cd /databricks/driver

# COMMAND ----------

# MAGIC %sh
# MAGIC ls

# COMMAND ----------

model_path = "/mymodel"
model.write().overwrite().save(model_path)

# COMMAND ----------

import os 
print(os.getcwd())
os.listdir()

# COMMAND ----------

model.save("dbfs:/mymodel2")

# COMMAND ----------



# COMMAND ----------

pdf = df.toPandas()
output_path = "/databricks/driver/metastore_db/tmp/ML_output.csv"
# Save the DataFrame to CSV
pdf.to_csv(output_path, index=False)


# COMMAND ----------

# MAGIC %md
# MAGIC opening streamlit using ngrok

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
# MAGIC tar -xvzf ngrok-v3-stable-linux-amd64.tgz
# MAGIC ./ngrok authtoken <YOUR_NGROK_AUTH_TOKEN>
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC ./ngrok authtoken 2iYQwea8a7yb0to9oapxijsGLzD_3PAtKPpbo4XneUi2FwMYH
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC ./ngrok version
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC streamlit run /databricks/python_shell/scripts/db_ipykernel_launcher.py &
# MAGIC ./ngrok http 443
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC streamlit run /databricks/python_shell/scripts/db_ipykernel_launcher.py

# COMMAND ----------

# MAGIC %sh
# MAGIC streamlit run /databricks/python_shell/scripts/db_ipykernel_launcher.py --server.port 443
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC ifconfig

# COMMAND ----------

# MAGIC %sh ipconfig

# COMMAND ----------


