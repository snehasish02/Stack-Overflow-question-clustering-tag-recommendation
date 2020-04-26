# AWS imports
import boto3

# Google cloud imports
import bq_helper
from bq_helper import BigQueryHelper

# Spark imports
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as sf
from pyspark.ml.feature import Word2Vec,Word2VecModel

print("Setting up spark session...")
conf = pyspark.SparkConf().setAppName('Stackoverflow')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

# Python imports
import os
import json
import pandas as pd
import re
import nltk
import inflect
# from nltk.corpus import stopwords
import spacy
from bs4 import BeautifulSoup
import lxml
from textblob import TextBlob
from pyspark.ml.feature import StopWordsRemover

EN = spacy.load('en_core_web_sm')
print("Imports done")

# stopwords_english = stopwords.words('english')

stopwords_english = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def tokenize_text(text):
    "Apply tokenization using spacy to docstrings."
    tokens = EN.tokenizer(text)
    return [token.text.lower() for token in tokens if not token.is_space]

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords_english:
            new_words.append(word)
    return new_words

def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

def tokenize_code(text):
    "A very basic procedure for tokenizing code strings."
    return RegexpTokenizer(r'\w+').tokenize(text)

def preprocess_text(text):
    return (' '.join(normalize(tokenize_text(text))).split(' '))

def pre_process(x):                   #remove the code section
    #updating questions i.e removing all the html tags using parsers
    #print(x)
    soup = BeautifulSoup(x, 'lxml')
    if soup.code: soup.code.decompose()     # Remove the code section
    tag_p = soup.p
    tag_pre = soup.pre
    text = ''
    if tag_p: text = text + tag_p.get_text()
    if tag_pre: text = text + tag_pre.get_text()
    #print(tag_pre,tag_p)
    return text

def TextBlob_1(x):
    return TextBlob(x).polarity



# Get Data from GC
ACCESS_KEY_ID = "AKIAYNJWAMXNRPDPUUED"
ACCESS_KEY = "Q4u8pfhnsjEcBkNODJz0QrB07oc5fpRialXHL4bq"
s3 = boto3.resourcs3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY_ID , aws_secret_access_key=ACCESS_KEY, region_name='us-east-2')
# s3 = boto3.resource('s3')
my_bucket = s3.Bucket('cloud-stack-overflow')
my_bucket.download_file("gc.json", "gc.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gc.json"
bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
QUERY_1  = "SELECT ID,TITLE FROM `bigquery-public-data.stackoverflow.posts_questions` WHERE ID = 57804"
QUERY = "SELECT q.id, q.title, q.body, q.tags, a.body as answers, a.score FROM `bigquery-public-data.stackoverflow.posts_questions` AS q INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a ON q.id = a.parent_id WHERE q.tags LIKE '%python%' LIMIT 100"

print("Queries done")

df = bq_assistant.query_to_pandas(QUERY)
data_frame = spark.createDataFrame(df)

# 	data_frame.coalesce(1).write.format(
# 		"parquet").option(
# 		"header", "true").save(
# 		"s3n://cloud-stack-overflow/dataset/big_query.parquet")

print("Saving data done")

udf_myFunction = udf(pre_process, StringType())
# #removing all the html tags from body and answers and titles and forming new columns for the same
# data_frame_procc = data_frame.withColumn('processed_body',udf_myFunction(data_frame.body))
# data_frame_procc_1 = data_frame_procc.withColumn('processed_title',udf_myFunction(data_frame.title))
# data_frame_final = data_frame_procc_1.withColumn('processed_answers',udf_myFunction(data_frame.answers))
# data_frame_final = data_frame_final.withColumn('sentiment',udf(TextBlob_1, FloatType())(data_frame_final.answers))

# #concatenating title,body, answers into joined_data
# df_new_col = data_frame_final.withColumn(
# 	'joined_data',
#     sf.concat(sf.col('title'),
#     sf.lit(' '),
#     sf.col('processed_body'),
#     sf.lit(' '),
#     sf.col('processed_answers')))

# # now preprocessing the joined_data and tokenizing them and normalizing the score
# data_frame_tokenized = df_new_col.withColumn('joined_data',udf(preprocess_text, ArrayType(StringType()))(df_new_col.joined_data))
# min_score = data_frame_tokenized.select("score").rdd.min()[0]
# max_score = data_frame_tokenized.select("score").rdd.max()[0]
# mean_score = data_frame_tokenized.groupBy().avg("score").take(1)[0][0]

# #normalizing the score
# data_frame_toknorm = data_frame_tokenized.withColumn("score",(data_frame_tokenized.score-mean_score)/(max_score-min_score))

# #saving the processed dataframe into a parquet file
# # save_filepath = os.getcwd()+"/dataset/processed_data.parquet"
# # data_frame_toknorm.write.format('parquet').mode("overwrite").save(save_filepath)

# #create a word2vec model
# word2vec = Word2Vec(vectorSize=200, seed=42, inputCol="joined_data", outputCol="features")

# #fitting the model with the data present
# model_word2vec = word2vec.fit(data_frame_toknorm)
# print("word2vec model done")

# #saving the word2vec model
# # save_path = os.getcwd()+'/dataset/word2vecmodel'
# # model_word2vec.write().overwrite().save(save_path)

# titles_dataframe = data_frame_toknorm
# titles_dataframe_tokenized = titles_dataframe.withColumn('joined_data',udf(preprocess_text, ArrayType(StringType()))(titles_dataframe.processed_title))
# titles_df_results = model_word2vec.transform(titles_dataframe_tokenized)

# Saving the results
# save_filepath = os.getcwd()+"/dataset/title_vectors.parquet"
# titles_df_results.write.mode("overwrite").format('parquet').save(save_filepath)

print("Stopping session")
sc.stop()

