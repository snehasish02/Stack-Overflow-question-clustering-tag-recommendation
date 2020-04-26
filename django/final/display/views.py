from django.shortcuts import render
from django.http import HttpResponse
from pyspark.ml.feature import Word2VecModel, Word2Vec
import os
import pandas as pd
import sys
import findspark
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
from django import forms
from .forms import q_form
import matplotlib.pyplot as plt


def index(request):
    return HttpResponse('This is a test webpage')


def home(request):
    # 1.
    # sns.catplot(kind='bar', x="tag", y="count",
    #             data=tag.head(20), height=10, aspect=1.5)
    # plt.title('Frequency of top 20 tags')
    # plt.xticks(rotation=45)
    # plt.xlabel('Tags')
    # plt.ylabel('Number of occurences')
    # plt.savefig("tag_bar.png")
    # bar_path = os.getcwd() + '/tag_bar.png'
    # bar_graph_context = {'bar_graph': bar_path}

    # # 2.
    # tag_dict = {}
    # for t, c in tag.head(wordCount).values:
    #     tag_dict[t] = c

    # wordcloud = WordCloud(width=1920, height=1080,
    #                       background_color='black',).generate_from_frequencies(tag_dict)
    # fig = plt.figure(figsize=(40, 30))
    # plt.title(f'Word Cloud of top {wordCount} Tags')
    # plt.imshow(wordcloud)
    # plt.axis('off')
    # plt.tight_layout(pad=0)
    # fig.savefig("tag_wordcloud.png")
    # plt.show()
    # wordcloud_path = os.getcwd() + '/tag_wordcloud.png'
    # wordcloud_context = {'word_cloud': wordcloud_path}

    # # 3.
    # fig = plt.figure(figsize=(15, 10))
    # plt.plot(tag["count"].values[0:wordCount], c='b')
    # plt.scatter(x=list(range(0, wordCount, 5)),
    #             y=tag["count"].values[0:wordCount:5], c='orange', label="quantiles with 0.05 intervals")

    # plt.scatter(x=list(range(0, wordCount, 10)),
    #             y=tag["count"].values[0:wordCount:10], c='m', label="quantiles with 0.1 intervals")

    # for x, y in zip(list(range(0, wordCount, 10)), tag["count"].values[0:wordCount:10]):
    #     plt.annotate(s="({} , {})".format(x, y), xy=(
    #         x, y), xytext=(x - 0.05, y + 20))

    # plt.title(f'Distribution of number of times tag appeared questions')
    # plt.grid()
    # plt.xlabel(f"First {wordCount} Popular tags")
    # plt.ylabel("Number of times tag appeared")
    # plt.legend()
    # fig.savefig("tag_distribution.png")
    # plt.show()
    # dist_path = os.getcwd() + '/tag_distribution.png'
    # dist_context = {'tag_distribution': dist_path}

    # # 4.
    # xlabel = [i + wordCount for i in range(0, len(total_tags), interval)]
    # fig = plt.figure(figsize=(15, 10))
    # plt.plot(xlabel, percentage)
    # plt.locator_params(axis='x', nbins=10)
    # plt.xlabel("Most Popular Tag Count")
    # plt.ylabel("Percentage Questions Covered by the tags")
    # plt.savefig("question_tag_percentage.png")
    # plt.show()
    # question_tag_path = os.getcwd() + '/question_tag_percentage.png'
    # question_tag_context = {'question_tag_percentage': question_tag_path}

    # graphIMG.save(buffer, "PNG")
    # return render(request, 'homepage.html', bar_graph_context, wordcloud_context, dist_context, question_tag_context)
    return render(request, 'homepage.html')


def result(request):
    print('abc')
    if request.method == 'POST':
        print('abc1')
        # form = q_form(request.POST)
        # question = request.POST.get("question", None)
        # question = request.POST['question']
        # print(form)
        # if form.is_valid():
        print('abc2')
        print(request.POST)
        question = request.POST.get("question", None)
        print(question)
        # question = form.cleaned_data

        # question_dataframe = spark.createDataFrame(
        #     [(input_question, )], ["question"])
        # question_tokenized_df = question_dataframe.withColumn('joined_data', udf(
        #     preprocess_text, ArrayType(StringType()))(question_dataframe.question))

        # saveword2vec_path = os.getcwd() + '/word2vecmodel'
        # model_word2vec = Word2VecModel.load(saveword2vec_path)

        # question_with_vector_df = model_word2vec.transform(
        #     question_tokenized_df)
        # question_dense_vec = question_with_vector_df.first()["features"]
        # df_cos_sim = title_vectors_df.withColumn("similarity_score", udf(cos_sim, FloatType())(
        #     col("sentiment"), col("score"), col("features"), array([lit(v) for v in question_dense_vec])))
        # result = df_cos_sim.orderBy(
        #     'similarity_score', ascending=False).take(2)

        # prediction = fake_model.fake_predict(user_input_age)
        # return render(request, 'result.html', {'result': result})

    else:
        form = q_form()
    return render(request, 'homepage.html', {})


def cos_sim(d, c, a, b):
    if np.dot(a, b) == 0:
        return 0
    return 0.4 * d + 0.1 * c + float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
