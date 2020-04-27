from django.http import JsonResponse
from django.shortcuts import render
from django.core import serializers
from pyspark.ml.feature import Word2Vec, Word2VecModel

def dashboard_home(request):
    return render(request, 'index.html', {})

def question(request):
	return render(request, 'question.html', {})
