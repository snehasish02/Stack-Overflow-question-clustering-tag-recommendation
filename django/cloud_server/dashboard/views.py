from django.http import JsonResponse
from django.shortcuts import render
from django.core import serializers

def dashboard_home(request):
    return render(request, 'index.html', {})

def question(request):
	return render(request, 'question.html', {})
