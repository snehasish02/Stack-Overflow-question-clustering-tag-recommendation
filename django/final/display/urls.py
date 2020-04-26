from django.urls import path
from . import views

urlpatterns = [
    path('test', views.index, name='index'),
    path('', views.home, name='home'),
    path('result', views.result, name='result'),
]
