from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_home, name='dashboard home'),
    path('ask', views.question, name='ask a question'),
]