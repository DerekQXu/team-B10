from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='site-home'),
    path('survey/', views.survey, name='site-survey'),
    path('about/', views.about, name='site-about'),
]
