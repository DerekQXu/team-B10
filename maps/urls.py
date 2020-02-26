from django.urls import path
from . import views

urlpatterns = [
    path('maps/', views.maps, name='site-maps'),
    path('about/', views.about, name='site-about'),
]
