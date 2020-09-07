from django.urls import path

from . import views

from rest_framework import routers

urlpatterns = [
    path('', views.index, name='index'),
    path('rest', views.TruncationRest.as_view(), name='truncation_rest'),
    #router.urls,
]