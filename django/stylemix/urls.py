from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('rest', views.StyleMixRest.as_view(), name='stylemix_rest')
]