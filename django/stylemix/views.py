from django.shortcuts import render
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
import stylegan.networks as networks
import io
from django.conf import settings
from .forms import StyleMixForm
from .models import StyleMixModel
from django.db import models
from .serializers import StyleMixImageSerializer, StyleMixImagePostSerializer
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from django.forms.models import model_to_dict


class StyleMixRest(APIView):
    """
    List all images, or create a new image.
    """
    def get(self, request, format=None):
        images = StyleMixModel.objects.all()
        serializer = StyleMixImageSerializer(images, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = StyleMixImagePostSerializer(data=request.data)
        if serializer.is_valid():
            cols = serializer.data['cols']
            print(cols)
            rows = serializer.data['rows']
            truncation = serializer.data['truncation']
            orientation = serializer.data['orientation']
            image_name = '{}.png'.format(StyleMixModel.objects.count())
            img = networks.get_style_mix(cols, rows, truncation, orientation)
            img.save('stylegan/media/stylemix/' + image_name)
            mixed_image = StyleMixModel.objects.create(truncation=truncation,
                                                       orientation=orientation,
                                                       rows=rows,
                                                       cols=cols,
                                                       image='stylemix/' +
                                                       image_name)
            serializer_output = StyleMixImageSerializer(
                data=model_to_dict(mixed_image))
            if serializer_output.is_valid():
                return Response(serializer_output.data)
            else:
                return Response(serializer_output.errors)
        return Response(serializer.errors)


# Create your views here.
def index(request):
    image_name = 'image-placeholder.png'
    if request.method == 'POST':
        form = StyleMixForm(request.POST)
        if form.is_valid():
            cols = form.cleaned_data['cols']
            if cols is None:
                cols = 4
            rows = form.cleaned_data['rows']
            if rows is None:
                rows = 4
            truncation = form.cleaned_data['truncation']
            if truncation is None:
                truncation = 1.0
            orientation = form.cleaned_data['column_orientation']
            image_name = '{}.png'.format(StyleMixModel.objects.count())
            img = networks.get_style_mix(cols, rows, truncation, orientation)
            img.save('stylegan/media/stylemix/' + image_name)
            obj = StyleMixModel()
            obj.truncation = truncation
            obj.orientation = orientation
            obj.rows = rows
            obj.cols = cols
            obj.image = 'stylemix/' + image_name
            obj.save()

            return render(request,
                          "stylemix/index.html",
                          context={
                              "image_url": 'stylemix/' + image_name,
                              'form': form
                          })
    form = StyleMixForm()

    return render(request,
                  "stylemix/index.html",
                  context={
                      "image_url": 'image-placeholder.png',
                      'form': form
                  })


def post_mix(request):
    image_name = 'image-placeholder.png'
    print('from post print')
    if request.method == 'POST':
        form = StyleMixForm(request.POST)
        if form.is_valid():
            cols = form.cleaned_data['cols']
            if cols is None:
                cols = 4
            rows = form.cleaned_data['rows']
            if rows is None:
                rows = 4
            truncation = form.cleaned_data['truncation']
            if truncation is None:
                truncation = 1.0
            orientation = form.cleaned_data['column_orientation']
            image_name = 'mix.png'
            img = networks.get_style_mix(cols, rows, truncation, orientation)
            img.save('stylegan/media/' + image_name)

        return render(request,
                      "stylemix/index.html",
                      context={
                          "image_url": image_name,
                          'form': form
                      })