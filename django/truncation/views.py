from django.shortcuts import render
from .forms import TruncationForm
from .models import TruncationModel

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
import stylegan.networks as networks
from django.forms.models import model_to_dict
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import TruncationImageSerializer, TruncationImagePostSerializer


class TruncationRest(APIView):
    """
    List all images, or create a new image.
    """
    def get(self, request, format=None):
        images = TruncationModel.objects.all()
        serializer = TruncationImageSerializer(images, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = TruncationImagePostSerializer(data=request.data)
        if serializer.is_valid():
            rows = serializer.data['rows']
            truncation_step = serializer.data['truncation_step']
            image_name = '{}.png'.format(TruncationModel.objects.count())
            img = networks.get_truncation(rows, truncation_step)
            img.save('stylegan/media/truncation/' + image_name)
            truncation_image = TruncationModel.objects.create(
                truncation_step=truncation_step,
                rows=rows,
                image='truncation/' + image_name)
            serializer_output = TruncationImageSerializer(
                data=model_to_dict(truncation_image))
            if serializer_output.is_valid():
                return Response(serializer_output.data)
            else:
                return Response(serializer_output.errors)
        return Response(serializer.errors)


# Create your views here.
def index(request):
    image_name = 'image-placeholder.png'
    if request.method == 'POST':
        form = TruncationForm(request.POST)
        if form.is_valid():
            rows = form.cleaned_data['rows']
            if rows is None:
                rows = 1
            truncation_step = form.cleaned_data['truncation_step']
            if truncation_step is None:
                truncation_step = 0.25

            image_name = '{}.png'.format(TruncationModel.objects.count())
            img = networks.get_truncation(rows, truncation_step)
            img.save('stylegan/media/truncation/' + image_name)
            obj = TruncationModel()
            obj.truncation_step = truncation_step
            obj.rows = rows
            obj.image = 'truncation/' + image_name
            obj.save()

            return render(request,
                          "truncation/index.html",
                          context={
                              "image_url": 'truncation/' + image_name,
                              'form': form
                          })

    form = TruncationForm()

    return render(request,
                  "truncation/index.html",
                  context={
                      "image_url": image_name,
                      'form': form
                  })


def post_truncation(request):
    return None