from rest_framework import serializers
from stylemix.models import StyleMixModel


class StyleMixImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = StyleMixModel
        fields = [
            'created', 'truncation', 'orientation', 'rows', 'cols', 'image'
        ]


class StyleMixImagePostSerializer(serializers.ModelSerializer):
    class Meta:
        model = StyleMixModel
        fields = ['truncation', 'orientation', 'rows', 'cols']
