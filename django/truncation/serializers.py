from rest_framework import serializers
from truncation.models import TruncationModel


class TruncationImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TruncationModel
        fields = ['created', 'truncation_step', 'rows', 'image']


class TruncationImagePostSerializer(serializers.ModelSerializer):
    class Meta:
        model = TruncationModel
        fields = ['truncation_step', 'rows']
