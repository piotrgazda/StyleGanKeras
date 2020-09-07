from django.db import models

# Create your models here.
from django.db import models
from pygments.lexers import get_all_lexers
from pygments.styles import get_all_styles


class StyleMixModel(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    truncation = models.FloatField(default=1.0)
    orientation = models.BooleanField()
    rows = models.IntegerField()
    cols = models.IntegerField()
    image = models.ImageField(upload_to='stylemix/')

    class Meta:
        ordering = ['created']