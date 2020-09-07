from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models
from pygments.lexers import get_all_lexers
from pygments.styles import get_all_styles


class TruncationModel(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    truncation_step = models.FloatField(default=0.25)
    rows = models.IntegerField()
    image = models.ImageField(upload_to='truncation/')

    class Meta:
        ordering = ['created']