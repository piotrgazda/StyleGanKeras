from django import forms


class StyleMixForm(forms.Form):
    truncation = forms.FloatField(min_value=0.0,
                                  max_value=1.0,
                                  required=False,
                                  initial=1.0)
    column_orientation = forms.BooleanField(required=False)
    cols = forms.IntegerField(min_value=1,
                              max_value=7,
                              required=False,
                              initial=4)
    rows = forms.IntegerField(min_value=1,
                              max_value=7,
                              required=False,
                              initial=4)
