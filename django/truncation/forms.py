from django import forms


class TruncationForm(forms.Form):
    truncation_step = forms.FloatField(min_value=0.25,
                                  max_value=1.0,
                                  required=False,
                                  initial=0.25)

    rows = forms.IntegerField(min_value=1,
                              max_value=7,
                              required=False,
                              initial=1)
