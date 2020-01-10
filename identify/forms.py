from django import forms
from .models import Xray


class boneform(forms.ModelForm):
    class Meta:
        model = Xray
        fields = ['patient', 'bone_image']