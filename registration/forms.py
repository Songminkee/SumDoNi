from django import forms


class UploadImageForm(forms.Form):
    img = forms.ImageField()
