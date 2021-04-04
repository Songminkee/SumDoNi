from django import forms


class UploadImageForm(forms.Form):
    img = forms.ImageField()


class UploadMultiImageForm(forms.Form):
    img = forms.ImageField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
