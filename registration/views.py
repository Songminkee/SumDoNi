from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views import View
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

from .forms import UploadImageForm


class RegistrationView(LoginRequiredMixin, TemplateView):
    login_url = '/accounts/login/'
    template_name = 'registration/registration.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = UploadImageForm()
        return context


class FaceRecognitionView(LoginRequiredMixin, TemplateView):
    login_url = '/accounts/login/'
    template_name = 'registration/face_selection.html'

    def post(self, request, *args, **kwargs):

        def get_img_for_template(img):
            from io import BytesIO
            import base64

            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode('ascii')

            return img_str

        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            from PIL import Image
            img = Image.open(form.cleaned_data['img'])
            img_str = get_img_for_template(img)

            context = self.get_context_data(**kwargs)
            context['img_str'] = img_str

            return self.render_to_response(context)
