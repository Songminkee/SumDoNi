from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views import View
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

from .forms import UploadMultiImageForm
from .models import Borrower, UserBorrower, User
from .face_recognition import face_recognition, save_features

from io import BytesIO
from PIL import Image
import base64
import cv2
import numpy as np


class RegistrationView(LoginRequiredMixin, TemplateView):
    login_url = '/accounts/login/'
    template_name = 'registration/registration.html'

    def get_context_data(self, **kwargs):
        video_path = '1.mp4'

        context = super().get_context_data(**kwargs)
        context['form'] = UploadMultiImageForm
        context['video_path'] = video_path

        cap = cv2.VideoCapture('./static/' + video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        context['fps'] = fps
        return context


class FaceRecognitionView(LoginRequiredMixin, TemplateView):
    login_url = '/accounts/login/'
    template_name = 'registration/face_selection.html'

    def post(self, request, *args, **kwargs):

        def get_img_for_template(img):
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode('ascii')

            return img_str

        form = UploadMultiImageForm(request.POST, request.FILES)
        if form.is_valid():
            from PIL import Image
            img_raw = form.cleaned_data['img']
            print(img_raw)
            # for img_raw in form.cleaned_data['img']:
            img = Image.open(img_raw)
            print(img.size)

            # reshape
            will_reshape = False
            if will_reshape:
                w, h = img.size
                ratio = w / h
                new_w = int(ratio * 480)
                new_h = 480
                img = img.resize((new_w, new_h))

            imgs = face_recognition(img)
            img_strs = list()
            for img in imgs:
                img_str = get_img_for_template(Image.fromarray(img))
                img_strs.append(img_str)

            context = self.get_context_data(**kwargs)
            context['range'] = range(len(img_strs))
            context['img_strs'] = img_strs
            context['zip_range'] = zip(range(len(img_strs)), img_strs)

            return self.render_to_response(context)


class FaceSaveView(LoginRequiredMixin, TemplateView):
    login_url = '/accounts/login/'
    template_name = 'registration/save_face.html'

    def post(self, request, *args, **kwargs):

        def get_str_to_img(b64_str):
            imgdata = base64.b64decode(b64_str)
            image = Image.open(BytesIO(imgdata))

            return np.array(image)

        # print(request.POST, request.FILES)
        data = request.POST
        print(data.keys())
        b_names = request.POST.getlist('b_name')
        checklist = request.POST.getlist('images')
        images = request.POST.getlist('hidden_image')

        checklist = list(map(int, checklist))
        images = list(map(get_str_to_img, images))
        for i in checklist:
            b_names[i] = b_names[i].lstrip().rstrip()
            save_features(images[i], b_names[i], request.user.uid)

            borrower = Borrower(b_name=b_names[i])
            user = User.objects.get(uid=request.user.uid)
            print(user.uid)
            print(type(user))
            userborrower = UserBorrower(uid=user, bid=borrower)
            borrower.save()
            userborrower.save()

            img = Image.fromarray(images[i])
            img.show()

        return HttpResponseRedirect('/')
