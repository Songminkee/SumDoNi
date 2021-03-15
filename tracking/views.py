# Reference
# - Messages
#   https://docs.djangoproject.com/en/3.1/ref/contrib/messages/#django.contrib.messages.storage.fallback.FallbackStorage


from django.http import HttpResponse, HttpResponseRedirect
from django.views import View
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse
from django.contrib import messages

from accounts.models import User

import cv2


class TrackingView(LoginRequiredMixin, TemplateView):
    login_url = '/accounts/login/'
    template_name = 'tracking/tracking.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = User.objects.get(username=self.request.user)

        storage = messages.get_messages(self.request)
        for message in storage:
            context['msg'] = message
            break

        return context
    

class TrackingOnView(LoginRequiredMixin, View):
    login_url = '/accounts/login/'

    def get(self, request, *args, **kwargs):
        user = User.objects.get(username=request.user)
        cctv_id, cctv_pw = user.cctv_id, user.cctv_pw
        cctv_ip = user.cctv_ip
        cctv_port = user.cctv_port
        cctv_quality = 'sub' # main: QHD, sub: HD or SD?
        url = f'rtsp://{cctv_id}:{cctv_pw}@{cctv_ip}:{cctv_port}//h264Preview_01_{cctv_quality}'

        # Check IP camera activation
        repeat_count, is_activate = 0, False
        while repeat_count < 3:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                cv2.waitKey(1)
                repeat_count += 1
                print(f'Try load Video (#{repeat_count})')
            else:
                is_activate = True
                break

        if is_activate:
            user = User.objects.get(username=request.user)
            user.tracking_status = True
            user.save()
            return HttpResponseRedirect('/tracking')
        else:
            messages.add_message(request, messages.INFO, 'IP Camera is not activated.')
            return HttpResponseRedirect('/tracking')

    def post(self, request, *args, **kwargs):
        return HttpResponse('Tracking On.')


class TrackingOffView(LoginRequiredMixin, View):
    login_url = '/accounts/login/'

    def get(self, request, *args, **kwargs):
        user = User.objects.get(username=self.request.user)
        user.tracking_status = False
        user.save()
        return HttpResponseRedirect('/tracking')
