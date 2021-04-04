# Reference
# - Messages
#   https://docs.djangoproject.com/en/3.1/ref/contrib/messages/#django.contrib.messages.storage.fallback.FallbackStorage


from django.http import HttpResponse, HttpResponseRedirect
from django.views import View
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse
from django.contrib import messages
from django.conf import settings
from django.utils.decorators import classonlymethod
from asgiref.sync import sync_to_async

from accounts.models import User
from registration.models import Borrower
from history.models import TrackingLog, BorrowerTrackingLog
from .tracking_utils import track_face

import subprocess
import os
import cv2
import numpy as np
import torch
import asyncio


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


class TrackingOnView(View):
    # login_url = '/accounts/login/'
    
    async def tracking(self, cap, username):
        await settings.TRACKING_MODELS.track_face(cap=cap, User=User, username=username,
                                                  Borrower=Borrower,
                                                  TrackingLog=TrackingLog,
                                                  BorrowerTrackingLog=BorrowerTrackingLog)

    async def get(self, request, *args, **kwargs):
        user = await sync_to_async(User.objects.get, thread_sensitive=True)(username=request.user)
        cctv_id, cctv_pw = user.cctv_id, user.cctv_pw
        cctv_ip = user.cctv_ip
        cctv_port = user.cctv_port
        cctv_quality = 'sub' # main: QHD, sub: HD or SD?
        url = f'rtsp://{cctv_id}:{cctv_pw}@{cctv_ip}:{cctv_port}//h264Preview_01_{cctv_quality}'

        # Check IP camera activation
        repeat_count, is_activate = 0, True
        cap = cv2.VideoCapture('sample_video_without_mask2.mp4')
        # while repeat_count < 3:
        #     cap = cv2.VideoCapture(url)
        #     if not cap.isOpened():
        #         cv2.waitKey(1)
        #         repeat_count += 1
        #         print(f'Try load Video (#{repeat_count})')
        #     else:
        #         is_activate = True
        #         break

        if is_activate:
            print('Tracking is activated!!')

            # Update an user's tracking status
            user = await sync_to_async(User.objects.get, thread_sensitive=True)(username=request.user)
            user.tracking_status = True
            await sync_to_async(user.save, thread_sensitive=True)()

            # Track faces
            loop = asyncio.get_event_loop()
            loop.create_task(self.tracking(cap, request.user))

            # while True:
            #     # Extract a frame from a remote IP CCTV
            #     ret, frame = cap.read()
            #     if not ret:
            #         break

            #     # Preprocess a frame
            #     device = torch.device("cuda")
            #     img = torch.from_numpy(np.float32(frame.copy()).transpose(2,0,1))
            #     print(img.size())
            #     img = img.to(device).half()
            #     img /= 255.0
            #     if img.ndimension() == 3:
            #         img = img.unsqueeze(0)

            #     # https://docs.python.org/ko/3.9/library/subprocess.html
            #     # p = subprocess.Popen(['python', './tracker/test.py'])
            #     # print(f'Process {p.pid} is started!!')
            #     # settings.PROCESS_DICT[p.pid] = p
            #     # print(settings.PROCESS_DICT)
            #     # user.pid = p.pid

            #     person_detector = settings.TRACKING_MODELS.person_detector
            #     pred = person_detector(img, augment=False)[0]
            #     print(f'User: {user.uid}')

            #     import time
            #     time.sleep(1)

            return HttpResponseRedirect('/tracking')
        else:
            messages.add_message(request, messages.INFO, 'IP Camera is not activated.')
            return HttpResponseRedirect('/tracking')

    def post(self, request, *args, **kwargs):
        return HttpResponse('Tracking On.')
    
    @classonlymethod
    def as_view(cls, **initkwargs):
        view = super().as_view(**initkwargs)
        view._is_coroutine = asyncio.coroutines._is_coroutine
        return view


class TrackingOffView(LoginRequiredMixin, View):
    login_url = '/accounts/login/'

    def get(self, request, *args, **kwargs):
        user = User.objects.get(username=self.request.user)
        user.tracking_status = False
        if user.pid != '':
            os.kill(int(user.pid), 9)
            print(f'Process {user.pid} is killed!!')
            user.pid = ''
            
        user.save()
        return HttpResponseRedirect('/tracking')

class TestView(View):
    login_url = '/accounts/login/'
    
    async def get(self, request, *args, **kwargs):
        user = await sync_to_async(User.objects.get, thread_sensitive=True)(username=self.request.user)
        loop = asyncio.get_event_loop()
        loop.create_task(self.test(user))
        return HttpResponseRedirect('/tracking')
    
    async def test(self, user):
        while True:
            import os
            print(os.getcwd())
            img = cv2.imread('/home/borrower_tracker/tracking/person.jpg')
            img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)
            device = torch.device("cuda")
            img = torch.from_numpy(np.float32(img.copy()).transpose(2,0,1))
            print(img.size())
            img = img.to(device).half()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            person_detector = settings.TRACKING_MODELS.person_detector
            pred = person_detector(img, augment=False)[0]
            print(f'User: {user.uid}')
            # print(f'Class Probability: {pred}')

            from time import sleep
            await asyncio.sleep(1)

    @classonlymethod
    def as_view(cls, **initkwargs):
        view = super().as_view(**initkwargs)
        view._is_coroutine = asyncio.coroutines._is_coroutine
        return view

