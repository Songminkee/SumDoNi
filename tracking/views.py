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
from registration.models import Borrower, UserBorrower
from history.models import TrackingLog, BorrowerTrackingLog

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
                                                  UserBorrower=UserBorrower,
                                                  TrackingLog=TrackingLog,
                                                  BorrowerTrackingLog=BorrowerTrackingLog)

    async def get(self, request, *args, **kwargs):
        # Set parameters for tracking
        user = await sync_to_async(User.objects.get, thread_sensitive=True)(username=request.user)
        cctv_id, cctv_pw = user.cctv_id, user.cctv_pw
        cctv_ip = user.cctv_ip
        cctv_port = user.cctv_port
        cctv_quality = 'sub' # main: QHD, sub: HD or SD?
        url = f'rtsp://{cctv_id}:{cctv_pw}@{cctv_ip}:{cctv_port}//h264Preview_01_{cctv_quality}'

        # Check IP camera activation
        repeat_count, is_activate = 0, False
        # camera_type = 'video'
        camera_type = 'cctv'
        if camera_type == 'video':
            cap = cv2.VideoCapture('hanamart-112128-112658_mask-o_glasses-x_cap-x.mp4')
            # cap = cv2.VideoCapture('hanamart-112128-112658_mask-o_glasses-o_cap-x.mp4')
            # cap = cv2.VideoCapture('hanamart-112128-112658_mask-o_glasses-x_cap-o.mp4')
            # cap = cv2.VideoCapture('hanamart-112128-112658_mask-o_glasses-o_cap-o.mp4')
            # cap = cv2.VideoCapture('hanamart-112128-112658_mask-x_glasses-x_cap-x.mp4')
            # cap = cv2.VideoCapture('hanamart-112128-112658_mask-x_glasses-o_cap-x.mp4')
            # cap = cv2.VideoCapture('hanamart-112128-112658_mask-x_glasses-x_cap-o.mp4')
            # cap = cv2.VideoCapture('hanamart-112128-112658_mask-x_glasses-o_cap-o.mp4')
            is_activate = True
        else:
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
            print('Tracking is activated!!')

            # Update an user's tracking status
            user = await sync_to_async(User.objects.get, thread_sensitive=True)(username=request.user)
            user.tracking_status = True
            await sync_to_async(user.save, thread_sensitive=True)()

            # Track faces
            loop = asyncio.get_event_loop()
            loop.create_task(self.tracking(cap, request.user))

            return HttpResponseRedirect('/tracking')
        else:
            messages.add_message(request, messages.INFO, 'IP Camera is not activated.')
            return HttpResponseRedirect('/tracking')

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
        user.save()
        return HttpResponseRedirect('/tracking')
