from django.http import HttpResponse
from django.views import View
from django.views.generic import TemplateView

import cv2


class TrackingView(TemplateView):
    template_name = 'tracking/tracking.html'
    

class TrackingOnView(View):

    def get(self, request, *args, **kwargs):
        ip = '211.55.92.51'
        port = 5000
        url = 'rtsp://hanamart-public:hanamart-public@121.151.100.57:9001//h264Preview_01_sub'

        # Check IP camera activation
        repeat_count, is_activate = 0, False
        while repeat_count < 10:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                cv2.waitKey(1)
                repeat_count += 1
                print(f'Try load Video (#{repeat_count})')
            else:
                is_activate = True
                break

        if is_activate:
            return HttpResponse('IP camera is activated.')
        else:
            return HttpResponse('IP camera is not activated.')

    def post(self, request, *args, **kwargs):
        return HttpResponse('Tracking On.')


class TrackingOffView(View):

    def get(self, request, *args, **kwargs):
        return HttpResponse('Tracking Off.')

    def post(self, request, *args, **kwargs):
        return HttpResponse('Tracking Off.')
