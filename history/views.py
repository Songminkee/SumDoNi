from django.shortcuts import render
from django.views.generic import TemplateView
from django.contrib.auth.decorators import login_required
from accounts.models import User
from registration.models import Borrower
from history.models import TrackingLog
import glob
import os

class HistoryView(TemplateView):
    def get(self,request,*args, **kwargs):
        uid = User.objects.get(username=request.user).uid
        borrowers = Borrower.objects.filter(userborrower__uid=uid)
        return render(request,'history/history.html',{'borrowers':borrowers})

class HistoryViewDetail(TemplateView):
    def get(self,request,bid,*args, **kwargs): 
        logs = TrackingLog.objects.filter(borrowertrackinglog__bid=bid)        
        return render(request,'history/history_detail.html',{'logs':logs,'bid':bid})

class HistoryVideo(TemplateView):
    def get(self,request,tid,*args, **kwargs): 
        log = TrackingLog.objects.get(tid=tid)        
        return render(request,'history/history_video.html',{'log':log})