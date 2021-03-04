from django.shortcuts import render
from django.views.generic import TemplateView


class HistoryView(TemplateView):
    template_name = 'history/history.html'
