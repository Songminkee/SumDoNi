from django.shortcuts import render
from django.views.generic import TemplateView


class RegistrationView(TemplateView):
    template_name = 'registration/registration.html'
