# accounts/views.py
from django.urls import reverse_lazy
from django.views import generic

from .forms import SignupForm


class SignUpView(generic.CreateView):
    form_class = SignupForm
    success_url = reverse_lazy('login')
    template_name = 'accounts/signup.html'


# Reference
# - Manager isn't available; 'auth.User' has been swapped for 'accounts.User'
#   https://stackoverflow.com/questions/17873855/manager-isnt-available-user-has-been-swapped-for-pet-person