from django.urls import path
from django.contrib.auth import views as auth_views

from .views import SignUpView


urlpatterns = [
    path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', SignUpView.as_view(), name='signup'),
]

# Reference
# - LoginView Redirect
#   https://roseline124.github.io/django/2019/04/03/pickmeal-loginview.html