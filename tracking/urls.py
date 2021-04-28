from django.urls import path

from . import views

app_name = 'tracking'
urlpatterns = [
    path('', views.TrackingView.as_view(), name='tracking'),
    path('on/', views.TrackingOnView.as_view(), name='tracking_on'),
    path('off/', views.TrackingOffView.as_view(), name='tracking_off'),
]