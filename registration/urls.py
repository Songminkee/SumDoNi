from django.urls import path

from . import views

app_name = 'registration'
urlpatterns = [
    path('', views.RegistrationView.as_view(), name='registration'),
    path('face_recognition', views.FaceRecognitionView.as_view(), name='face_recognition'),
    path('save_face', views.FaceSaveView.as_view(), name='save_face')
]