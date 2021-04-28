from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from . import views

app_name = 'history'
urlpatterns = [
    path('', views.HistoryView.as_view(), name='history'),
    path('<int:bid>/',views.HistoryViewDetail.as_view(),name='history_detail'),
    path('<int:bid>/<int:tid>',views.HistoryVideo.as_view(),name='history_video'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns += staticfiles_urlpatterns() 