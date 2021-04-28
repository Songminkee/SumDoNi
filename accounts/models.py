from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    uid = models.AutoField(primary_key=True)
    cctv_ip = models.CharField(max_length=15)
    cctv_port = models.CharField(max_length=4)
    tracking_status = models.BooleanField(default=False)
    cctv_id = models.CharField(max_length=30)
    cctv_pw = models.CharField(max_length=30)
