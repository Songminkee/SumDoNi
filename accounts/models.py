from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    cctv_ip = models.CharField(max_length=15)
    tracking_status = models.BooleanField(default=False)
