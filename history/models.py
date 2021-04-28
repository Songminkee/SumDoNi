from django.db import models
from registration.models import Borrower


class TrackingLog(models.Model):
    tid = models.AutoField(primary_key=True)
    start_datetime = models.DateTimeField()
    end_datetime = models.DateTimeField()
    video_path = models.CharField(max_length=255)

    def __str__(self):
        return f'video = {self.video_path}\n{self.start_datetime} ~ {self.end_datetime}'


class BorrowerTrackingLog(models.Model):
    btid = models.AutoField(primary_key=True)
    bid = models.ForeignKey(Borrower, on_delete=models.CASCADE)
    tid = models.ForeignKey(TrackingLog, on_delete=models.CASCADE)
    def __str__(self):
        return f'btid : {self.btid}, bid : {self.bid}, tid :  {self.tid}'
