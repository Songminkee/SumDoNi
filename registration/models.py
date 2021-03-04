from django.db import models
from accounts.models import User


class Borrower(models.Model):
    bid = models.AutoField(primary_key=True)
    b_name = models.CharField(max_length=150)


class UserBorrower(models.Model):
    ubid = models.AutoField(primary_key=True)
    uid = models.ForeignKey(User, on_delete=models.CASCADE)
    bid = models.ForeignKey(Borrower, on_delete=models.CASCADE)


# Reference
# - Automatic primary key field
#   https://docs.djangoproject.com/en/3.1/topics/db/models/#automatic-primary-key-fields
