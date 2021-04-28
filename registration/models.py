from django.db import models
from accounts.models import User


class Borrower(models.Model):
    bid = models.AutoField(primary_key=True)
    b_name = models.CharField(max_length=150)

    def __str__(self):
        return f'bid : {self.bid}, b_name : {self.b_name}'


class UserBorrower(models.Model):
    ubid = models.AutoField(primary_key=True)
    uid = models.ForeignKey(User, on_delete=models.CASCADE)
    bid = models.ForeignKey(Borrower, on_delete=models.CASCADE)

    def __str__(self):
        return f'ubid : {self.ubid}, uid : {self.uid}, bid : {self.bid}'

# Reference
# - Automatic primary key field
#   https://docs.djangoproject.com/en/3.1/topics/db/models/#automatic-primary-key-fields
