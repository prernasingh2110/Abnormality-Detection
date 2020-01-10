from django.db import models


class Xray(models.Model):
    condition = models.CharField(max_length=15)
    body_part = models.CharField(max_length=15)
    patient = models.CharField(max_length=100)
    bone_image = models.FileField()
    probability = models.CharField(max_length=10)

    def __str__(self):
        return self.condition + self.body_part + self.patient
