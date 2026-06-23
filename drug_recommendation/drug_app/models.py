from django.db import models
from django.contrib.auth.models import User


class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    age = models.IntegerField()
    sex = models.CharField(max_length=1)
    blood_pressure = models.CharField(max_length=10)
    cholesterol = models.CharField(max_length=10)
    na_to_k = models.FloatField()
    predicted_drug = models.CharField(max_length=50)
    best_model = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.predicted_drug} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
