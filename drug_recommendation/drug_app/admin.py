from django.contrib import admin
from .models import PredictionHistory

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'age', 'sex', 'blood_pressure', 'cholesterol', 'na_to_k', 'predicted_drug', 'best_model', 'created_at']
    list_filter = ['predicted_drug', 'sex', 'blood_pressure']
    search_fields = ['user__username', 'predicted_drug']
