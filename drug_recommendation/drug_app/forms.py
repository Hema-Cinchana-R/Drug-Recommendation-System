from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=50, required=True)
    last_name = forms.CharField(max_length=50, required=True)

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']


class PredictionForm(forms.Form):
    AGE_CHOICES = [(i, str(i)) for i in range(1, 101)]
    SEX_CHOICES = [('M', 'Male'), ('F', 'Female')]
    BP_CHOICES = [('HIGH', 'High'), ('NORMAL', 'Normal'), ('LOW', 'Low')]
    CHOLESTEROL_CHOICES = [('HIGH', 'High'), ('NORMAL', 'Normal')]

    age = forms.IntegerField(
        min_value=1, max_value=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter age (1-100)'})
    )
    sex = forms.ChoiceField(
        choices=SEX_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    blood_pressure = forms.ChoiceField(
        choices=BP_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    cholesterol = forms.ChoiceField(
        choices=CHOLESTEROL_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    na_to_k = forms.FloatField(
        min_value=0.0, max_value=50.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 15.4 (range: 6–38)', 'step': '0.001'})
    )
