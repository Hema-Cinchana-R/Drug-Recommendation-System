from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from .forms import RegisterForm, PredictionForm
from .models import PredictionHistory
from .ml_utils import predict_drug


def home_view(request):
    if request.user.is_authenticated:
        return redirect('predict')
    return redirect('login')


def register_view(request):
    if request.user.is_authenticated:
        return redirect('predict')
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome, {user.first_name}! Account created successfully.')
            return redirect('predict')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = RegisterForm()
    return render(request, 'drug_app/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('predict')
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f'Welcome back, {user.first_name or user.username}!')
            return redirect('predict')
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()
    return render(request, 'drug_app/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('login')


@login_required
def predict_view(request):
    result = None
    form = PredictionForm()

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            age = form.cleaned_data['age']
            sex = form.cleaned_data['sex']
            blood_pressure = form.cleaned_data['blood_pressure']
            cholesterol = form.cleaned_data['cholesterol']
            na_to_k = form.cleaned_data['na_to_k']

            try:
                result = predict_drug(age, sex, blood_pressure, cholesterol, na_to_k)

                PredictionHistory.objects.create(
                    user=request.user,
                    age=age,
                    sex=sex,
                    blood_pressure=blood_pressure,
                    cholesterol=cholesterol,
                    na_to_k=na_to_k,
                    predicted_drug=result['predicted_drug'],
                    best_model=result['best_model'],
                )
                messages.success(request, 'Prediction completed successfully!')
            except Exception as e:
                messages.error(request, f'Prediction error: {str(e)}')

    return render(request, 'drug_app/predict.html', {
        'form': form,
        'result': result,
        'algos': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Naive Bayes'],
    })


@login_required
def history_view(request):
    history = PredictionHistory.objects.filter(user=request.user)
    return render(request, 'drug_app/history.html', {'history': history})


@login_required
def accuracy_view(request):
    model_accuracies = {
        'Logistic Regression': 87.50,
        'Decision Tree': 100.00,
        'Random Forest': 100.00,
        'SVM': 62.50,
        'KNN': 70.00,
        'Naive Bayes': 92.50,
    }
    best_model = max(model_accuracies, key=model_accuracies.get)
    return render(request, 'drug_app/accuracy.html', {
        'model_accuracies': model_accuracies,
        'best_model': best_model,
    })
