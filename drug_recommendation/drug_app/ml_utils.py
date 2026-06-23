import os
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / 'ml_models'

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'Random Forest': 'random_forest.pkl',
    'SVM': 'svm.pkl',
    'KNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
}


def load_encoders():
    le_sex = joblib.load(ML_DIR / 'le_sex.pkl')
    le_bp = joblib.load(ML_DIR / 'le_bp.pkl')
    le_chol = joblib.load(ML_DIR / 'le_chol.pkl')
    le_drug = joblib.load(ML_DIR / 'le_drug.pkl')
    return le_sex, le_bp, le_chol, le_drug


def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k):
    le_sex, le_bp, le_chol, le_drug = load_encoders()

    sex_enc = le_sex.transform([sex])[0]
    bp_enc = le_bp.transform([blood_pressure])[0]
    chol_enc = le_chol.transform([cholesterol])[0]

    features = np.array([[age, sex_enc, bp_enc, chol_enc, na_to_k]])

    results = {}
    predictions = {}

    for model_name, model_file in MODEL_FILES.items():
        model_path = ML_DIR / model_file
        if model_path.exists():
            model = joblib.load(model_path)
            pred_enc = model.predict(features)[0]
            pred_drug = le_drug.inverse_transform([pred_enc])[0]
            predictions[model_name] = pred_drug

    # Majority vote
    from collections import Counter
    vote_counts = Counter(predictions.values())
    majority_drug = vote_counts.most_common(1)[0][0]

    # Best model accuracies (pre-computed)
    model_accuracies = {
        'Logistic Regression': 87.50,
        'Decision Tree': 100.00,
        'Random Forest': 100.00,
        'SVM': 62.50,
        'KNN': 70.00,
        'Naive Bayes': 92.50,
    }

    best_model = max(model_accuracies, key=model_accuracies.get)

    return {
        'predicted_drug': majority_drug,
        'model_predictions': predictions,
        'model_accuracies': model_accuracies,
        'best_model': best_model,
        'best_accuracy': model_accuracies[best_model],
    }
