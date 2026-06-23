"""
Run this script ONCE to train and save all ML models.
Usage: python train_models.py
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'drug200.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  Drug Recommendation System - Model Training")
print("=" * 60)

# Load data
df = pd.read_csv(DATASET_PATH)
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Drug classes: {df['Drug'].unique().tolist()}")
print(f"   Distribution:\n{df['Drug'].value_counts().to_string()}\n")

# Encode categorical features
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()

df['Sex_enc'] = le_sex.fit_transform(df['Sex'])
df['BP_enc'] = le_bp.fit_transform(df['BP'])
df['Cholesterol_enc'] = le_chol.fit_transform(df['Cholesterol'])
df['Drug_enc'] = le_drug.fit_transform(df['Drug'])

# Features and target
X = df[['Age', 'Sex_enc', 'BP_enc', 'Cholesterol_enc', 'Na_to_K']]
y = df['Drug_enc']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
}

results = {}
print(f"{'Algorithm':<25} {'Accuracy':>10}")
print("-" * 38)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    results[name] = round(acc, 2)
    print(f"{name:<25} {acc:>9.2f}%")
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(model, os.path.join(MODELS_DIR, f'{safe_name}.pkl'))

# Save encoders
joblib.dump(le_sex, os.path.join(MODELS_DIR, 'le_sex.pkl'))
joblib.dump(le_bp, os.path.join(MODELS_DIR, 'le_bp.pkl'))
joblib.dump(le_chol, os.path.join(MODELS_DIR, 'le_chol.pkl'))
joblib.dump(le_drug, os.path.join(MODELS_DIR, 'le_drug.pkl'))

best = max(results, key=results.get)
print(f"\n🏆 Best Model: {best} ({results[best]}%)")
print(f"\n✅ All models saved to: {MODELS_DIR}")
print("\n" + "=" * 60)
print("  Training Complete! You can now run the Django server.")
print("  python manage.py runserver")
print("=" * 60)
