"""
Utilitaires pour le chargement et l'utilisation des modèles ML.
"""
import pickle
import numpy as np
import pandas as pd
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

MODEL_FILES = {
     "Logistic Regression": "logistic_regression.pkl",
   
}

MODEL_METRICS = {
    "Random Forest": {
        "accuracy": 0.9833, "precision": 1.0000,
        "recall": 0.9333, "f1": 0.9655,
        "cv_f1": "0.947 ± 0.035",
        "description": "Meilleur modèle — Ensemble de 200 arbres de décision. Très robuste, précision parfaite (0 faux positif).",
        "color": "#27ae60",
    },
    "Logistic Regression": {
        "accuracy": 0.8667, "precision": 0.7333,
        "recall": 0.7333, "f1": 0.7333,
        "cv_f1": "0.753 ± 0.057",
        "description": "Modèle linéaire de référence, rapide et interprétable.",
        "color": "#2980b9",
    },
    "SVM": {
        "accuracy": 0.8500, "precision": 0.7500,
        "recall": 0.6000, "f1": 0.6667,
        "cv_f1": "0.729 ± 0.018",
        "description": "Support Vector Machine avec noyau RBF. Efficace en haute dimension.",
        "color": "#8e44ad",
    },
}


def load_model(model_name: str):
    path = os.path.join(MODELS_DIR, MODEL_FILES[model_name])
    with open(path, "rb") as f:
        return pickle.load(f)


def load_scaler():
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_columns():
    path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_input_df(age, gender, average_grade, absenteeism_rate,
                   internet_access, study_time_hours, extra_activities,
                   feature_columns):
    """Construit le DataFrame d'entrée pour la prédiction."""
    student = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "average_grade": average_grade,
        "absenteeism_rate": absenteeism_rate,
        "internet_access": internet_access,
        "study_time_hours": study_time_hours,
        "extra_activities": extra_activities,
    }])

    # Feature engineering
    student["presence_rate"] = 1 - student["absenteeism_rate"]
    student["global_score"] = (
        (student["average_grade"] / 20) * 0.5
        + (1 - student["absenteeism_rate"]) * 0.3
        + (student["study_time_hours"] / 5) * 0.2
    )

    # One-hot encoding
    student = pd.get_dummies(
        student,
        columns=["gender", "internet_access", "extra_activities"],
        drop_first=True,
    )

    # Aligner les colonnes
    for col in feature_columns:
        if col not in student.columns:
            student[col] = 0
    student = student[feature_columns]

    return student


def predict(model, scaler, input_df):
    """Retourne (prediction, probabilite_risque)."""
    X_scaled = scaler.transform(input_df)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]
    return int(pred), float(proba)


def risk_level(proba: float):
    """Retourne (niveau, couleur, emoji) selon la probabilité."""
    if proba < 0.30:
        return "Faible", "#27ae60", "✅"
    elif proba < 0.60:
        return "Modéré", "#f39c12", "⚠️"
    else:
        return "Élevé", "#e74c3c", "🚨"
