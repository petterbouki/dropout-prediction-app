# 🎓 Prédiction du Risque d'Abandon Scolaire

Application Streamlit pour la prédiction du risque d'abandon scolaire basée sur des algorithmes de Machine Learning.

## 🚀 Installation & Lancement

```bash
# 1. Cloner / télécharger le dossier
cd dropout_prediction_app

# 2. (Optionnel) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

Accéder à l'application : **http://localhost:8501**

## 📁 Structure du projet

```
dropout_prediction_app/
│
├── app.py                      ← Point d'entrée Streamlit (navigation)
├── requirements.txt            ← Dépendances Python
├── README.md                   ← Ce fichier
│
├── pages/                      ← Pages de l'application
│   ├── accueil.py              ← Page d'accueil
│   ├── prediction.py           ← Prédiction individuelle + par lot CSV
│   ├── dashboard.py            ← Performances modèles + EDA interactive
│   ├── donnees.py              ← Exploration et filtrage du dataset
│   └── about.py                ← À propos du projet
│
├── utils/                      ← Fonctions utilitaires
│   └── model_utils.py          ← Chargement modèles, prédiction, niveaux de risque
│
├── models/                     ← Modèles entraînés (sérialisés)
│   ├── random_forest.pkl       ← Random Forest (meilleur modèle)
│   ├── logistic_regression.pkl ← Régression Logistique
│   ├── svm.pkl                 ← Support Vector Machine
│   ├── scaler.pkl              ← StandardScaler
│   └── feature_columns.pkl     ← Liste des colonnes d'entrée
│
├── data/                       ← Dataset
│   └── student_dropout_dataset.csv
│
└── assets/                     ← Figures et images
    ├── fig_eda.png
    ├── fig_corr.png
    ├── fig_confusion.png
    ├── fig_metrics.png
    └── fig_importance.png
```

## 🤖 Modèles disponibles

| Modèle | Accuracy | F1-Score |
|--------|----------|----------|
| **Random Forest** ⭐ | 98.3% | 96.6% |
| Logistic Regression | 86.7% | 73.3% |
| SVM | 85.0% | 66.7% |

## 📋 Fonctionnalités

- **Prédiction individuelle** : saisie du profil d'un étudiant → résultat instantané avec jauge de probabilité
- **Prédiction par lot** : import d'un fichier CSV → prédictions en masse téléchargeables
- **Tableau de bord** : graphiques comparatifs, matrices de confusion, radar chart
- **EDA interactive** : filtres dynamiques, scatter plots, histogrammes, matrice de corrélation
- **Recommandations automatiques** : conseils personnalisés selon le profil détecté
