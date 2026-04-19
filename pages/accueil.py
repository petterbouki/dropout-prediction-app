"""Page d'accueil."""
import streamlit as st

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%);
    border-radius: 16px;
    padding: 40px 48px;
    color: white;
    margin-bottom: 32px;
}
.hero h1 { font-size: 2.6rem; margin-bottom: 8px; }
.hero p  { font-size: 1.1rem; opacity: 0.9; }
.card {
    background: #f8f9fa;
    border-left: 5px solid #2E75B6;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card h3 { margin: 0 0 6px; color: #1F4E79; }
.stat-box {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.stat-num  { font-size: 2.2rem; font-weight: 700; color: #1F4E79; }
.stat-label{ font-size: 0.85rem; color: #6c757d; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1> Prédiction du Risque d'Abandon Scolaire</h1>
    <p>Système intelligent de détection précoce basé sur le Machine Learning.<br>
       Identifiez les étudiants en difficulté avant qu'il ne soit trop tard.</p>
</div>
""", unsafe_allow_html=True)

# ── Statistiques clés ──────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

stats = [
    ("300", "Étudiants analysés"),
    ("98.3%", "Accuracy (Random Forest)"),
    ("96.6%", "F1-Score meilleur modèle"),
    ("3", "Algorithmes comparés"),
]
for col, (num, label) in zip([col1, col2, col3, col4], stats):
    with col:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{num}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Description du projet ──────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.markdown("##  À propos du projet")
    st.markdown("""
    Ce système utilise des algorithmes de **Machine Learning** pour prédire,
    à partir de données académiques et comportementales, si un étudiant
    risque d'abandonner ses études.

    **Logique de détection :** Un étudiant est considéré à risque s'il remplit
    au moins **2 des 3 conditions** suivantes :
    -  Moyenne générale **< 10 / 20**
    -  Taux d'absentéisme **> 30 %**
    -  Temps d'étude journalier **< 1 heure**
    """)

    st.markdown("""
    <div class="card">
        <h3> Commencer</h3>
        <p>Utilisez le menu de gauche pour naviguer dans l'application :</p>
        <ul>
            <li><b>Prédiction</b> — Évaluer le risque d'un étudiant en temps réel</li>
            <li><b>Tableau de bord</b> — Visualiser les performances des modèles</li>
            <li><b>Données</b> — Explorer le dataset complet</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("##  Modèles disponibles")

    models_info = [
        (" Random Forest", "98.3%", "#27ae60",
         "Meilleur modèle — 200 arbres, 0 faux positif"),
        (" Logistic Regression", "86.7%", "#2980b9",
         "Modèle linéaire de référence, rapide"),
        (" SVM", "85.0%", "#8e44ad",
         "Support Vector Machine, noyau RBF"),
    ]

    for name, acc, color, desc in models_info:
        st.markdown(f"""
        <div style="background:white; border:1px solid #dee2e6; border-radius:10px;
                    padding:14px 18px; margin-bottom:12px; border-left:5px solid {color};">
            <b style="color:{color};">{name}</b>
            <span style="float:right; font-weight:700; color:{color};">{acc}</span><br>
            <small style="color:#6c757d;">{desc}</small>
        </div>
        """, unsafe_allow_html=True)

# ── Pipeline ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("##  Pipeline du Projet")

steps = [
    
    ("1", "Data\nCollection", "#9b59b6"),
    ("2", "EDA &\nPreprocessing", "#e67e22"),
    ("3", "Feature\nEngineering", "#e74c3c"),
    ("4", "Modélisation", "#27ae60"),
    ("5", "Évaluation &\nOptimisation", "#1abc9c"),
    ("6", "Déploiement\nStreamlit", "#f39c12"),
]

cols = st.columns(len(steps))
for col, (num, label, color) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div style="text-align:center;">
            <div style="background:{color}; color:white; border-radius:50%;
                        width:48px; height:48px; line-height:48px;
                        font-size:1.3rem; font-weight:700; margin:auto;">{num}</div>
            <div style="font-size:0.78rem; color:#495057; margin-top:8px;
                        font-weight:600; white-space:pre-line;">{label}</div>
        </div>
        """, unsafe_allow_html=True)
