"""
Application Streamlit — Prédiction du Risque d'Abandon Scolaire
Point d'entrée principal (navigation multi-pages).
"""
import streamlit as st

st.set_page_config(
    page_title="Abandon Scolaire — ML App",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/graduation-cap.png", width=80
)
st.sidebar.title("🎓 Dropout Predictor")
st.sidebar.markdown("---")

pages = {
    "🏠  Accueil": "pages/accueil.py",
    "🔮  Prédiction": "pages/prediction.py",
    "📊  Tableau de bord": "pages/dashboard.py",
    "📂  Données": "pages/donnees.py",
    
}

# On utilise session_state pour mémoriser la page active
if "page" not in st.session_state:
    st.session_state.page = "🏠  Accueil"

for label in pages:
    if st.sidebar.button(label, use_container_width=True,
                         type="primary" if st.session_state.page == label else "secondary"):
        st.session_state.page = label
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Examen IA · Avril 2026")

# ── Routing ────────────────────────────────────────────────────────────────
page_file = pages[st.session_state.page]

with open(page_file, encoding="utf-8") as f:
    exec(compile(f.read(), page_file, "exec"), {"__name__": "__main__"})
