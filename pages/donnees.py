"""Page d'exploration du dataset."""
import streamlit as st
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "student_dropout_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("📂 Exploration des Données")
st.markdown(f"Dataset complet : **{len(df)} étudiants × {len(df.columns)} variables**")

# ── Filtres ────────────────────────────────────────────────────────────────
st.markdown("### 🔎 Filtrer le dataset")

c1, c2, c3 = st.columns(3)
with c1:
    genre_filter = st.multiselect("Genre", df["gender"].unique(), default=list(df["gender"].unique()))
with c2:
    risk_filter = st.multiselect("Risque d'abandon", [0, 1], default=[0, 1],
                                  format_func=lambda x: "✅ Stable" if x == 0 else "⚠️ À risque")
with c3:
    age_range = st.slider("Âge", int(df["age"].min()), int(df["age"].max()),
                           (int(df["age"].min()), int(df["age"].max())))

note_range = st.slider("Moyenne générale", 0.0, 20.0, (0.0, 20.0), 0.1)

# Apply filters
df_filtered = df[
    df["gender"].isin(genre_filter) &
    df["dropout_risk"].isin(risk_filter) &
    df["age"].between(*age_range) &
    df["average_grade"].between(*note_range)
]

st.caption(f"{len(df_filtered)} étudiants correspondent aux filtres")

# ── Tableau ────────────────────────────────────────────────────────────────
st.dataframe(
    df_filtered.style.map(
        lambda v: "background-color:#fde8e8; color:#c0392b;" if v == 1 else
                  "background-color:#e8f8e8; color:#1e8449;",
        subset=["dropout_risk"]
    ),
    use_container_width=True,
    height=420,
)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "⬇️ Télécharger les données filtrées",
        data=df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="dataset_filtre.csv",
        mime="text/csv",
    )
with col_dl2:
    st.download_button(
        "⬇️ Télécharger le dataset complet",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_dropout_dataset.csv",
        mime="text/csv",
    )

# ── Statistiques descriptives ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Statistiques descriptives")
st.dataframe(df_filtered.describe().round(3), use_container_width=True)

# ── Variables catégorielles ────────────────────────────────────────────────
st.markdown("### 🏷️ Variables catégorielles")
c1, c2, c3 = st.columns(3)
for col, var in zip([c1, c2, c3], ["gender", "internet_access", "extra_activities"]):
    with col:
        st.markdown(f"**{var}**")
        st.dataframe(
            df_filtered[var].value_counts().rename("Effectif").to_frame(),
            use_container_width=True,
        )
