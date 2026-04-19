"""Page de prédiction individuelle et prédiction par lot."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.model_utils import (
    load_model, load_scaler, load_feature_columns,
    build_input_df, predict, risk_level, MODEL_METRICS
)

# ── Cache ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_assets():
    scaler = load_scaler()
    features = load_feature_columns()
    models = {name: load_model(name) for name in MODEL_METRICS}
    return scaler, features, models

scaler, feature_columns, models = get_assets()

# ── Page ───────────────────────────────────────────────────────────────────
st.title(" Prédiction du Risque d'Abandon")
st.markdown("Renseignez le profil d'un étudiant pour évaluer son risque d'abandon scolaire.")

tab1, tab2 = st.tabs([" Prédiction individuelle", " Prédiction par lot (CSV)"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Prédiction individuelle
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Profil de l'étudiant")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        with st.container(border=True):
            st.markdown("** Informations académiques**")
            average_grade = st.slider(
                "Moyenne générale (/20)", 0.0, 20.0, 12.0, 0.1,
                help="Moyenne sur 20 de l'étudiant"
            )
            absenteeism_rate = st.slider(
                "Taux d'absentéisme", 0.0, 0.5, 0.10, 0.01,
                format="%.0f%%",
                help="Proportion de cours manqués (0 = jamais absent, 0.5 = 50% absent)"
            )
            study_time_hours = st.slider(
                "Temps d'étude journalier (heures)", 0.0, 5.0, 2.5, 0.1
            )

        with st.container(border=True):
            st.markdown("** Informations personnelles**")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Âge", 15, 30, 19)
                gender = st.selectbox("Genre", ["Male", "Female"])
            with c2:
                internet_access = st.selectbox("Accès Internet", ["Yes", "No"])
                extra_activities = st.selectbox("Activités extrascolaires", ["Yes", "No"])

        selected_model = st.selectbox(
            " Modèle de prédiction",
            list(MODEL_METRICS.keys()),
            index=0,
        )

        predict_btn = st.button(" Lancer la prédiction", type="primary", use_container_width=True)

    # ── Résultats ─────────────────────────────────────────────────────────
    with col_result:
        if predict_btn:
            input_df = build_input_df(
                age, gender, average_grade, absenteeism_rate,
                internet_access, study_time_hours, extra_activities,
                feature_columns,
            )
            model = models[selected_model]
            pred, proba = predict(model, scaler, input_df)
            level, color, emoji = risk_level(proba)

            # ── Gauge ──────────────────────────────────────────────────
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 42, "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 30],  "color": "#d4efdf"},
                        {"range": [30, 60], "color": "#fdebd0"},
                        {"range": [60, 100],"color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": proba * 100,
                    },
                },
                title={"text": f"Probabilité de risque<br><b>{emoji} Niveau : {level}</b>",
                       "font": {"size": 16}},
            ))
            fig.update_layout(height=280, margin=dict(t=60, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            # ── Verdict ────────────────────────────────────────────────
            if pred == 1:
                st.error(f"### {emoji} Risque d'abandon détecté !")
                st.markdown(f"""
                **Niveau de risque : {level}** ({proba:.1%})

                Cet étudiant présente un profil à risque. Des mesures préventives sont recommandées.
                """)
            else:
                st.success(f"### {emoji} Pas de risque majeur détecté")
                st.markdown(f"""
                **Niveau de risque : {level}** ({proba:.1%})

                L'étudiant présente un profil stable. Continuez le suivi régulier.
                """)

            # ── Score global calculé ───────────────────────────────────
            presence = 1 - absenteeism_rate
            global_sc = (average_grade/20)*0.5 + presence*0.3 + (study_time_hours/5)*0.2

            st.markdown("---")
            st.markdown("** Indicateurs calculés**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Taux de présence", f"{presence:.0%}")
            m2.metric("Score global", f"{global_sc:.3f}")
            m3.metric("Modèle utilisé", selected_model.split()[0])

            # ── Recommandations ────────────────────────────────────────
            st.markdown("---")
            st.markdown("** Recommandations**")
            recs = []
            if average_grade < 10:
                recs.append(" Mettre en place un soutien scolaire personnalisé")
            if absenteeism_rate > 0.30:
                recs.append(" Investiguer les raisons de l'absentéisme élevé")
            if study_time_hours < 1:
                recs.append(" Encourager un meilleur temps de travail personnel")
            if internet_access == "No":
                recs.append(" Faciliter l'accès aux ressources numériques")
            if not recs:
                recs.append(" Profil satisfaisant — maintenir le suivi habituel")
            for r in recs:
                st.markdown(f"- {r}")
        else:
            st.info(" Renseignez le profil de l'étudiant et cliquez sur **Lancer la prédiction**.")
            st.markdown("""
            <div style="background:#f0f4f8; border-radius:12px; padding:24px; margin-top:16px; text-align:center;">
                <div style="font-size:4rem;"></div>
                <div style="color:#6c757d; margin-top:8px;">Le résultat s'affichera ici</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Prédiction par lot
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Prédiction sur un fichier CSV")
    st.markdown("""
    Importez un fichier CSV contenant plusieurs étudiants.
    Le fichier doit avoir les mêmes colonnes que le dataset d'origine :
    `age, gender, average_grade, absenteeism_rate, internet_access, study_time_hours, extra_activities`
    """)

    uploaded = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        required = ["age","gender","average_grade","absenteeism_rate",
                    "internet_access","study_time_hours","extra_activities"]
        missing_cols = [c for c in required if c not in df_batch.columns]

        if missing_cols:
            st.error(f"Colonnes manquantes : {missing_cols}")
        else:
            st.success(f" {len(df_batch)} étudiants chargés")
            batch_model = st.selectbox("Modèle", list(MODEL_METRICS.keys()), key="batch_model")
            model = models[batch_model]

            if st.button(" Lancer la prédiction sur tout le fichier", type="primary"):
                results = []
                for _, row in df_batch.iterrows():
                    inp = build_input_df(
                        row["age"], row["gender"], row["average_grade"],
                        row["absenteeism_rate"], row["internet_access"],
                        row["study_time_hours"], row["extra_activities"],
                        feature_columns,
                    )
                    pred, proba = predict(model, scaler, inp)
                    level, _, emoji = risk_level(proba)
                    results.append({
                        "Prédiction": " Risque" if pred == 1 else " Stable",
                        "Probabilité": f"{proba:.1%}",
                        "Niveau": f"{emoji} {level}",
                    })

                df_result = pd.concat([df_batch.reset_index(drop=True),
                                       pd.DataFrame(results)], axis=1)

                at_risk = sum(1 for r in results if "Risque" in r["Prédiction"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Total étudiants", len(df_batch))
                c2.metric("À risque", at_risk, delta=f"{at_risk/len(df_batch):.0%}", delta_color="inverse")
                c3.metric("Stables", len(df_batch) - at_risk)

                st.dataframe(df_result, use_container_width=True)

                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Télécharger les résultats",
                    data=csv_out,
                    file_name="predictions_abandon.csv",
                    mime="text/csv",
                )
    else:
        # Template download
        template = pd.DataFrame([{
            "age": 19, "gender": "Male", "average_grade": 12.5,
            "absenteeism_rate": 0.10, "internet_access": "Yes",
            "study_time_hours": 2.5, "extra_activities": "No",
        }])
        st.download_button(
            "⬇️ Télécharger un fichier template CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="template_etudiants.csv",
            mime="text/csv",
        )
