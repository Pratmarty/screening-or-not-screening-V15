
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# ========= V14 : Modèle entraîné à chaque exécution =========

# Données d'entraînement fictives
df = pd.DataFrame({
    "dorsiflexion": [12, 20, 30, 10, 14, 25, 18, 16, 22, 19],
    "adducteur_strength": [35, 50, 60, 30, 42, 70, 65, 38, 55, 40],
    "sprint_time": [4.0, 2.8, 2.5, 4.2, 3.9, 2.7, 2.6, 3.8, 2.9, 3.5],
    "squat_1RM": [60, 90, 120, 50, 70, 130, 100, 80, 110, 85],
    "vertical_jump": [35, 60, 70, 30, 40, 75, 65, 45, 68, 50],
    "charge_var": [50, 20, 15, 60, 45, 10, 18, 55, 25, 35],
    "fatigue": [5, 2, 1, 4, 5, 1, 2, 4, 3, 3],
    "sommeil": [1, 4, 5, 2, 1, 5, 4, 2, 3, 3],
    "historique_blessure": [1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
    "risque": [2, 0, 0, 2, 2, 0, 0, 1, 0, 1]
})

X = df.drop(columns="risque")
y = df["risque"]

# Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# === Interface Streamlit ===
st.set_page_config(page_title="Risque de Blessure V14.0", layout="wide")
st.title("🧠 Prédiction du Risque de Blessure - Version 14.0")

st.markdown("Remplis le formulaire ci-dessous pour estimer ton risque de blessure.")

with st.form("formulaire"):
    col1, col2, col3 = st.columns(3)

    with col1:
        dorsiflexion = st.slider("Amplitude dorsiflexion (en °)", 0, 40, 20)
        adducteur_strength = st.slider("Force adducteurs (kg)", 0, 100, 50)
        vertical_jump = st.slider("Saut vertical (cm)", 10, 100, 50)

    with col2:
        sprint_time = st.number_input("Temps sprint 10m (s)", 1.0, 10.0, 2.5)
        squat_1RM = st.slider("Squat 1RM (kg)", 0, 200, 100)
        charge_var = st.slider("Variation charge entraînement (%)", 0, 100, 20)

    with col3:
        fatigue = st.slider("Fatigue (1 à 5)", 1, 5, 3)
        sommeil = st.slider("Qualité du sommeil (1 à 5)", 1, 5, 3)
        historique_blessure = st.selectbox("Blessure antérieure ?", ["Non", "Oui"])

    submit = st.form_submit_button("Analyser")

if submit:
    blessure_bin = 1 if historique_blessure == "Oui" else 0

    features = np.array([
        dorsiflexion,
        adducteur_strength,
        sprint_time,
        squat_1RM,
        vertical_jump,
        charge_var,
        fatigue,
        sommeil,
        blessure_bin
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]

    niveau = ["🟢 Faible", "🟠 Modéré", "🔴 Élevé"]
    st.subheader("Résultat de l'analyse")
    st.markdown(f"### Risque estimé : {niveau[prediction]}")
