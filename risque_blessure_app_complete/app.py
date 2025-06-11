
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from fpdf import FPDF
import base64
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# === Modèle entraîné à chaque exécution ===
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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# === Interface ===
st.set_page_config(page_title="Risque de Blessure", layout="wide")
st.title("🧠 Estimation du Risque de Blessure avec Rapport PDF & Radar")

with st.form("formulaire"):
    col1, col2, col3 = st.columns(3)
    with col1:
        dorsiflexion = st.slider("Dorsiflexion (°)", 0, 40, 20)
        adducteur_strength = st.slider("Force adducteurs (kg)", 0, 100, 50)
        vertical_jump = st.slider("Saut vertical (cm)", 10, 100, 50)
    with col2:
        sprint_time = st.number_input("Temps sprint 10m (s)", 1.0, 10.0, 2.5)
        squat_1RM = st.slider("Squat 1RM (kg)", 0, 200, 100)
        charge_var = st.slider("Variation charge (%)", 0, 100, 20)
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
    st.markdown(f"### ✅ Risque estimé : {niveau[prediction]}")

    # === Radar chart ===
    radar_labels = ["Dorsiflexion", "Adducteurs", "Sprint", "Squat", "Saut", "Charge", "Fatigue", "Sommeil"]
    radar_values = [dorsiflexion, adducteur_strength, sprint_time * 10, squat_1RM, vertical_jump, charge_var, fatigue * 20, sommeil * 20]
    angles = [n / float(len(radar_labels)) * 2 * pi for n in range(len(radar_labels))]
    radar_values += radar_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, radar_values, linewidth=2, linestyle='solid')
    ax.fill(angles, radar_values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)
    radar_path = "/tmp/radar_plot.png"
    fig.savefig(radar_path)
    st.pyplot(fig)

    # === PDF generation ===
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Rapport de Risque de Blessure", ln=1, align="C")
    pdf.ln(10)
    champs = [
        ("Dorsiflexion (°)", dorsiflexion),
        ("Force adducteurs (kg)", adducteur_strength),
        ("Temps sprint (s)", sprint_time),
        ("Squat 1RM (kg)", squat_1RM),
        ("Saut vertical (cm)", vertical_jump),
        ("Variation charge (%)", charge_var),
        ("Fatigue (1-5)", fatigue),
        ("Sommeil (1-5)", sommeil),
        ("Blessure antérieure", "Oui" if blessure_bin else "Non"),
        ("Risque estimé", niveau[prediction])
    ]
    for label, val in champs:
        pdf.cell(200, 10, txt=f"{label} : {val}", ln=1)

    pdf.image(radar_path, x=30, y=pdf.get_y(), w=150)
    pdf_path = "/tmp/rapport_risque.pdf"
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="rapport_risque.pdf">📄 Télécharger le rapport PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
