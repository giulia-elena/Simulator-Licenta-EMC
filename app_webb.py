import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import gdown

# Import pentru ANN
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

# ==========================================
# 1. CLASA SAFETY-FIRST
# ==========================================
class SafetyFirstPredictor:
    def __init__(self, model_baza, safety_factor=3.0):
        self.model = model_baza
        self.safety_factor = safety_factor
        self.mae_istoric = 0.001673

    def predict_safe(self, X):
        pred_base = self.model.predict(X)
        if hasattr(pred_base, 'flatten'): pred_base = pred_base.flatten()
        bias = self.safety_factor * self.mae_istoric
        val_base = max(0, float(pred_base[0]))
        pred_safe = val_base + bias
        return val_base, [pred_safe]

# ==========================================
# 2. BAZA DE DATE (MAPPING)
# ==========================================
mapping = {
    "dinamica": {
        "interior": {
            "Pct A": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Interior", "dist": ["0.9", "1.5"]},
            "Pct B": {"sursa": "Boghiu motor", "zona": "655-5-A Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct C": {"sursa": "Aer condiționat călători", "zona": "655-5-A Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct D": {"sursa": "Boghiu motor", "zona": "655-5-A Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct E": {"sursa": "Transformator auxiliar", "zona": "655-5-C Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct F": {"sursa": "Aer condiționat călători", "zona": "655-5-C Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct G": {"sursa": "Transformator auxiliar", "zona": "655-5-C Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct H": {"sursa": "Pantograf", "zona": "655-5-B Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct I": {"sursa": "Transformator de tracțiune", "zona": "655-5-B Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct J": {"sursa": "Aer condiționat călători", "zona": "655-5-B Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct K": {"sursa": "Pantograf", "zona": "655-5-B Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct L": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Interior", "dist": ["0.9", "1.5"]},
            "Ambiental": {"sursa": "Fundal", "zona": "Ambiental Interior", "dist": ["0.0"]}
        }
    },
    "statica": {
        "interior": {
            "Pct A": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Interior", "dist": ["0.9", "1.5"]},
            "Pct B": {"sursa": "Boghiu motor", "zona": "655-5-A Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct C_F_J": {"sursa": "Aer condiționat călători", "zona": "Zone 655-5-A+B+C Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct D": {"sursa": "Boghiu motor", "zona": "655-5-A Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct E_G": {"sursa": "Transformator auxiliar", "zona": "Zone 655-5-C Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct H_K": {"sursa": "Pantograf", "zona": "655-5-B Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct I": {"sursa": "Transformator de tracțiune", "zona": "655-5-B Interior", "dist": ["0.3", "0.9", "1.5"]},
            "Pct L": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Interior", "dist": ["0.9", "1.5"]},
            "Ambiental": {"sursa": "Fundal", "zona": "Ambiental Interior", "dist": ["0.0"]}
        },
        "exterior": {
            "Pct 1": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 2": {"sursa": "Boghiu motor stânga/dreapta", "zona": "655-5-A Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 3": {"sursa": "Boghiu motor stânga/dreapta", "zona": "655-5-A Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 4": {"sursa": "Aer condiționat călători stânga/dreapta", "zona": "655-5-A Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 5": {"sursa": "Aer condiționat călători stânga/dreapta", "zona": "655-5-A Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 6": {"sursa": "Boghiu motor stânga/dreapta", "zona": "655-5-A Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 7": {"sursa": "Boghiu motor stânga/dreapta", "zona": "655-5-A Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 8": {"sursa": "Transformator auxiliar stânga/dreapta", "zona": "655-5-C Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 9": {"sursa": "Transformator auxiliar stânga/dreapta", "zona": "655-5-C Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 10_18": {"sursa": "Aer condiționat călători stânga/dreapta", "zona": "Zone 655-5-B+C Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 11": {"sursa": "Aer condiționat călători stânga/dreapta", "zona": "655-5-C Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 12": {"sursa": "Transformator auxiliar stânga/dreapta", "zona": "655-5-C Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 13": {"sursa": "Transformator auxiliar stânga/dreapta", "zona": "655-5-C Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 14_21": {"sursa": "Pantograf stânga/dreapta", "zona": "655-5-B Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 15": {"sursa": "Fundal", "zona": "655-5-B Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 16": {"sursa": "Transformator tracțiune stânga/dreapta", "zona": "655-5-B Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 17": {"sursa": "Transformator tracțiune stânga/dreapta", "zona": "655-5-B Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 19": {"sursa": "Aer condiționat călători stânga/dreapta", "zona": "655-5-B Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 20": {"sursa": "Pantograf stânga/dreapta", "zona": "655-5-B Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Pct 22": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Exterior", "dist": ["0.5", "1.5", "2.5"]},
            "Ambiental": {"sursa": "Fundal", "zona": "Ambiental Exterior", "dist": ["0.0"]}
        }
    }
}

# ==========================================
# 3. FUNCȚII UTILITARE
# ==========================================
@st.cache_resource
def incarca_modele():
    path = '.'
    cale_rf = os.path.join(path, 'model_RF.joblib')
    cale_hybrid = os.path.join(path, 'model_algoritm_custom_Safety_First.joblib')
    cale_ann = os.path.join(path, 'model_ann_keras_drive.h5')
    
    if not os.path.exists(cale_rf):
        id_rf = '1kGxhzD5qPhQOid2JLqwQkUQ8Whryhh3S'
        gdown.download(f'https://drive.google.com/uc?id={id_rf}', cale_rf, quiet=False)

    if not os.path.exists(cale_hybrid):
        id_hybrid = '1ULwgBFyQmYrFjSlFKiY-Q9sB9nzp0W3b'
        gdown.download(f'https://drive.google.com/uc?id={id_hybrid}', cale_hybrid, quiet=False)

    if not os.path.exists(cale_ann):
        id_ann = '1kOasTirnVhrfYcxoHlHbFIAJ0xd_JiE9'
        gdown.download(f'https://drive.google.com/uc?id={id_ann}', cale_ann, quiet=False)

    encoders = joblib.load(os.path.join(path, 'encoders.joblib'))
    scaler = joblib.load(os.path.join(path, 'scaler_ann_keras.joblib'))
    rf_model = joblib.load(cale_rf)
    ann_model = load_model(cale_ann, compile=False)
    hybrid_model = joblib.load(cale_hybrid)
    
    return encoders, scaler, rf_model, ann_model, hybrid_model

def calculeaza_limite_exacte(f):
    icnirp = 40000 / (f ** 2) if f < 8 else 5000 / f if f < 800 else 6.25
    eu_dir = 200000 / (f ** 2) if f < 8 else 25000 / f if f < 25 else 1000 if f < 300 else 300000 / f if f < 3000 else 100
    return icnirp, eu_dir

def deseneaza_grafic(predictie, lim_icnirp, lim_eu, culoare_bar):
    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
    fig.patch.set_alpha(0.0) 
    ax.set_facecolor((0,0,0,0))

    bars = ax.barh([''], [predictie], color=culoare_bar, height=0.3)

    ax.axvline(x=lim_icnirp, color='#f59e0b', linestyle='--', linewidth=1.5, label=f'ICNIRP (Public): {lim_icnirp:.2f} µT')
    ax.axvline(x=lim_eu, color='#ef4444', linestyle='-.', linewidth=1.5, label=f'Dir. 2013/35 (Prof): {lim_eu:.2f} µT')

    ax.set_xscale('log')
    ax.set_xlabel('Inducție Magnetică B [µT] (Scară Logaritmică)', fontsize=10, color='gray', labelpad=10)

    ax.grid(axis='x', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('gray')
    ax.tick_params(axis='x', colors='gray', labelsize=9)
    ax.tick_params(axis='y', left=False)

    ax.legend(loc='lower right', fontsize=9, framealpha=0.8)
    plt.tight_layout()
    return fig

# ==========================================
# 4. INTERFAȚA WEB (Streamlit)
# ==========================================
st.set_page_config(page_title="Simulator EMC", layout="wide", page_icon="🚄")
encoders, scaler, rf_model, ann_model, hybrid_model = incarca_modele()

# --- BARA LATERALĂ ---
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>PANOU DE CONTROL</h2>", unsafe_allow_html=True)

# "Card" centrat pentru informațiile trenului
st.sidebar.markdown(
    """
    <div style='text-align: center; background-color: #1e3a8a15; padding: 15px; border-radius: 10px; border: 1px solid #3b82f640; margin-bottom: 25px;'>
        <div style='font-size: 45px;'>🚄</div>
        <div style='font-weight: bold; font-size: 16px; color: #3b82f6; margin-top: 5px;'>RAMĂ ELECTRICĂ</div>
        <div style='font-size: 12px; color: #60a5fa; margin-top: 2px;'>25 kV | 160 km/h | 4x105 kW</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Cascadă de filtre
regim = st.sidebar.selectbox("1. Regim de funcționare:", ["Dinamic", "Static"])
rk_label = "dinamica" if regim == "Dinamic" else "statica"

locatii_disponibile = ["Interior"] if regim == "Dinamic" else ["Interior", "Exterior"]
locatie = st.sidebar.selectbox("2. Locația măsurătorii:", locatii_disponibile)
lk_label = locatie.lower()

pm = mapping[rk_label][lk_label]
puncte_formatate = [f"{pid} | {info['zona']} | {info['sursa']}" for pid, info in pm.items()]
punct_text = st.sidebar.selectbox("3. Punct (ID | Zonă | Sursă):", puncte_formatate)

pid = punct_text.split(" | ")[0]
info = pm[pid]
distanta_selectata = st.sidebar.selectbox("4. Distanța față de sursă (m):", info['dist'])
distanta = float(distanta_selectata)

frecventa = st.sidebar.number_input("5. Frecvența de lucru (Hz):", min_value=4.5, max_value=20025.0, value=50.0, step=1.0)
st.sidebar.caption("Interval permis: 4.5 - 20025 Hz")

demo_mode = st.sidebar.selectbox("MOD DEMO (TESTARE ALERTE)", ["Normal (Real)", "Atenție (Portocaliu)", "Pericol (Roșu)"])

st.sidebar.markdown("**6. Selectați Modelele AI:**")
use_rf = st.sidebar.checkbox("Random Forest (Stabil | R²=0.9452)", value=True)
use_ann = st.sidebar.checkbox("Neural Network (Extrapolare | R²=0.8942)", value=True)
use_hybrid = st.sidebar.checkbox("Safety-First (RF + Marjă 3xMAE)", value=True)

btn_analiza = st.sidebar.button("ANALIZEAZĂ CONFORMITATEA", type="primary", use_container_width=True)

# --- ZONA PRINCIPALĂ ---
st.title("Sistem de Analiză Multi-Model și Conformitate Legală EMC")

if btn_analiza:
    if not (use_rf or use_ann or use_hybrid):
        st.error("EROARE: Selectați cel puțin un model AI pentru analiză!")
    else:
        # LOGICA DE CALCUL
        lim_i, lim_e = calculeaza_limite_exacte(frecventa)
        predictii = {}

        if pid == "Ambiental":
            val_base = 0.00045
            if use_rf: predictii["Random Forest"] = val_base
            if use_ann: predictii["Neural Network"] = val_base
            if use_hybrid: predictii["Safety-First Hybrid"] = val_base + 0.005019
        else:
            s_enc = encoders['sursa'].transform([info['sursa']])[0]
            z_enc = encoders['zona'].transform([info['zona']])[0]
            l_enc = encoders['locatie'].transform([lk_label])[0]
            t_enc = encoders['tip'].transform([rk_label])[0]

            X = pd.DataFrame([[frecventa, distanta, s_enc, z_enc, l_enc, t_enc]],
                             columns=['Frecventa', 'Distanta', 'Sursa_Encoded', 'Zona_Encoded', 'Locatie_Encoded', 'Tip_Encoded'])
            X_scaled = scaler.transform(X.values)

            p_rf = max(0, float(rf_model.predict(X)[0]))
            p_ann = max(0, float(ann_model.predict(X_scaled, verbose=0).flatten()[0]))
            _, p_hy_list = hybrid_model.predict_safe(X)
            p_hy = max(0, float(p_hy_list[0]))

            factor = 1.0
            if "Portocaliu" in demo_mode:
                factor = (lim_i * 1.5) / max(p_hy, 0.001)
            elif "Roșu" in demo_mode:
                factor = (lim_e * 1.1) / max(p_hy, 0.001)

            if use_rf: predictii["Random Forest"] = p_rf * factor
            if use_ann: predictii["Neural Network"] = p_ann * factor
            if use_hybrid: predictii["Safety-First Hybrid"] = p_hy * factor

        # Extragere Rezultate
        model_critic = max(predictii, key=predictii.get)
        val_max = predictii[model_critic]
        procent_limita = (val_max / lim_i) * 100
        procent_bara = min(procent_limita, 100) / 100.0

        # Determinare Status, Culori și Iconițe dinamice
        if val_max >= lim_e:
            culoare_hex = "#ef4444" # Roșu
            status_txt = "PERICOL (Depășire limită profesională)"
            iconita_mare = "☢️"
        elif val_max >= lim_i:
            culoare_hex = "#f59e0b" # Portocaliu
            status_txt = "ATENȚIE (Depășire limită public)"
            iconita_mare = "⚠️"
        else:
            culoare_hex = "#10b981" # Verde
            status_txt = "CONFORM (Zonă sigură)"
            iconita_mare = "🛡️"

        # ---------------------------------------------------------
        # AFIȘARE DASHBOARD
        # ---------------------------------------------------------
        # 1. Cardul Principal
        st.markdown("### 🛡️ PREDICȚIE MAXIMĂ DE EXPUNERE")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"<h1 style='color: {culoare_hex}; margin-bottom: 0px;'>{val_max:.5f} µT</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='padding: 20px; background-color: {culoare_hex}20; border-radius: 10px; color: {culoare_hex}; font-weight: bold;'>STATUS: {status_txt}</div>", unsafe_allow_html=True)
        
        st.divider()

        # 2. Graficul Vizual
        st.markdown("### 📊 ANALIZĂ VIZUALĂ A CONFORMITĂȚII")
        fig = deseneaza_grafic(val_max, lim_i, lim_e, culoare_hex)
        st.pyplot(fig, transparent=True)
        
        st.divider()

        # 3. Secțiunea de Jos: Raport și Bara de Progres
        col_raport, col_indicator = st.columns([2, 1])
        
        with col_raport:
            st.markdown("### 📝 JURNAL DIAGNOZĂ (Date Brute)")
            raport = f"1. CONFIGURAȚIE SCENARIU:\n"
            raport += f"   > Regim funcționare:     {regim.upper()}\n"
            raport += f"   > Amplasament:           {lk_label.upper()}\n"
            raport += f"   > Punct analizat:        {pid} ({info['zona']})\n"
            raport += f"   > Sursă emisie:          {info['sursa']}\n"
            raport += f"   > Parametri fizici:      {distanta} m distanță | {frecventa} Hz frecvență\n\n"
            raport += "2. ANALIZĂ PREDICTIVĂ MODELE AI:\n"
            for m, v in predictii.items():
                maxim_tag = " [VALOARE MAXIMĂ PREZISĂ]" if m == model_critic else ""
                raport += f"   > {m.ljust(22)} {v:.6f} µT{maxim_tag}\n"
            raport += "\n3. EVALUARE CONFORMITATE LEGALĂ:\n"
            raport += f"   > Limită ICNIRP (Public): {lim_i:.2f} µT\n"
            raport += f"   > Limită Dir. 2013/35:    {lim_e:.2f} µT\n"
            raport += f"   > Grad ocupare ICNIRP:    {procent_limita:.4f}%\n"
            
            st.code(raport, language="text")
            
            # Buton util pentru descărcarea jurnalului (impresionează mereu la prezentări)
            st.download_button(
                label="📥 Descarcă Jurnalul (TXT)",
                data=raport,
                file_name="Raport_Diagnoza_EMC.txt",
                mime="text/plain"
            )

        with col_indicator:
            st.markdown("### 🎛️ INDICATOR SIGURANȚĂ")
            
            # Iconița Uriașă Centrată
            st.markdown(f"<div style='text-align: center; font-size: 110px; margin-top: 10px; margin-bottom: 20px;'>{iconita_mare}</div>", unsafe_allow_html=True)
            
            # Bara de Progres
            st.progress(procent_bara)
            st.markdown(f"<p style='text-align: center; font-weight: bold; color: {culoare_hex};'>{procent_limita:.2f}% din limita publică</p>", unsafe_allow_html=True)

else:
    # Mesaj de întâmpinare subtil înainte de a apăsa butonul
    st.info("Alegeți parametrii din meniul lateral și apăsați 'ANALIZEAZĂ CONFORMITATEA' pentru a începe.")
