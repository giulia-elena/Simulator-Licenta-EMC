import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import gdown

# Import pentru ANN
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

# 1. Clasa pentru modelul Hybrid Safety-First
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

# 2. Funcție pentru încărcarea și descărcarea modelelor din Google Drive
@st.cache_resource
def incarca_modele():
    path = '.'
    
    # Ne asigurăm că folderul există pe serverul de cloud
    if not os.path.exists(path):
        os.makedirs(path)
        
    cale_rf = os.path.join(path, 'model_RF.joblib')
    cale_hybrid = os.path.join(path, 'model_algoritm_custom_Safety_First.joblib')
    
    # Descărcăm Random Forest dacă lipsește
    if not os.path.exists(cale_rf):
        st.info("⏳ Se descarcă modelul Random Forest (Baza) din cloud... (Acest proces are loc doar la prima rulare)")
        id_rf = '1kGxhzD5qPhQOid2JLqwQkUQ8Whryhh3S'
        gdown.download(f'https://drive.google.com/uc?id={id_rf}', cale_rf, quiet=False)

    # Descărcăm Safety First dacă lipsește
    if not os.path.exists(cale_hybrid):
        st.info("⏳ Se descarcă modelul Safety-First (Hibrid) din cloud... (Acest proces are loc doar la prima rulare)")
        id_hybrid = '1ULwgBFyQmYrFjSlFKiY-Q9sB9nzp0W3b'
        gdown.download(f'https://drive.google.com/uc?id={id_hybrid}', cale_hybrid, quiet=False)

    # Încărcarea efectivă a fișierelor în memorie
    encoders = joblib.load(os.path.join(path, 'encoders.joblib'))
    scaler = joblib.load(os.path.join(path, 'scaler_ann_keras.joblib'))
    rf_model = joblib.load(cale_rf)
    ann_model = load_model(os.path.join(path, 'model_ann_keras.h5'), compile=False)
    hybrid_model = joblib.load(cale_hybrid)
    
    return encoders, scaler, rf_model, ann_model, hybrid_model

# Inițializăm modelele o singură dată
encoders, scaler, rf_model, ann_model, hybrid_model = incarca_modele()

# 3. Configurarea Paginii Web
st.set_page_config(page_title="Simulator EMC", layout="wide")
st.title("Sistem de Analiză Multi-Model și Conformitate Legală EMC")

# --- BARA LATERALĂ (Sidebar) ---
st.sidebar.header("PANOU DE CONTROL")

# Selectoare
regim = st.sidebar.selectbox("1. Regim de funcționare:", ["Dinamic", "Static"])
locatie = st.sidebar.selectbox("2. Locația măsurătorii:", ["Interior", "Exterior"])

punct = st.sidebar.text_input("3. Punct (ID | Zonă | Sursă):", "Pct A | 655-5-A Interior | Aparat...")
distanta = st.sidebar.number_input("4. Distanța față de sursă (m):", min_value=0.0, value=1.0)
frecventa = st.sidebar.number_input("5. Frecvența de lucru (Hz):", min_value=4.5, max_value=20025.0, value=50.0)

st.sidebar.subheader("6. Selectați Modelele AI:")
use_rf = st.sidebar.checkbox("Random Forest (Stabil | R²=0.9452)", value=True)
use_ann = st.sidebar.checkbox("Neural Network (Extrapolare | R²=0.8942)", value=True)
use_hybrid = st.sidebar.checkbox("Safety-First (RF + Marjă 3xMAE)", value=True)

# --- ZONA PRINCIPALĂ ---
if st.sidebar.button("ANALIZEAZĂ CONFORMITATEA", type="primary"):
    st.subheader("🛡️ PREDICȚIE MAXIMĂ DE EXPUNERE")

    # Acestea sunt valori demonstrative, logica completă urmează a fi integrată
    val_max = 12.5
    lim_icnirp = 100.0

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Nivel maxim estimat", value=f"{val_max} µT")
    with col2:
        if val_max < lim_icnirp:
            st.success("STATUS: CONFORM (Zonă sigură)")
        else:
            st.error("STATUS: PERICOL")
