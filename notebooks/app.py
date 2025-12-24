import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📱 Prediksi Churn Pelanggan Telco")
st.write("Aplikasi untuk memprediksi apakah pelanggan akan berhenti berlangganan")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    try:
        if os.path.exists('best_churn_model.pkl'):
            model = joblib.load('best_churn_model.pkl')
            return model
        else:
            st.error("File model tidak ditemukan. Jalankan notebook 04 terlebih dahulu.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    st.sidebar.success("✅ Model siap digunakan")

# ==================== INPUT FORM ====================
st.sidebar.header("Input Data")

# Simple form
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (bulan)", 0, 72, 24)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, tenure * monthly_charges)

# Default values for other features
multiple_lines = "No"
online_security = "No"
online_backup = "No"
device_protection = "No"
tech_support = "No"
streaming_tv = "No"
streaming_movies = "No"

# Create input dataframe
input_data = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_data])

# ==================== DISPLAY INPUT ====================
st.subheader("Data Input")
st.write(input_df)

# ==================== PREDICTION ====================
if st.button("Prediksi Churn"):
    if model:
        try:
            # Predict
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
            # Display results
            st.subheader("Hasil Prediksi")
            
            if prediction == 1:
                st.error("## 🔴 CHURN: YA")
                st.write("Pelanggan berpotensi berhenti berlangganan")
            else:
                st.success("## 🟢 CHURN: TIDAK")
                st.write("Pelanggan cenderung tetap berlangganan")
            
            # Probabilities - SIMPLE VERSION
            st.subheader("Probabilitas")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tidak Churn", f"{probabilities[0]*100:.1f}%")
            with col2:
                st.metric("Churn", f"{probabilities[1]*100:.1f}%")
            
            # Simple bar chart WITHOUT color parameter
            prob_df = pd.DataFrame({
                'Status': ['Tidak Churn', 'Churn'],
                'Probabilitas': [probabilities[0]*100, probabilities[1]*100]
            })
            
            st.bar_chart(prob_df.set_index('Status'))
            
            # Recommendations
            st.subheader("Rekomendasi")
            if prediction == 1:
                st.warning("""
                **Tindakan yang disarankan:**
                1. Hubungi pelanggan untuk feedback
                2. Tawarkan diskon khusus
                3. Perbaiki layanan yang dikeluhkan
                """)
            else:
                st.info("""
                **Strategi retensi:**
                1. Program loyalitas
                2. Tawaran eksklusif
                3. Komunikasi rutin
                """)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Model belum dimuat. Pastikan file model ada.")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Proyek UAS Bengkel Koding Data Science - Universitas Dian Nuswantoro")