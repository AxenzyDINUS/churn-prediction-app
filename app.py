# app.py - UPDATED VERSION WITH LATEST PACKAGES
import sys
import subprocess
import importlib
import os

# ==================== AUTO-INSTALL MECHANISM ====================
def ensure_packages():
    """Ensure all required packages are installed"""
    required = [
        ("streamlit", "1.32.0"),
        ("scikit-learn", "1.5.0"),
        ("numpy", "2.0.2"),
        ("pandas", "2.2.2"),
        ("joblib", "1.4.2")
    ]
    
    for package, version in required:
        try:
            if package == "scikit-learn":
                importlib.import_module("sklearn")
            else:
                importlib.import_module(package)
        except ImportError:
            print(f"📦 Installing {package}=={version}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                f"{package}=={version}"
            ])

# Run auto-install (only in Streamlit Cloud environment)
if "streamlit" not in sys.modules:
    ensure_packages()

# ==================== IMPORTS SETELAH PASTI TERINSTALL ====================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Debug info - tampilkan versi
import sklearn
st.sidebar.write(f"**Python:** {sys.version[:6]}")
st.sidebar.write(f"**scikit-learn:** {sklearn.__version__}")
st.sidebar.write(f"**numpy:** {np.__version__}")
st.sidebar.write(f"**pandas:** {pd.__version__}")

# ==================== KODE APP ANDA YANG ASLI (DIMODIFIKASI) ====================

# Set page config
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📱 Prediksi Churn Pelanggan Telco")
st.write("Aplikasi untuk memprediksi apakah pelanggan akan berhenti berlangganan")

# ==================== LOAD MODEL DENGAN MULTI-PATH ====================
@st.cache_resource
def load_model_and_preprocessor():
    """
    Coba load model dari berbagai lokasi yang mungkin
    """
    # Cari model dengan ekstensi .joblib atau .pkl
    model_paths = [
        'model.joblib',                    # Nama baru setelah re-train
        'best_churn_model.joblib',
        'best_churn_model.pkl',
        'notebooks/model.joblib',
        './model.joblib',
        '../model.joblib'
    ]
    
    # Cari preprocessor/feature names
    preprocessor_paths = [
        'preprocessor.joblib',
        'preprocessor.pkl',
        'feature_names.joblib',
        'feature_names.pkl',
        'notebooks/preprocessor.joblib'
    ]
    
    model = None
    preprocessor = None
    model_location = ""
    preprocessor_location = ""
    
    # Cari model
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Coba load dengan joblib (format baru)
                model = joblib.load(path)
                model_location = path
                st.sidebar.success(f"✅ Model loaded from: {path}")
                break
            except Exception as e:
                st.sidebar.warning(f"Gagal load dari {path}: {str(e)[:50]}")
                # Coba dengan pickle sebagai fallback
                try:
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                    model_location = f"{path} (pickle)"
                    st.sidebar.success(f"✅ Model loaded with pickle from: {path}")
                    break
                except:
                    continue
    
    # Cari preprocessor
    for path in preprocessor_paths:
        if os.path.exists(path):
            try:
                preprocessor = joblib.load(path)
                preprocessor_location = path
                st.sidebar.info(f"Preprocessor dari: {path}")
                break
            except:
                try:
                    with open(path, 'rb') as f:
                        preprocessor = pickle.load(f)
                    preprocessor_location = f"{path} (pickle)"
                    st.sidebar.info(f"Preprocessor dari: {path}")
                    break
                except:
                    continue
    
    # Jika tidak ditemukan
    if not model:
        st.warning("""
        ⚠️ **Model tidak ditemukan!** 
        
        Pastikan Anda sudah:
        1. Melatih ulang model dengan `retrain_latest.py`
        2. File model disimpan sebagai `model.joblib`
        3. File sudah di-push ke GitHub
        
        Jika model belum ada, aplikasi akan berjalan dalam mode demo.
        """)
    
    return model, preprocessor, model_location, preprocessor_location

# Load model dan preprocessor
model, preprocessor, model_loc, preproc_loc = load_model_and_preprocessor()

if model:
    st.sidebar.success(f"✅ Model siap digunakan")
    # Info tambahan tentang model
    try:
        if hasattr(model, 'named_steps'):
            st.sidebar.info(f"Pipeline dengan {len(model.named_steps)} steps")
        elif hasattr(model, 'predict'):
            st.sidebar.info(f"Model: {type(model).__name__}")
    except:
        pass

# ==================== INPUT FORM ====================
st.sidebar.header("📝 Input Data")

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
multiple_lines = "No phone service" if phone_service == "No" else "No"
online_security = "No internet service" if internet_service == "No" else "No"
online_backup = "No internet service" if internet_service == "No" else "No"
device_protection = "No internet service" if internet_service == "No" else "No"
tech_support = "No internet service" if internet_service == "No" else "No"
streaming_tv = "No internet service" if internet_service == "No" else "No"
streaming_movies = "No internet service" if internet_service == "No" else "No"

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
st.subheader("📋 Data Input")
st.dataframe(input_df, use_container_width=True)

# ==================== PREDICTION ====================
if st.button("🎯 Prediksi Churn", type="primary"):
    if model:
        try:
            # Jika model adalah Pipeline
            if hasattr(model, 'named_steps'):
                st.info("🔧 Model adalah Pipeline - preprocessing otomatis")
                
                # Pastikan kolom sesuai
                if hasattr(model, 'feature_names_in_'):
                    required_cols = list(model.feature_names_in_)
                    missing_cols = [col for col in required_cols if col not in input_df.columns]
                    
                    if missing_cols:
                        st.warning(f"Kolom yang hilang: {missing_cols[:3]}")
                        # Tambah kolom default
                        for col in missing_cols:
                            if any(keyword in col.lower() for keyword in ['charge', 'tenure', 'month']):
                                input_df[col] = 0.0
                            else:
                                input_df[col] = 'Unknown'
                
                # Predict
                with st.spinner("Memprediksi..."):
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    
            else:
                # Model non-Pipeline
                if preprocessor and hasattr(preprocessor, 'transform'):
                    try:
                        input_transformed = preprocessor.transform(input_df)
                    except Exception as e:
                        st.warning(f"Transformasi gagal: {e}")
                        input_transformed = input_df
                else:
                    input_transformed = input_df
                
                with st.spinner("Memprediksi..."):
                    prediction = model.predict(input_transformed)[0]
                    probabilities = model.predict_proba(input_transformed)[0]
            
            # ===== TAMPILKAN HASIL =====
            st.subheader("🎯 Hasil Prediksi")
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.error("## 🔴 CHURN: YA")
                    st.write("Pelanggan berpotensi berhenti berlangganan")
                else:
                    st.success("## 🟢 CHURN: TIDAK")
                    st.write("Pelanggan cenderung tetap berlangganan")
            
            with col2:
                # Probabilitas
                st.metric("Probabilitas Tidak Churn", f"{probabilities[0]*100:.1f}%")
                st.metric("Probabilitas Churn", f"{probabilities[1]*100:.1f}%")
            
            # Visualisasi
            prob_df = pd.DataFrame({
                'Status': ['Tidak Churn', 'Churn'],
                'Probabilitas (%)': [probabilities[0]*100, probabilities[1]*100]
            })
            
            st.bar_chart(prob_df.set_index('Status'))
            
            # Rekomendasi
            st.subheader("💡 Rekomendasi")
            if prediction == 1:
                st.warning("""
                **Tindakan yang disarankan:**
                1. **Hubungi pelanggan** untuk feedback langsung
                2. **Tawarkan diskon khusus** atau paket loyalitas
                3. **Tinjau layanan** yang sering dikeluhkan
                4. **Proactive monitoring** penggunaan layanan
                """)
            else:
                st.info("""
                **Strategi retensi:**
                1. **Program loyalitas** untuk mempertahankan
                2. **Tawaran eksklusif** untuk pengguna setia
                3. **Komunikasi rutin** dengan newsletter
                4. **Upselling** layanan tambahan
                """)
                
        except Exception as e:
            st.error(f"❌ Error saat prediksi: {str(e)}")
            
            # Debug information
            with st.expander("🔍 Debug Details"):
                st.write("**Input columns:**", list(input_df.columns))
                st.write("**Input dtypes:**", input_df.dtypes.to_dict())
                st.write("**Model type:**", type(model).__name__)
                
                # Coba dengan data sample
                st.write("**Testing with sample data:**")
                try:
                    sample_data = pd.DataFrame({
                        'gender': ['Female'],
                        'SeniorCitizen': [0],
                        'Partner': ['Yes'],
                        'Dependents': ['No'],
                        'tenure': [12],
                        'PhoneService': ['Yes'],
                        'MultipleLines': ['No'],
                        'InternetService': ['DSL'],
                        'OnlineSecurity': ['No'],
                        'OnlineBackup': ['No'],
                        'DeviceProtection': ['No'],
                        'TechSupport': ['No'],
                        'StreamingTV': ['No'],
                        'StreamingMovies': ['No'],
                        'Contract': ['Month-to-month'],
                        'PaperlessBilling': ['Yes'],
                        'PaymentMethod': ['Electronic check'],
                        'MonthlyCharges': [70.0],
                        'TotalCharges': [840.0]
                    })
                    
                    if hasattr(model, 'predict'):
                        sample_pred = model.predict(sample_data)
                        sample_proba = model.predict_proba(sample_data)
                        st.success(f"✅ Sample test berhasil! Prediksi: {sample_pred[0]}")
                    else:
                        st.warning("Model tidak memiliki metode predict")
                except Exception as e2:
                    st.error(f"Sample test gagal: {e2}")
    else:
        st.error("""
        ⚠️ **Model tidak tersedia**
        
        Silakan:
        1. Jalankan `retrain_latest.py` untuk melatih model baru
        2. Pastikan file `model.joblib` ada di folder yang sama
        3. Refresh aplikasi setelah model tersedia
        """)

# ==================== DEPLOYMENT INFO ====================
with st.sidebar.expander("🚀 Deployment Info"):
    st.write("**Versi Package:**")
    st.write(f"- scikit-learn: {sklearn.__version__}")
    st.write(f"- numpy: {np.__version__}")
    st.write(f"- pandas: {pd.__version__}")
    
    st.write("**File Status:**")
    st.write(f"- Model: {'✅ Found' if model else '❌ Missing'}")
    if model_loc:
        st.write(f"- Location: {model_loc}")
    
    # Check environment
    st.write("**Environment:**")
    st.write(f"- Python: {sys.version[:6]}")
    st.write(f"- Platform: {sys.platform}")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Proyek UAS Bengkel Koding Data Science - Universitas Dian Nuswantoro")
st.caption(f"Deployed with: scikit-learn {sklearn.__version__}, Streamlit {st.__version__}")