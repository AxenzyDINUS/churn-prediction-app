# test_model_probabilities.py
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('best_churn_model.pkl')

# Create test input
test_input = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['Yes'],
    'tenure': [24],
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
    'TotalCharges': [1680.0]
})

# Predict
prediction = model.predict(test_input)
probabilities = model.predict_proba(test_input)

print("="*50)
print("MODEL TEST RESULTS")
print("="*50)
print(f"Prediction: {'CHURN' if prediction[0] == 1 else 'NO CHURN'}")
print(f"Probabilities: {probabilities[0]}")
print(f"Sum of probabilities: {probabilities[0].sum():.4f}")
print(f"Probability NO CHURN: {probabilities[0][0]*100:.1f}%")
print(f"Probability CHURN: {probabilities[0][1]*100:.1f}%")

# Check if valid
if probabilities[0].sum() < 0.99 or probabilities[0].sum() > 1.01:
    print("❌ ERROR: Probabilities don't sum to 1!")
    print("Try recalibrating the model or check the training process.")
else:
    print("✅ Probabilities are valid")