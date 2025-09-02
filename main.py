import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load the saved model & scaler
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

st.title("Heart Disease Prediction")

st.markdown("""
Please provide the patient’s medical details to assess the risk of heart disease
""")

# -----------------------------
# Collect input features
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 30, 80, 60)
    sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
    cp = st.selectbox("Chest Pain (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 90)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 180)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 70, 220, 130)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    # thal = st.selectbox("Thal (1 = Normal, 2 = Fixed, 3 = Reversible)", [1, 2, 3])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    # Arrange features as in training
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca]])
    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ The assessment suggests probable heart disease.")
    else:
        st.success("✅ The assessment indicates no evidence of heart disease.")
