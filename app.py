import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
rf = joblib.load('nourishkids_rf_model.pkl')
scaler = joblib.load('nourishkids_scaler.pkl')

st.set_page_config(page_title="NourishKids Risk Checker", layout="centered")
st.title("🧒 NourishKids AI Risk Checker")
st.markdown("Estimate a child’s risk of malnutrition using basic health & household data.")
st.sidebar.header("👶 Enter Child Details")

# Input fields — these must come BEFORE the prediction button
age = st.sidebar.slider("Age (months)", 0, 59, 24)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
region = st.sidebar.selectbox("Region (Encoded)", list(range(10)))
mother_edu = st.sidebar.selectbox("Mother's Education (Encoded)", list(range(5)))
wealth_index = st.sidebar.slider("Household Wealth Index", 1, 5, 3)

height = st.sidebar.number_input("Height (cm)", min_value=30.0, max_value=120.0, value=85.0, step=0.5)
weight = st.sidebar.number_input("Weight (kg)", min_value=2.0, max_value=25.0, value=10.0, step=0.1)

stunting = st.sidebar.selectbox("Stunted?", ["No", "Yes"])
underweight = st.sidebar.selectbox("Underweight?", ["No", "Yes"])
overweight = st.sidebar.selectbox("Overweight?", ["No", "Yes"])
anemia = st.sidebar.selectbox("Anemia?", ["No", "Yes"])
malaria = st.sidebar.selectbox("Malaria?", ["No", "Yes"])
diarrhea = st.sidebar.selectbox("Diarrhea?", ["No", "Yes"])
tb = st.sidebar.selectbox("Tuberculosis?", ["No", "Yes"])

# When user clicks the button
if st.button("🔍 Predict Nutrition Status"):

    # Engineered features
    weight_height_ratio = weight / height
    mother_edu_level = 0 if mother_edu in [0, 1] else 1
    health_risk_sum = sum([anemia == "Yes", diarrhea == "Yes", tb == "Yes"])

    # Feature vector
    features = np.array([[
        age,
        0 if gender == "Male" else 1,
        region,
        mother_edu,
        wealth_index,
        height,
        weight,
        1 if stunting == "Yes" else 0,
        1 if underweight == "Yes" else 0,
        1 if overweight == "Yes" else 0,
        1 if anemia == "Yes" else 0,
        1 if malaria == "Yes" else 0,
        1 if diarrhea == "Yes" else 0,
        1 if tb == "Yes" else 0,
        weight_height_ratio,
        mother_edu_level,
        health_risk_sum
    ]])

    # Scale input
    features_scaled = scaler.transform(features)

    # Predict
    prediction = rf.predict(features_scaled)[0]
    probs = rf.predict_proba(features_scaled)[0]

    labels = ['Normal', 'At Risk', 'Malnourished']
    status_map = {0: "🟢 Normal", 1: "🟠 At Risk", 2: "🔴 Malnourished"}

    # Override prediction based on thresholds (force red flag if Malnourished is high)
    if probs[2] > 0.30:
        override = "🔴 Malnourished"
    elif probs[1] > 0.30:
        override = "🟠 At Risk"
    else:
        override = "🟢 Normal"

    st.subheader("Prediction (Rule-Augmented):")
    st.markdown(f"<h3 style='color: teal;'>{override}</h3>", unsafe_allow_html=True)


    # Display prediction
    st.subheader("Prediction:")
    st.markdown(f"<h3 style='color: teal;'>{status_map[prediction]}</h3>", unsafe_allow_html=True)
    st.markdown("This result is based on clinical and socioeconomic risk factors.")

    # Display probabilities
    st.markdown("#### Prediction Confidence:")
    for i, prob in enumerate(probs):
        st.write(f"{labels[i]}: {prob * 100:.2f}%")

    # Show bar chart
    fig, ax = plt.subplots()
    ax.bar(labels, probs, color=["green", "orange", "red"])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)

st.markdown("---")
st.caption("This prototype is based on a machine learning model trained on child nutrition data from Ethiopia.")
