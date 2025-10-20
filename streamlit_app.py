# streamlit_app.py

import streamlit as st
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression  # optional if used


# Load trained model
model = joblib.load("model.pkl")

# App UI
st.set_page_config(page_title="Student Pass Predictor", layout="centered")
st.title("🎓 Student Pass Predictor")
st.markdown("Enter student info to predict if they'll pass.")

# Input sliders
score = st.slider("📊 Score", 0, 100, 85)
age = st.slider("🎂 Age", 15, 30, 22)

# Predict button
if st.button("🔍 Predict"):
    # 🔄 FIXED: Match the model's expected input order: ["Age", "Score"]
    input_df = pd.DataFrame([[age, score]], columns=["Age", "Score"])
    result = model.predict(input_df)[0]
    
    if result == 1:
        st.success("🎉 Prediction: Passed ✅")
    else:
        st.error("❌ Prediction: Failed")

# Optional Footer
st.markdown("---")
st.caption("🔧 Powered by Nithish's ML Capstone Project 🚀")
