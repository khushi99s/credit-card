"""
Credit Card Fraud Detection – Streamlit Web App
================================================
Run with:
    streamlit run app.py

Requirements (install once):
    pip install streamlit joblib scikit-learn xgboost imbalanced-learn
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join("saved_models", "best_fraud_model.pkl")
FEATURES_PATH = os.path.join("saved_models", "feature_names.pkl")


# ── Load model & feature list ─────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names


try:
    model, feature_names = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    "Enter the anonymised transaction features **(V1 – V28)** and the "
    "transaction **Amount** below to get an instant fraud prediction."
)

if not model_loaded:
    st.error(
        "⚠️  Model files not found in `saved_models/`. "
        "Please run the notebook first to train and save the model "
        "(Section 8 · Save Best Model)."
    )
    st.stop()

# ── Sidebar – model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️  About")
    st.markdown(
        """
**Project:** Smart Credit Card Fraud Detection System  
**Programme:** BSc (Hons) Computer Science  
**Module:** CN6000 Final Year Project  

**Model file:** `saved_models/best_fraud_model.pkl`  
**Features:** {}
""".format(len(feature_names))
    )
    st.divider()
    st.caption("Use the form on the right to predict whether a transaction is fraudulent.")

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("Transaction Features")

with st.form("prediction_form"):
    cols_per_row = 4
    grid_cols    = st.columns(cols_per_row)
    inputs: dict = {}

    for idx, feat in enumerate(feature_names):
        col = grid_cols[idx % cols_per_row]
        inputs[feat] = col.number_input(
            label=feat,
            value=0.0,
            format="%.4f",
            step=0.0001,
        )

    submitted = st.form_submit_button("🔍  Predict", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    X_input = pd.DataFrame([inputs], columns=feature_names)

    prediction = model.predict(X_input)[0]

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_input)[0][1])
    else:
        proba = None

    st.divider()

    if prediction == 1:
        st.error("🚨  **FRAUDULENT** transaction detected!")
    else:
        st.success("✅  Transaction appears **LEGITIMATE**.")

    if proba is not None:
        col_a, col_b = st.columns(2)
        col_a.metric("Fraud Probability", f"{proba:.2%}")
        col_b.metric("Legitimate Probability", f"{1 - proba:.2%}")
        st.progress(proba, text=f"Fraud confidence: {proba:.2%}")

    with st.expander("📋 View input values"):
        st.dataframe(
            X_input.T.rename(columns={0: "Value"}),
            use_container_width=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "BSc (Hons) Computer Science · CN6000 Final Year Project · "
    "Smart Credit Card Fraud Detection System Using Machine Learning"
)
