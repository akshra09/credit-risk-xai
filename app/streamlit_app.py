"""
Streamlit dashboard for the credit default risk model.

Two views:
  - Single Applicant: input a borrower profile, get probability + SHAP
    waterfall explanation (the "loan officer" view).
  - Model Overview: global SHAP summary plot + training metrics (the
    "model governance / validator" view, in the spirit of SR 11-7).

Can run standalone (loads model artifacts directly) or against the
FastAPI service via API_URL env var.
"""

import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import engineer_features  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
API_URL = os.environ.get("API_URL")  # if set, use the FastAPI backend instead

st.set_page_config(page_title="Credit Risk Explainability", layout="wide")


@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load(f"{MODELS_DIR}/xgb_model.pkl")
    explainer = joblib.load(f"{MODELS_DIR}/shap_explainer.pkl")
    feature_cols = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
    medians = joblib.load(f"{MODELS_DIR}/imputation_medians.pkl")
    shap_sample = pd.read_pickle(f"{MODELS_DIR}/shap_sample.pkl")
    shap_values_sample = np.load(f"{MODELS_DIR}/shap_values_sample.npy")
    with open(f"{MODELS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    return xgb_model, explainer, feature_cols, medians, shap_sample, shap_values_sample, metrics


st.title("🏦 Credit Default Risk — Predictive Model with Explainable AI")
st.caption(
    "Logistic Regression baseline → XGBoost, with SHAP-based per-decision "
    "explanations for model governance and regulatory transparency "
    "(Basel / SR 11-7 style)."
)

tab1, tab2 = st.tabs(["📋 Single Applicant", "📊 Model Overview"])

(xgb_model, explainer, feature_cols, medians,
 shap_sample, shap_values_sample, metrics) = load_artifacts()

# ============================== TAB 1 ==============================
with tab1:
    st.subheader("Borrower Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 38)
        monthly_income = st.number_input("Monthly Income ($)", 0, 100000, 5400, step=100)
        dependents = st.number_input("Number of Dependents", 0, 15, 0)

    with col2:
        revolving_utilization = st.slider("Credit Card Utilization", 0.0, 1.5, 0.3, 0.01)
        debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 2.0, 0.3, 0.01)
        open_credit_lines = st.number_input("Open Credit Lines/Loans", 0, 50, 8)
        real_estate_loans = st.number_input("Real Estate Loans/Lines", 0, 10, 1)

    with col3:
        late_30_59 = st.number_input("Times 30-59 Days Late", 0, 20, 0)
        late_60_89 = st.number_input("Times 60-89 Days Late", 0, 20, 0)
        late_90_plus = st.number_input("Times 90+ Days Late", 0, 20, 0)

    if st.button("Predict Default Risk", type="primary"):
        raw_row = {
            "target": np.nan,
            "revolving_utilization": revolving_utilization,
            "age": age,
            "late_30_59": late_30_59,
            "debt_ratio": debt_ratio,
            "monthly_income": monthly_income,
            "open_credit_lines": open_credit_lines,
            "late_90_plus": late_90_plus,
            "real_estate_loans": real_estate_loans,
            "late_60_89": late_60_89,
            "dependents": dependents,
        }

        if API_URL:
            resp = requests.post(f"{API_URL}/predict", json={
                k: v for k, v in raw_row.items() if k != "target"
            })
            result = resp.json()
            proba = result["default_probability"]
        else:
            df = pd.DataFrame([raw_row])
            df, _ = engineer_features(df, fit_medians=medians)
            df = df.drop(columns=["target"], errors="ignore")
            X = df.reindex(columns=feature_cols, fill_value=0)
            proba = float(xgb_model.predict_proba(X)[:, 1][0])

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Default Probability", f"{proba:.1%}",
                   delta=f"{(proba - metrics['default_rate']):.1%} vs. base rate",
                   delta_color="inverse")
        risk_band = ("Low" if proba < 0.05 else "Medium" if proba < 0.15
                     else "High" if proba < 0.35 else "Very High")
        c2.metric("Risk Band", risk_band)
        decision = "Approve" if proba < 0.15 else ("Manual Review" if proba < 0.35 else "Decline")
        c3.metric("Suggested Decision", decision)

        if not API_URL:
            st.subheader("Why this prediction? (SHAP waterfall)")
            shap_val = explainer.shap_values(X)
            fig, ax = plt.subplots(figsize=(9, 5))
            explanation = shap.Explanation(
                values=shap_val[0],
                base_values=explainer.expected_value,
                data=X.iloc[0].values,
                feature_names=feature_cols,
            )
            shap.plots.waterfall(explanation, max_display=10, show=False)
            st.pyplot(fig, clear_figure=True)
        else:
            st.subheader("Top contributing factors")
            for c in result["top_contributors"]:
                arrow = "🔺" if c["direction"] == "increases_risk" else "🔻"
                st.write(f"{arrow} **{c['feature']}** = {c['value']} "
                         f"(SHAP: {c['shap_contribution']:+.3f})")

# ============================== TAB 2 ==============================
with tab2:
    st.subheader("Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Logistic Regression AUC", metrics["logreg_auc"])
    m2.metric("XGBoost AUC", metrics["xgb_auc"])
    m3.metric("XGBoost PR-AUC", metrics["xgb_pr_auc"])
    m4.metric("Brier Score", metrics["xgb_brier_score"])

    st.caption(
        f"Trained on {metrics['n_train']:,} borrowers, evaluated on "
        f"{metrics['n_test']:,} held-out borrowers. Base default rate: "
        f"{metrics['default_rate']:.2%}."
    )

    st.divider()
    st.subheader("Global Feature Importance (SHAP summary)")
    st.caption(
        "Computed on a held-out sample of 2,000 borrowers. Shows which "
        "features drive the model's predictions overall — the kind of "
        "global transparency view a model risk / validation team would "
        "want to see."
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(
        shap_values_sample, shap_sample, feature_names=feature_cols,
        show=False, max_display=15
    )
    st.pyplot(fig, clear_figure=True)
