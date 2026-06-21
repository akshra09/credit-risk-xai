"""
FastAPI service for the credit default risk model.

Endpoints:
  GET  /health        liveness check
  GET  /metrics        model evaluation metrics from training
  POST /predict         default probability + SHAP-based explanation
"""

import os
import sys

import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import engineer_features  # noqa: E402
from schemas import BorrowerInput, PredictionResponse, ShapContribution  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(
    title="Credit Default Risk API",
    description=(
        "Predicts probability of serious delinquency within 2 years, "
        "with per-prediction SHAP explainability for model governance "
        "(Basel / SR 11-7 style transparency)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load model artifacts once at startup ----------------
xgb_model = joblib.load(f"{MODELS_DIR}/xgb_model.pkl")
explainer = joblib.load(f"{MODELS_DIR}/shap_explainer.pkl")
feature_cols = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
medians = joblib.load(f"{MODELS_DIR}/imputation_medians.pkl")

with open(f"{MODELS_DIR}/metrics.json") as f:
    TRAINING_METRICS = json.load(f)

BASE_RATE = TRAINING_METRICS["default_rate"]

FEATURE_LABELS = {
    "revolving_utilization": "Credit card utilization",
    "age": "Age",
    "late_30_59": "Times 30-59 days late",
    "debt_ratio": "Debt-to-income ratio",
    "monthly_income": "Monthly income",
    "open_credit_lines": "Open credit lines/loans",
    "late_90_plus": "Times 90+ days late",
    "real_estate_loans": "Real estate loans/lines",
    "late_60_89": "Times 60-89 days late",
    "dependents": "Number of dependents",
    "total_times_late": "Total late-payment incidents",
    "any_severe_delinquency": "History of severe delinquency",
    "delinquency_severity_score": "Delinquency severity score",
    "income_per_dependent": "Income per dependent",
    "credit_lines_per_age": "Credit lines relative to age",
    "utilization_x_lines": "Utilization x open lines",
    "high_utilization_flag": "High utilization (>80%)",
    "macro_stress_index": "Macro stress overlay",
}


def _risk_band(p: float) -> str:
    if p < 0.05:
        return "Low"
    elif p < 0.15:
        return "Medium"
    elif p < 0.35:
        return "High"
    return "Very High"


def _decision(p: float) -> str:
    if p < 0.15:
        return "Approve"
    return "Manual Review" if p < 0.35 else "Decline"


def _build_feature_row(borrower: BorrowerInput) -> pd.DataFrame:
    raw_row = {
        "target": np.nan,  # placeholder, unused downstream
        "revolving_utilization": borrower.revolving_utilization,
        "age": borrower.age,
        "late_30_59": borrower.late_30_59,
        "debt_ratio": borrower.debt_ratio,
        "monthly_income": borrower.monthly_income,
        "open_credit_lines": borrower.open_credit_lines,
        "late_90_plus": borrower.late_90_plus,
        "real_estate_loans": borrower.real_estate_loans,
        "late_60_89": borrower.late_60_89,
        "dependents": borrower.dependents,
    }
    df = pd.DataFrame([raw_row])
    df, _ = engineer_features(df, fit_medians=medians)
    df = df.drop(columns=["target"], errors="ignore")
    # Align to the exact training feature schema (handles one-hot bucket
    # columns that won't all appear for a single row)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return TRAINING_METRICS


@app.post("/predict", response_model=PredictionResponse)
def predict(borrower: BorrowerInput):
    try:
        X = _build_feature_row(borrower)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {e}")

    proba = float(xgb_model.predict_proba(X)[:, 1][0])

    shap_values = explainer.shap_values(X)[0]
    contributions = list(zip(feature_cols, X.iloc[0].values, shap_values))
    contributions.sort(key=lambda t: abs(t[2]), reverse=True)

    top = []
    for feat, val, sv in contributions[:6]:
        if abs(sv) < 1e-6:
            continue
        top.append(ShapContribution(
            feature=FEATURE_LABELS.get(feat, feat),
            value=round(float(val), 3),
            shap_contribution=round(float(sv), 4),
            direction="increases_risk" if sv > 0 else "decreases_risk",
        ))

    return PredictionResponse(
        default_probability=round(proba, 4),
        risk_band=_risk_band(proba),
        decision=_decision(proba),
        base_rate=BASE_RATE,
        top_contributors=top,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
