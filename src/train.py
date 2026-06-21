"""
Trains and compares two models for credit default prediction:
  1. Logistic Regression  — interpretable baseline, standard in credit
     risk scorecards (and a regulator-familiar benchmark).
  2. XGBoost               — gradient-boosted trees, typically the
     strongest tabular performer on this dataset.

Saves both models, the SHAP explainer, evaluation metrics, and the
imputation medians (needed at inference time) to /models.
"""

import json
import time

import joblib
import numpy as np
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    classification_report, brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_prep import load_raw, engineer_features, get_feature_columns

MODELS_DIR = "models"
RANDOM_STATE = 42


def main():
    print("Loading data...")
    raw = load_raw()
    df, medians = engineer_features(raw)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["target"]

    print(f"Rows: {len(df):,} | Features: {len(feature_cols)} | "
          f"Default rate: {y.mean():.3%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ---------------- Logistic Regression baseline ----------------
    print("\nTraining Logistic Regression baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    )
    logreg.fit(X_train_scaled, y_train)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_auc = roc_auc_score(y_test, logreg_proba)
    print(f"Logistic Regression AUC: {logreg_auc:.4f}")

    # ---------------- XGBoost ----------------
    print("\nTraining XGBoost...")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    xgb_model.fit(X_train, y_train)
    print(f"Trained in {time.time() - t0:.1f}s")

    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    print(f"XGBoost AUC: {xgb_auc:.4f}")

    # PR-AUC (more informative than ROC-AUC given ~6.7% positive rate)
    prec, rec, _ = precision_recall_curve(y_test, xgb_proba)
    pr_auc = auc(rec, prec)
    brier = brier_score_loss(y_test, xgb_proba)

    print("\nXGBoost classification report (threshold=0.5):")
    print(classification_report(y_test, (xgb_proba > 0.5).astype(int)))

    # ---------------- SHAP explainability ----------------
    print("\nBuilding SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(xgb_model)
    # Background sample for global summary plot generation in the app
    shap_sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
    shap_values_sample = explainer.shap_values(shap_sample)

    # ---------------- Persist everything ----------------
    import os
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(logreg, f"{MODELS_DIR}/logreg_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(xgb_model, f"{MODELS_DIR}/xgb_model.pkl")
    joblib.dump(explainer, f"{MODELS_DIR}/shap_explainer.pkl")
    joblib.dump(feature_cols, f"{MODELS_DIR}/feature_columns.pkl")
    joblib.dump(medians, f"{MODELS_DIR}/imputation_medians.pkl")

    # Save a SHAP background sample + values for the dashboard's global
    # summary plot (so the deployed app doesn't need to recompute it)
    shap_sample.to_pickle(f"{MODELS_DIR}/shap_sample.pkl")
    np.save(f"{MODELS_DIR}/shap_values_sample.npy", shap_values_sample)

    metrics = {
        "logreg_auc": round(float(logreg_auc), 4),
        "xgb_auc": round(float(xgb_auc), 4),
        "xgb_pr_auc": round(float(pr_auc), 4),
        "xgb_brier_score": round(float(brier), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "default_rate": round(float(y.mean()), 4),
    }
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved models + metrics to /models")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
