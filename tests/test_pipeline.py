import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "api"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from data_prep import load_raw, engineer_features, get_feature_columns  # noqa: E402

GOOD_BORROWER = {
    "revolving_utilization": 0.1,
    "age": 45,
    "late_30_59": 0,
    "debt_ratio": 0.2,
    "monthly_income": 8000,
    "open_credit_lines": 10,
    "late_90_plus": 0,
    "real_estate_loans": 2,
    "late_60_89": 0,
    "dependents": 1,
}

RISKY_BORROWER = {
    "revolving_utilization": 0.95,
    "age": 24,
    "late_30_59": 3,
    "debt_ratio": 0.9,
    "monthly_income": 1800,
    "open_credit_lines": 2,
    "late_90_plus": 2,
    "real_estate_loans": 0,
    "late_60_89": 1,
    "dependents": 3,
}


def test_load_raw():
    df = load_raw()
    assert "target" in df.columns
    assert len(df) > 100000


def test_engineer_features_no_nulls():
    raw = load_raw()
    feats, medians = engineer_features(raw)
    assert feats.isna().sum().sum() == 0
    assert "macro_stress_index" in feats.columns
    assert "total_times_late" in feats.columns


def test_feature_columns_exclude_target():
    raw = load_raw()
    feats, _ = engineer_features(raw)
    cols = get_feature_columns(feats)
    assert "target" not in cols


@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_metrics(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "xgb_auc" in body
    assert body["xgb_auc"] > 0.8


def test_predict_good_borrower_lower_risk_than_risky(client):
    good = client.post("/predict", json=GOOD_BORROWER).json()
    risky = client.post("/predict", json=RISKY_BORROWER).json()
    assert good["default_probability"] < risky["default_probability"]
    assert good["decision"] == "Approve"
    assert risky["decision"] == "Decline"


def test_predict_response_schema(client):
    resp = client.post("/predict", json=GOOD_BORROWER)
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_band"] in {"Low", "Medium", "High", "Very High"}
    assert isinstance(body["top_contributors"], list)
    assert len(body["top_contributors"]) > 0


def test_predict_invalid_input_rejected(client):
    bad = dict(GOOD_BORROWER)
    bad["age"] = 5  # below allowed minimum
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
