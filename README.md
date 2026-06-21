# Credit Default Risk — Predictive Modeling with Explainable AI

A production-style credit risk scoring system: predicts probability of
serious delinquency within 2 years for a borrower, with **SHAP-based
explainability** for every prediction — the kind of transparency banks
need under **Basel / SR 11-7** model-governance requirements.

Built to demonstrate the full ML-for-finance pipeline: predictive
modeling, feature engineering, classification, and explainable AI,
deployed as a real API + dashboard rather than a notebook.

## Live Components

| Component | What it does | Tech |
|---|---|---|
| `src/data_prep.py` | Cleaning, imputation, feature engineering | pandas, numpy |
| `src/train.py` | Trains LogReg baseline + XGBoost, fits SHAP explainer | scikit-learn, xgboost, shap |
| `api/main.py` | REST API — predict + explain | FastAPI |
| `app/streamlit_app.py` | Interactive dashboard (single applicant + model overview) | Streamlit |
| `tests/test_pipeline.py` | Data pipeline + API tests | pytest |
| `.github/workflows/ci.yml` | Lint, retrain, test, build Docker images on push | GitHub Actions |

## Dataset

[Give Me Some Credit](https://github.com/wang-weishuai/GiveMeSomeCredit)
(Kaggle, 2011) — 150,000 borrowers, 6.68% default rate. A public mirror
is used here since the original Kaggle competition requires
authenticated download; the CSV is included in `data/`.

**Note on macro overlays:** this dataset is a single cross-sectional
snapshot with no loan origination date, so a genuine time-varying join
against interest-rate/unemployment series isn't possible. A
`macro_stress_index` feature is included as an **illustrative overlay**
documenting how the pipeline would ingest external macro signals
(e.g. from FRED) in a production setting with vintage data — this is
called out explicitly in `src/data_prep.py` rather than presented as a
real signal.

## Modeling

- **Logistic Regression** (class-balanced, scaled features) — interpretable
  baseline comparable to a traditional credit scorecard.
- **XGBoost** (depth-4, 400 trees, `scale_pos_weight` for class imbalance) —
  primary model.
- **SHAP `TreeExplainer`** — exact, fast Shapley values for tree ensembles;
  used for both per-prediction waterfalls and global summary plots.

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.859 |
| XGBoost | 0.868 |

(XGBoost PR-AUC: 0.40, Brier score: 0.137 — reported because ROC-AUC alone
is optimistic on a ~7% positive-rate dataset like this one.)

### Engineered features

- Payment history aggregates: total late incidents, severity-weighted
  delinquency score, severe-delinquency flag
- Utilization features: high-utilization flag, utilization × open lines
- Ratio features: income per dependent, credit lines per age, log-debt-ratio
- Missingness flags for income/dependents (imputed with train-set medians,
  computed once and reused at inference to avoid leakage)
- Age-bucket one-hot encoding (risk is non-linear in age)

## Quickstart

```bash
git clone https://github.com/akshra09/credit-risk-xai.git
cd credit-risk-xai
pip install -r requirements.txt

# 1. Train (regenerates everything in /models — only needed if you want
#    to retrain; pretrained artifacts are already committed)
cd src && python train.py && cd ..

# 2. Run the API
cd api && uvicorn main:app --reload --port 8000

# 3. Run the dashboard (in a separate terminal)
cd app && streamlit run streamlit_app.py
```

API docs (Swagger UI): http://localhost:8000/docs
Dashboard: http://localhost:8501

## Run with Docker

```bash
docker compose up --build
```

This starts both the API (`:8000`) and dashboard (`:8501`), with the
dashboard configured to call the API service over the Docker network.

## API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "revolving_utilization": 0.85,
    "age": 29,
    "late_30_59": 2,
    "debt_ratio": 0.6,
    "monthly_income": 2800,
    "open_credit_lines": 4,
    "late_90_plus": 1,
    "real_estate_loans": 0,
    "late_60_89": 0,
    "dependents": 1
  }'
```

```json
{
  "default_probability": 0.9246,
  "risk_band": "Very High",
  "decision": "Decline",
  "base_rate": 0.0668,
  "top_contributors": [
    {"feature": "Delinquency severity score", "value": 5.0, "shap_contribution": 0.8567, "direction": "increases_risk"},
    {"feature": "Total late-payment incidents", "value": 3.0, "shap_contribution": 0.7485, "direction": "increases_risk"},
    {"feature": "Credit card utilization", "value": 0.85, "shap_contribution": 0.3188, "direction": "increases_risk"}
  ],
  "model_version": "xgboost_v1"
}
```

## CI/CD

On every push to `main`: lint (`flake8`) → retrain model → run pytest
suite → build both Docker images. See `.github/workflows/ci.yml`.

## Project Structure

```
credit-risk-xai/
├── data/cs-training.csv
├── src/
│   ├── data_prep.py     # cleaning + feature engineering
│   └── train.py          # LogReg + XGBoost + SHAP training pipeline
├── api/
│   ├── main.py            # FastAPI app
│   └── schemas.py
├── app/
│   └── streamlit_app.py   # dashboard
├── models/                # trained artifacts (committed for instant deploy)
├── tests/test_pipeline.py
├── .github/workflows/ci.yml
├── Dockerfile.api
├── Dockerfile.app
├── docker-compose.yml
└── requirements.txt

```
Access the app here:- https://akshra09-credit-risk-xai-appstreamlit-app-vsvp5a.streamlit.app/
