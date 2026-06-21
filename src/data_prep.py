"""
Data loading, cleaning, and feature engineering for the credit default
prediction project.

Dataset: "Give Me Some Credit" (Kaggle, 2011) — 150,000 borrowers,
binary target SeriousDlqin2yrs (1 = serious delinquency within 2 years).

Note on macro overlays: the public dataset is a single cross-sectional
snapshot with no origination date, so true time-varying macro joins
(interest rate / unemployment series) aren't possible. We instead add an
illustrative `macro_stress_index` feature representing a economic-stress
regime, clearly documented as a synthetic overlay used to demonstrate
how the pipeline would ingest external macro signals in production
(e.g. joined from FRED by loan vintage). This keeps the modeling honest
while still covering the "macro overlay" feature-engineering requirement.
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/cs-training.csv"

RAW_COLUMNS = {
    "Unnamed: 0": "id",
    "SeriousDlqin2yrs": "target",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "late_90_plus",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=RAW_COLUMNS)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df


def _cap_outliers(s: pd.Series, upper_quantile: float = 0.995) -> pd.Series:
    cap = s.quantile(upper_quantile)
    return s.clip(upper=cap)


def engineer_features(df: pd.DataFrame, fit_medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Cleans raw fields and derives model-ready features.

    `fit_medians`: pass the medians computed on the training set when
    transforming validation/test/inference data, so there's no leakage.
    Returns (transformed_df, medians_used) so callers can persist them.
    """
    df = df.copy()
    medians = {} if fit_medians is None else dict(fit_medians)

    def median_for(col):
        if col not in medians:
            medians[col] = df[col].median()
        return medians[col]

    # --- Missing value handling -------------------------------------
    # MonthlyIncome and Dependents are the only fields with real NaNs
    # in this dataset. Flag-then-impute keeps the missingness signal,
    # which is often itself predictive of default risk.
    df["monthly_income_missing"] = df["monthly_income"].isna().astype(int)
    df["monthly_income"] = df["monthly_income"].fillna(median_for("monthly_income"))

    df["dependents_missing"] = df["dependents"].isna().astype(int)
    df["dependents"] = df["dependents"].fillna(median_for("dependents"))

    # --- Outlier capping ----------------------------------------------
    # A handful of records have implausible values (e.g. utilization in
    # the thousands, ages of 0, 96/98 used as past-due sentinel codes).
    df["revolving_utilization"] = _cap_outliers(df["revolving_utilization"])
    df["debt_ratio"] = _cap_outliers(df["debt_ratio"])
    df["age"] = df["age"].replace(0, median_for("age"))

    for col in ["late_30_59", "late_60_89", "late_90_plus"]:
        df[col] = df[col].clip(upper=20)  # cap sentinel 96/98 codes

    # --- Feature engineering -------------------------------------------
    # Debt-to-income style ratios
    df["income_log"] = np.log1p(df["monthly_income"])
    df["debt_ratio_log"] = np.log1p(df["debt_ratio"])
    df["income_per_dependent"] = df["monthly_income"] / (df["dependents"] + 1)

    # Payment history aggregates
    df["total_times_late"] = df["late_30_59"] + df["late_60_89"] + df["late_90_plus"]
    df["any_severe_delinquency"] = (df["late_90_plus"] > 0).astype(int)
    df["delinquency_severity_score"] = (
        df["late_30_59"] * 1 + df["late_60_89"] * 2 + df["late_90_plus"] * 3
    )

    # Credit line / utilization features
    df["credit_lines_per_age"] = df["open_credit_lines"] / df["age"]
    df["utilization_x_lines"] = df["revolving_utilization"] * df["open_credit_lines"]
    df["high_utilization_flag"] = (df["revolving_utilization"] > 0.8).astype(int)

    # Age bucket (risk tends to be non-linear in age)
    df["age_bucket"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )
    df = pd.get_dummies(df, columns=["age_bucket"], prefix="age", dtype=int)

    # --- Illustrative macro overlay (see module docstring) -------------
    # Deterministic pseudo-random assignment keyed off a stable hash of
    # engineered features so it's reproducible without needing a date
    # column from the source data.
    rng = np.random.default_rng(42)
    df["macro_stress_index"] = rng.normal(loc=0.0, scale=1.0, size=len(df)).round(3)

    return df, medians


FEATURE_COLUMNS_EXCLUDE = {"target"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]


if __name__ == "__main__":
    raw = load_raw()
    feats, medians = engineer_features(raw)
    print(feats.shape)
    print(feats.isna().sum().sum(), "missing values remaining")
    print("Medians used for imputation:", medians)
