from pydantic import BaseModel, Field


class BorrowerInput(BaseModel):
    revolving_utilization: float = Field(
        ..., ge=0, description="Total balance on credit cards / credit limits"
    )
    age: int = Field(..., ge=18, le=110)
    late_30_59: int = Field(0, ge=0, description="Times 30-59 days past due")
    debt_ratio: float = Field(..., ge=0)
    monthly_income: float | None = Field(None, ge=0)
    open_credit_lines: int = Field(..., ge=0)
    late_90_plus: int = Field(0, ge=0, description="Times 90+ days late")
    real_estate_loans: int = Field(0, ge=0)
    late_60_89: int = Field(0, ge=0, description="Times 60-89 days past due")
    dependents: int | None = Field(0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "revolving_utilization": 0.45,
                "age": 38,
                "late_30_59": 1,
                "debt_ratio": 0.32,
                "monthly_income": 5400,
                "open_credit_lines": 8,
                "late_90_plus": 0,
                "real_estate_loans": 1,
                "late_60_89": 0,
                "dependents": 2,
            }
        }


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float
    direction: str  # "increases_risk" | "decreases_risk"


class PredictionResponse(BaseModel):
    default_probability: float
    risk_band: str
    decision: str
    base_rate: float
    top_contributors: list[ShapContribution]
    model_version: str = "xgboost_v1"
