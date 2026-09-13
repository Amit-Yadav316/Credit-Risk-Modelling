"""FastAPI scoring service backed by the MLflow registry champion."""
from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk.scoring.service import get_scoring_service

app = FastAPI(title="Credit Risk Scoring API", version="0.1.0")

# The domains the model was actually trained on, taken from the adapted book
# rather than invented. A request outside them is extrapolation, so the schema
# rejects it here instead of returning a confident score for an unseen region.
LoanPurpose = Literal[
    "debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
    "renewable_energy", "educational",
]
EmploymentType = Literal["unknown", "junior", "mid", "long_tenure"]
IncomeVerification = Literal["verified", "source_verified", "not_verified"]
ResidenceType = Literal["owned", "rented", "family"]

NEVER_DELINQUENT = 999.0
NO_PUBLIC_RECORD = 999.0


class LoanApplication(BaseModel):
    """One application, as it looks at origination.

    Derived features (loan-to-income, EMI burden, delinquency recency,
    utilisation band) are never computed here. They are built inside the fitted
    pipeline, so the MLflow artifact is the whole decision function and serving
    cannot drift from training.

    No borrower age, sex, race or marital status. Those are protected
    characteristics under ECOA / Regulation B and the scorecard does not use them.
    """

    loan_amount: float = Field(gt=0, le=100_000, description="Principal requested")
    annual_income: float = Field(gt=0, le=10_000_000, description="Self-reported gross annual income")
    tenure_months: int = Field(ge=6, le=120, description="Term. The book contains 36 and 60")
    interest_rate: float = Field(ge=1, le=40, description="Offered APR, percent")
    credit_utilisation: float = Field(ge=0, le=2, description="Revolving balance over limit")
    bureau_score: float = Field(ge=300, le=900, description="FICO range midpoint at origination")
    debt_to_income: float = Field(ge=0, le=1000, description="Monthly debt over monthly income, percent")
    months_since_delinq: float = Field(
        ge=0,
        le=999,
        description=(
            f"Months since the last delinquency. Send {NEVER_DELINQUENT:.0f} for a borrower "
            "who has never been delinquent: that is the best outcome, not a missing value, "
            "and it must not be imputed to a median."
        ),
    )

    installment_amount: float = Field(
        gt=0, le=10_000, description="Scheduled monthly payment on the offered terms"
    )
    revolving_balance: float = Field(ge=0, le=1_000_000, description="Total revolving balance")
    total_accounts: int = Field(ge=1, le=200, description="Credit lines ever opened")
    credit_history_months: float = Field(
        ge=0, le=900, description="Months between the first credit line and today"
    )
    months_since_public_record: float = Field(
        ge=0,
        le=999,
        description=(
            f"Months since the last public record. Send {NO_PUBLIC_RECORD:.0f} for a borrower "
            "who has never had one, which is the case for 82% of the book."
        ),
    )
    income_verification: IncomeVerification

    # Bureau counters are genuinely absent for a small share of applications, and
    # the pipeline's median imputer is the right place to resolve that, so null is
    # accepted rather than forcing the caller to invent a zero.
    num_open_accounts: int | None = Field(default=None, ge=0, le=150)
    num_delinquent_accounts: int | None = Field(default=None, ge=0, le=50)
    public_records: int | None = Field(default=None, ge=0, le=100)
    enquiries_6m: int | None = Field(default=None, ge=0, le=50)

    loan_purpose: LoanPurpose
    employment_type: EmploymentType
    residence_type: ResidenceType
    state: str = Field(min_length=2, max_length=2, description="Two-letter US state or DC")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "loan_amount": 12000,
                    "annual_income": 65000,
                    "tenure_months": 36,
                    "interest_rate": 12.4,
                    "credit_utilisation": 0.54,
                    "bureau_score": 697,
                    "debt_to_income": 17.5,
                    "months_since_delinq": 999,
                    "months_since_public_record": 999,
                    "installment_amount": 401.2,
                    "revolving_balance": 12000,
                    "total_accounts": 24,
                    "credit_history_months": 190,
                    "income_verification": "source_verified",
                    "num_open_accounts": 11,
                    "num_delinquent_accounts": 0,
                    "public_records": 0,
                    "enquiries_6m": 0,
                    "loan_purpose": "debt_consolidation",
                    "employment_type": "long_tenure",
                    "residence_type": "owned",
                    "state": "CA",
                }
            ]
        }
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict[str, str]:
    svc = get_scoring_service()
    _ = svc.model
    return {"name": svc._model_name, "alias": svc.alias, "version": svc._version}


@app.post("/score")
def score(application: LoanApplication) -> dict:
    try:
        return get_scoring_service().score_one(application.model_dump()).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Scoring failed: {exc}") from exc


@app.post("/score/batch")
def score_batch(applications: list[LoanApplication]) -> list[dict]:
    try:
        frame = pd.DataFrame([a.model_dump() for a in applications])
        return get_scoring_service().score_batch(frame).to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Scoring failed: {exc}") from exc
