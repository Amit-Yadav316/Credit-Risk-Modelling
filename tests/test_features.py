import pandas as pd

from credit_risk.config import AppConfig
from credit_risk.features import FeatureBuilder

# Lending Club schema: no borrower age, no DPD history. Delinquency arrives as
# months since the last one, with 999 standing in for never delinquent.
ROW = {
    "loan_amount": 5_00_000, "annual_income": 10_00_000, "tenure_months": 36,
    "interest_rate": 14.0, "num_open_accounts": 3, "num_delinquent_accounts": 0,
    "credit_utilisation": 0.35, "bureau_score": 740, "months_since_delinq": 999.0,
    "public_records": 0, "debt_to_income": 18.4, "enquiries_6m": 1,
    # features pass
    "months_since_public_record": 999.0, "credit_history_months": 190.0,
    "total_accounts": 12, "revolving_balance": 2_00_000,
    "installment_amount": 17_089.0, "income_verification": "verified",
    "loan_purpose": "personal", "employment_type": "salaried",
    "residence_type": "owned", "state": "MH",
}


def test_derived_features_are_correct():
    cfg = AppConfig.load()
    out = FeatureBuilder.from_config(cfg).fit_transform(pd.DataFrame([ROW]))
    assert abs(out.loc[0, "lti_ratio"] - 0.5) < 1e-9
    assert 15_000 < out.loc[0, "emi_to_income"] * (10_00_000 / 12) < 20_000
    assert out.loc[0, "utilisation_band"] == "moderate"


def test_delinq_recency_is_near_zero_when_never_delinquent():
    cfg = AppConfig.load()
    out = FeatureBuilder.from_config(cfg).fit_transform(pd.DataFrame([ROW]))
    assert abs(out.loc[0, "delinq_recency"] - 0.1) < 1e-9


def test_delinq_recency_rises_as_the_delinquency_gets_more_recent():
    cfg = AppConfig.load()
    builder = FeatureBuilder.from_config(cfg)
    rows = [dict(ROW, months_since_delinq=m) for m in (999.0, 60.0, 24.0, 3.0)]
    out = builder.fit_transform(pd.DataFrame(rows))
    recency = out["delinq_recency"].tolist()
    assert recency == sorted(recency), "recent delinquency must score higher than old"
    assert abs(recency[-1] - 25.0) < 1e-9


def test_public_record_recency_mirrors_delinquency_recency():
    """Never having had a public record is the best outcome, not a missing value."""
    cfg = AppConfig.load()
    builder = FeatureBuilder.from_config(cfg)
    rows = [dict(ROW, months_since_public_record=m) for m in (999.0, 60.0, 12.0, 0.0)]
    out = builder.fit_transform(pd.DataFrame(rows))
    recency = out["public_record_recency"].tolist()
    assert recency == sorted(recency)
    assert abs(recency[0] - 0.1) < 1e-9
    assert abs(recency[-1] - 100.0) < 1e-9


def test_features_pass_ratios_are_correct():
    cfg = AppConfig.load()
    out = FeatureBuilder.from_config(cfg).fit_transform(pd.DataFrame([ROW]))
    # installment against monthly income
    assert abs(out.loc[0, "installment_to_income"] - 17_089.0 / (10_00_000 / 12)) < 1e-9
    assert abs(out.loc[0, "revol_bal_to_income"] - 0.2) < 1e-9
    assert abs(out.loc[0, "open_to_total_accounts"] - 3 / 12) < 1e-9


def test_output_columns_match_config():
    cfg = AppConfig.load()
    out = FeatureBuilder.from_config(cfg).fit_transform(pd.DataFrame([ROW]))
    assert list(out.columns) == cfg.features.all_numeric + cfg.features.all_categorical
