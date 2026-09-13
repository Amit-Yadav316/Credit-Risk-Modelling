"""Deterministic feature engineering shared by training and serving.

The same transformer object is fitted in training, logged to MLflow inside the
model pipeline, and reused at scoring time. Nothing is recomputed by hand in the
Streamlit app, which is how training-serving skew gets in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.config import AppConfig


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds credit-domain derived features on top of the raw application row."""

    UTILISATION_BINS = [-np.inf, 0.30, 0.60, 0.90, np.inf]
    UTILISATION_LABELS = ["low", "moderate", "high", "stretched"]

    def __init__(self, numeric: list[str], categorical: list[str]) -> None:
        # Plain lists only. The fitted transformer is pickled into the MLflow
        # artifact, so it must not drag the whole application config with it.
        self.numeric = numeric
        self.categorical = categorical

    @classmethod
    def from_config(cls, config: AppConfig) -> FeatureBuilder:
        return cls(config.features.all_numeric, config.features.all_categorical)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureBuilder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Loan-to-income: the single strongest affordability signal on retail books
        df["lti_ratio"] = df["loan_amount"] / df["annual_income"].clip(lower=1.0)

        # Reducing-balance EMI, then EMI burden against monthly income
        monthly_rate = df["interest_rate"] / 1200.0
        n = df["tenure_months"].clip(lower=1)
        factor = (1 + monthly_rate) ** n
        emi = np.where(
            monthly_rate > 0,
            df["loan_amount"] * monthly_rate * factor / (factor - 1),
            df["loan_amount"] / n,
        )
        df["emi"] = emi
        df["emi_to_income"] = df["emi"] / (df["annual_income"].clip(lower=1.0) / 12.0)

        # Delinquency recency, inverted. Lending Club reports months since the
        # last delinquency, not days past due, so a recent delinquency scores
        # high and a never-delinquent borrower sits near zero. That keeps the
        # sign consistent with the DPD severity feature this replaces.
        df["delinq_recency"] = 100.0 / (1.0 + df["months_since_delinq"])

        # The real scheduled payment, where emi_to_income reconstructs one from
        # rate and term. Both are kept: the reconstruction is what the synthetic
        # fixture and any book without an installment column can still provide.
        df["installment_to_income"] = df["installment_amount"] / (
            df["annual_income"].clip(lower=1.0) / 12.0
        )

        # Utilisation is a ratio, so it hides scale. A 60% utilisation on a $2,000
        # limit and on a $60,000 limit are very different exposures.
        df["revol_bal_to_income"] = df["revolving_balance"] / df["annual_income"].clip(lower=1.0)

        # Share of the credit file still open. A borrower who has closed most of
        # what they ever opened looks different from one who has kept it all live.
        df["open_to_total_accounts"] = df["num_open_accounts"] / df["total_accounts"].clip(lower=1)

        # Public-record recency, mirroring delinq_recency: recent scores high,
        # never-had-one (sentinel 999) sits near zero.
        df["public_record_recency"] = 100.0 / (1.0 + df["months_since_public_record"])

        df["utilisation_band"] = pd.cut(
            df["credit_utilisation"],
            bins=self.UTILISATION_BINS,
            labels=self.UTILISATION_LABELS,
        ).astype(str)

        keep = list(self.numeric) + list(self.categorical)
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected columns after feature build: {missing}")
        return df[keep]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(list(self.numeric) + list(self.categorical))
