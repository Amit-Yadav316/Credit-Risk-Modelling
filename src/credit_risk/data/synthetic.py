"""Synthetic loan book shaped like the adapted Lending Club extract.

This is the CI fixture and the way the repo runs end to end without the 1.6 GB
Kaggle download. It therefore has to satisfy exactly the same column contract as
`scripts/adapters/lending_club.py` produces, which `tests/test_synthetic.py`
asserts on every run: when the feature list in the config moves, that test fails
here rather than four minutes into a training job.

The signal structure is deliberate, not noise with a label stapled on:
affordability (loan-to-income, EMI burden, DTI), bureau history (FICO,
utilisation, delinquency recency, enquiries, public records) and a vintage
effect where later months are slightly riskier, so the out-of-time split and the
PSI monitor both have something real to detect.

No borrower age. Age is a protected characteristic under ECOA / Regulation B, it
is absent from the Lending Club file for that reason, and generating a column the
scorecard must never use only invites someone to use it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# US states plus DC, matching the addr_state domain the adapter passes through
STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID",
    "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC",
    "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]

# Lending Club's purpose domain, with roughly its observed mix
PURPOSES = [
    "debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
    "renewable_energy", "educational",
]
PURPOSE_P = [0.572, 0.236, 0.058, 0.055, 0.021, 0.013, 0.012, 0.011, 0.007, 0.007,
             0.0045, 0.0035, 0.0008, 0.0007]

# The buckets LendingClubAdapter._employment derives from emp_length
EMPLOYMENT = ["long_tenure", "mid", "junior", "unknown"]
EMPLOYMENT_P = [0.399, 0.306, 0.239, 0.056]

RESIDENCE = ["owned", "rented", "family"]
RESIDENCE_P = [0.574, 0.425, 0.001]

# Whether Lending Club verified the stated income
VERIFICATION = ["source_verified", "verified", "not_verified"]
VERIFICATION_P = [0.423, 0.305, 0.272]

# Lending Club leaves mths_since_last_delinq blank for borrowers who have never
# been delinquent, and the adapter maps that to this sentinel. Half the book.
# mths_since_last_record is blank far more often -- 82% have never had a public
# record -- and carries the same sentinel for the same reason.
NEVER_DELINQUENT = 999.0
NO_PUBLIC_RECORD = 999.0


# Multiplies the whole log-odds structure below. Calibrated, not guessed: the
# coefficients are written at readable magnitudes and this scales them until the
# fixture separates risk about as well as the real book does. Measured on 60k
# generated rows, the AUC learnable from the deterministic signal is
#
#     scale 1.0 -> AUC 0.653, KS 0.227     (below the config's promotion floors)
#     scale 1.4 -> AUC 0.711, KS 0.314     (real book: OOT AUC 0.697, KS 0.283)
#     scale 1.8 -> AUC 0.761, KS 0.392     (implausibly separable for a retail book)
#
# It matters because the promotion gate in configs/config.yaml is tuned for the
# real portfolio. A fixture that cannot clear those floors means CI trains a model
# and then fails when nothing can be promoted, which looks like a pipeline bug and
# is really just an unrealistic fixture.
SIGNAL_SCALE = 1.4


def _weights(p: list[float]) -> np.ndarray:
    """Normalise a hand-written mix so rounding cannot break the generator."""
    w = np.asarray(p, dtype=float)
    return w / w.sum()


class SyntheticBookGenerator:
    """Builds a book with the same columns, dtypes and domains as the real extract."""

    def __init__(
        self,
        n: int = 50_000,
        seed: int = 42,
        target_bad_rate: float = 0.148,
        start: str = "2013-01-01",
        months: int = 36,
    ) -> None:
        self.n = int(n)
        self.seed = int(seed)
        self.target_bad_rate = float(target_bad_rate)
        self.start = pd.Timestamp(start)
        self.months = int(months)

    def generate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.n
        span_days = int(self.months * 30.44)
        dates = pd.Series(self.start + pd.to_timedelta(rng.integers(0, span_days, n), unit="D"))

        annual_income = np.round(rng.lognormal(mean=11.1, sigma=0.55, size=n), -2).clip(1_200, 250_000)
        loan_amount = np.round(annual_income * rng.uniform(0.05, 0.45, n), -2).clip(500, 35_000)
        tenure = rng.choice([36, 60], n, p=[0.92, 0.08])
        interest = np.round(rng.normal(12.4, 4.1, n).clip(5.32, 28.99), 2)
        open_accts = rng.poisson(11.2, n)
        delinq = rng.poisson(0.31, n)
        utilisation = (rng.beta(2.6, 2.2, n) * 1.25).clip(0, 2).round(3)
        bureau = rng.normal(697, 31, n).clip(612, 850).round(1)
        enquiries = rng.poisson(0.71, n)
        public_records = rng.poisson(0.20, n)
        dti = np.round(rng.normal(17.5, 8.3, n).clip(0, 60), 2)

        # Half the book has never been delinquent and carries the sentinel; the
        # rest gets a recency in months, skewed towards the recent end.
        ever_delinquent = rng.random(n) < 0.483
        months_since = np.where(
            ever_delinquent, rng.gamma(2.2, 14.0, n).clip(0, 150).round(0), NEVER_DELINQUENT
        )
        has_record = rng.random(n) < 0.176
        months_since_record = np.where(
            has_record, rng.normal(66.0, 24.0, n).clip(0, 120).round(0), NO_PUBLIC_RECORD
        )

        # Length of the credit file, in months. Correlated with nothing else here
        # on purpose: it carries signal FICO does not already contain.
        credit_history = rng.gamma(4.0, 46.0, n).clip(6, 720).round(0)
        # Accounts ever opened must be at least the number still open
        total_accounts = (open_accts + rng.poisson(13.5, n)).clip(1, 200)
        revolving_balance = np.round(rng.lognormal(9.2, 1.05, n), -1).clip(0, 250_000)
        # The real scheduled payment: a reducing-balance EMI on the drawn terms
        monthly_rate = interest / 1200.0
        growth = (1 + monthly_rate) ** tenure
        installment = np.round(
            loan_amount * monthly_rate * growth / (growth - 1), 2
        )
        verification = rng.choice(VERIFICATION, n, p=_weights(VERIFICATION_P))

        purpose = rng.choice(PURPOSES, n, p=_weights(PURPOSE_P))
        employment = rng.choice(EMPLOYMENT, n, p=_weights(EMPLOYMENT_P))
        residence = rng.choice(RESIDENCE, n, p=_weights(RESIDENCE_P))
        state = rng.choice(STATES, n)

        lti = loan_amount / annual_income
        # The same transforms FeatureBuilder applies, so the fixture carries signal
        # in the derived features the model actually sees
        delinq_recency = 100.0 / (1.0 + months_since)
        record_recency = 100.0 / (1.0 + months_since_record)
        revol_to_income = revolving_balance / np.clip(annual_income, 1.0, None)
        vintage = np.asarray((dates - self.start).dt.days, dtype=float) / span_days

        raw_logit = SIGNAL_SCALE * (
            1.85 * lti
            + 0.020 * delinq_recency
            + 0.95 * utilisation
            - 0.0120 * (bureau - 697)
            + 0.13 * delinq
            + 0.11 * enquiries
            + 0.16 * public_records
            + 0.0180 * dti
            + 0.060 * (interest - 12.4)
            + 0.20 * (employment == "unknown")
            + 0.18 * (residence == "rented")
            + 0.35 * vintage
            # Features pass: a longer credit file is safer, a recent public record
            # is not, and unverified income makes the stated figure less reliable.
            - 0.0016 * (credit_history - 190.0)
            + 0.014 * record_recency
            + 0.22 * revol_to_income
            + 0.12 * (verification == "not_verified")
        ) + rng.normal(0, 0.45, n)

        intercept = self._solve_intercept(raw_logit)
        p_default = 1 / (1 + np.exp(-(raw_logit + intercept)))
        default_flag = rng.binomial(1, p_default)

        return pd.DataFrame(
            {
                "application_id": [f"APP{i:07d}" for i in range(n)],
                "disbursal_date": dates,
                "loan_amount": loan_amount,
                "annual_income": annual_income,
                "tenure_months": tenure,
                "interest_rate": interest,
                "credit_utilisation": utilisation,
                "bureau_score": bureau,
                "num_open_accounts": open_accts,
                "num_delinquent_accounts": delinq,
                "enquiries_6m": enquiries,
                "debt_to_income": dti,
                "public_records": public_records,
                "months_since_delinq": months_since,
                "months_since_public_record": months_since_record,
                "credit_history_months": credit_history,
                "total_accounts": total_accounts,
                "revolving_balance": revolving_balance,
                "installment_amount": installment,
                "income_verification": verification,
                "loan_purpose": purpose,
                "employment_type": employment,
                "residence_type": residence,
                "state": state,
                "default_flag": default_flag,
            }
        ).sort_values("disbursal_date").reset_index(drop=True)

    def _solve_intercept(self, raw_logit: np.ndarray) -> float:
        """Bisect for the intercept that lands the book on the target bad rate.

        Solving for it keeps the generator honest when the coefficients change,
        instead of someone hand-tuning a constant until the number looks right.
        """
        lo, hi = -12.0, 4.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(raw_logit + mid)))).mean() > self.target_bad_rate:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
