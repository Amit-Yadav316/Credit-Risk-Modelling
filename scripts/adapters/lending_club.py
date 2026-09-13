"""Turn the raw Lending Club accepted-loans file into the schema this pipeline expects.

Usage:
    python scripts/adapters/lending_club.py --input ~/Downloads/accepted_2007_to_2018Q4.csv

Writes data/raw/loans.csv. After running it, apply the two config edits printed
at the end, because Lending Club has no DPD history at origination and the
delinquency features have to change.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("lending_club")

# Recorded after the loan was written. Every one of these leaks the outcome.
POST_ORIGINATION = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "debt_settlement_flag",
    "settlement_status", "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term", "hardship_flag", "hardship_type", "hardship_status",
    "hardship_start_date", "hardship_end_date", "hardship_amount",
]

# Lending Club's own risk grade. Keeping it makes the model a wrapper around theirs.
COMPETING_MODEL = ["grade", "sub_grade"]

BAD_STATUSES = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD_STATUSES = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}

SOURCE_COLUMNS = [
    "id", "issue_d", "loan_status", "loan_amnt", "annual_inc", "int_rate", "term",
    "revol_util", "fico_range_low", "fico_range_high", "open_acc", "delinq_2yrs",
    "inq_last_6mths", "mths_since_last_delinq", "pub_rec", "dti", "purpose",
    "emp_length", "home_ownership", "addr_state",
    # Origination-time columns, added in the features pass. Every one of these is
    # known when the application is decided, so none of them leaks the outcome.
    "earliest_cr_line",        # -> credit_history_months
    "revol_bal",               # absolute revolving debt, which utilisation hides
    "total_acc",               # accounts ever opened, against open_acc
    "installment",             # the real scheduled payment, not a reconstructed EMI
    "verification_status",     # was the stated income actually verified
    "mths_since_last_record",  # public-record recency, same sentinel as delinquency
]

# zip_code and addr_state both sit in the raw file. State is kept and zip is not:
# a three-digit zip prefix is a well-documented proxy for race in the US, and a
# scorecard that splits on it imports a disparate-impact problem for very little
# predictive gain.

HOME_OWNERSHIP_MAP = {
    "MORTGAGE": "owned", "OWN": "owned", "RENT": "rented",
    "ANY": "family", "NONE": "family", "OTHER": "family",
}


class LendingClubAdapter:
    """Maps, filters and cleans Lending Club into the pipeline's column contract."""

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        income_cap_pct: float = 0.99,
        chunk_size: int = 200_000,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.income_cap_pct = income_cap_pct
        self.chunk_size = chunk_size

    def run(self) -> pd.DataFrame:
        df = self._read()
        df = self._label(df)
        df = self._drop_immature_vintages(df)
        df = self._map_columns(df)
        df = self._clean(df)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info(
            "Wrote %s rows to %s | bad rate %.2f%% | %s to %s",
            f"{len(df):,}", self.output_path, 100 * df.default_flag.mean(),
            df.disbursal_date.min().date(), df.disbursal_date.max().date(),
        )
        self._print_config_edits()
        return df

    def _read(self) -> pd.DataFrame:
        available = pd.read_csv(self.input_path, nrows=0).columns
        usecols = [c for c in SOURCE_COLUMNS if c in available]
        missing = set(SOURCE_COLUMNS) - set(usecols)
        if missing:
            logger.warning("Columns absent from this dump: %s", sorted(missing))
        excluded = [c for c in POST_ORIGINATION + COMPETING_MODEL if c in available]
        logger.info("Leakage and competing-model columns left unread: %s", len(excluded))

        # The dump is 1.6 GB over 151 columns, several of them free-text fields
        # with embedded newlines, so a single read needs several GB of resident
        # memory and dies on an ordinary laptop. Stream it instead and reduce each
        # chunk immediately: rows without a finished outcome can never be labelled,
        # and dropping them here is the same filter _label applies downstream.
        keep_status = BAD_STATUSES | GOOD_STATUSES
        chunks: list[pd.DataFrame] = []
        scanned = 0
        for chunk in pd.read_csv(self.input_path, usecols=usecols, chunksize=self.chunk_size):
            scanned += len(chunk)
            chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
            chunk = chunk.dropna(subset=["issue_d", "loan_status"])
            chunks.append(chunk[chunk["loan_status"].isin(keep_status)].copy())
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        logger.info(
            "Scanned %s rows, kept %s with a known outcome", f"{scanned:,}", f"{len(df):,}"
        )
        return df

    def _label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Outcome is only known for loans that finished. Everything else goes."""
        df = df[df["loan_status"].isin(BAD_STATUSES | GOOD_STATUSES)].copy()
        df["default_flag"] = df["loan_status"].isin(BAD_STATUSES).astype(int)
        return df.drop(columns=["loan_status"])

    def _drop_immature_vintages(self, df: pd.DataFrame) -> pd.DataFrame:
        """A 36-month loan issued 12 months before the cutoff cannot have defaulted yet.

        Without this filter the most recent vintages look artificially clean, which
        is exactly the window the out-of-time split uses to judge the model.
        """
        df["term_months"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
        last = df["issue_d"].max()
        matured = df["issue_d"] + pd.to_timedelta(df["term_months"] * 30.44, unit="D") <= last
        logger.info("Dropping %s immature loans", f"{(~matured).sum():,}")
        return df[matured].copy()

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "application_id": df.get("id", pd.Series(range(len(df)), index=df.index)).astype(str),
                "disbursal_date": df["issue_d"],
                "loan_amount": df["loan_amnt"].astype(float),
                "annual_income": df["annual_inc"].astype(float),
                "tenure_months": df["term_months"].astype(int),
                "interest_rate": self._pct(df["int_rate"]),
                "credit_utilisation": self._pct(df["revol_util"]) / 100.0,
                "bureau_score": (df["fico_range_low"] + df["fico_range_high"]) / 2.0,
                "num_open_accounts": df["open_acc"],
                "num_delinquent_accounts": df["delinq_2yrs"],
                "enquiries_6m": df["inq_last_6mths"],
                "debt_to_income": df["dti"],
                "public_records": df["pub_rec"],
                "installment_amount": df["installment"].astype(float),
            "revolving_balance": df["revol_bal"].astype(float),
            "total_accounts": df["total_acc"],
            "credit_history_months": self._credit_history_months(df),
            "income_verification": self._verification(df),
            "loan_purpose": df["purpose"].astype(str),
                "employment_type": self._employment(df),
                "residence_type": df["home_ownership"].map(HOME_OWNERSHIP_MAP).fillna("family"),
                "state": df["addr_state"].astype(str),
                "default_flag": df["default_flag"],
            }
        )
        # Lending Club reports recency of the last delinquency, not days past due.
        # Never-delinquent borrowers are blank, which is the best outcome, not a
        # missing value, so they get a large sentinel rather than an imputed median.
        out["months_since_delinq"] = df["mths_since_last_delinq"].fillna(999.0)
        # Same reasoning for public records: 82% of the book is blank because the
        # borrower has never had one, which is the best outcome and not a gap.
        out["months_since_public_record"] = df["mths_since_last_record"].fillna(999.0)
        return out

    @staticmethod
    def _credit_history_months(df: pd.DataFrame) -> pd.Series:
        """Months between the first credit line and origination.

        Length of credit file is one of the strongest bureau signals that is not
        already inside FICO: two borrowers can share a score while one has been
        borrowing for twenty years and the other for eighteen months.
        """
        opened = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        months = (df["issue_d"] - opened).dt.days / 30.44
        return months.clip(lower=0)

    @staticmethod
    def _verification(df: pd.DataFrame) -> pd.Series:
        """Whether Lending Club checked the stated income.

        It conditions how much the model should trust annual_inc, which every
        affordability feature is built on.
        """
        mapping = {
            "Verified": "verified",
            "Source Verified": "source_verified",
            "Not Verified": "not_verified",
        }
        return df["verification_status"].map(mapping).fillna("not_verified")

    @staticmethod
    def _pct(series: pd.Series) -> pd.Series:
        """Lending Club stores rates as strings like '13.56%' in some dumps."""
        return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")

    @staticmethod
    def _employment(df: pd.DataFrame) -> pd.Series:
        years = df["emp_length"].astype(str).str.extract(r"(\d+)").astype(float)[0]
        return pd.cut(
            years.fillna(-1), bins=[-2, -0.5, 2, 7, 100], labels=["unknown", "junior", "mid", "long_tenure"]
        ).astype(str)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["annual_income"] > 0]
        # A handful of rows carry absurd revolving balances; cap rather than drop
        df["revolving_balance"] = df["revolving_balance"].clip(
            upper=df["revolving_balance"].quantile(self.income_cap_pct)
        )
        df["credit_history_months"] = df["credit_history_months"].clip(0, 900)
        cap = df["annual_income"].quantile(self.income_cap_pct)
        df["annual_income"] = df["annual_income"].clip(upper=cap)
        util = df["credit_utilisation"].clip(0, 2)
        df["credit_utilisation"] = util.fillna(util.median())
        df["bureau_score"] = df["bureau_score"].clip(300, 900)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.sort_values("disbursal_date").reset_index(drop=True)

    @staticmethod
    def _print_config_edits() -> None:
        logger.info(
            "\n%s",
            """
Two edits before training on this file
--------------------------------------
1. configs/config.yaml, features.numeric: replace
       - mean_dpd
       - max_dpd
   with
       - months_since_delinq
       - public_records
       - debt_to_income
   and in features.derived replace dpd_severity with delinq_recency

2. src/credit_risk/features/builder.py, in transform(), replace the
   dpd_severity line with

       df["delinq_recency"] = 100.0 / (1.0 + df["months_since_delinq"])

   Recent delinquency scores high, never-delinquent scores near zero, which
   keeps the sign consistent with the DPD feature it replaces.
""",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt Lending Club to the pipeline schema")
    parser.add_argument("--input", required=True, help="path to accepted_2007_to_2018Q4.csv")
    parser.add_argument("--output", default="data/raw/loans.csv")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="rows per streamed chunk")
    args = parser.parse_args()
    LendingClubAdapter(args.input, args.output, chunk_size=args.chunk_size).run()


if __name__ == "__main__":
    main()
