"""Loading and out-of-time splitting of the application book."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class SplitBundle:
    """Three-way split: in-time train, in-time test, out-of-time holdout."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_oot: pd.DataFrame
    y_oot: pd.Series

    def summary(self) -> dict[str, float]:
        return {
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "n_oot": len(self.X_oot),
            "bad_rate_train": float(self.y_train.mean()),
            "bad_rate_test": float(self.y_test.mean()),
            "bad_rate_oot": float(self.y_oot.mean()),
        }


class DataLoader:
    """Reads the raw book and produces a leakage-safe split.

    The out-of-time window is carved off by disbursal date before any random
    split, so model selection is judged on a period the model never saw. This is
    the split regulators and model-risk teams expect on a scorecard.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.dc = config.data

    def load_raw(self, path: str | Path | None = None) -> pd.DataFrame:
        path = self.config.resolve(str(path or self.dc.raw_path))
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/make_synthetic_data.py` first "
                "or point data.raw_path at your own extract."
            )
        df = pd.read_csv(path, parse_dates=[self.dc.date_col])
        logger.info("Loaded %s rows from %s", f"{len(df):,}", path)
        return df

    def split(self, df: pd.DataFrame) -> SplitBundle:
        df = df.sort_values(self.dc.date_col).reset_index(drop=True)
        cutoff = df[self.dc.date_col].max() - pd.DateOffset(months=self.dc.oot_months)
        in_time = df[df[self.dc.date_col] <= cutoff]
        oot = df[df[self.dc.date_col] > cutoff]

        drop_cols = [self.dc.target, self.dc.date_col, self.dc.id_col]
        X = in_time.drop(columns=[c for c in drop_cols if c in in_time.columns])
        y = in_time[self.dc.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.dc.test_size,
            random_state=self.dc.random_state,
            stratify=y,
        )
        bundle = SplitBundle(
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            X_oot=oot.drop(columns=[c for c in drop_cols if c in oot.columns]).reset_index(drop=True),
            y_oot=oot[self.dc.target].reset_index(drop=True),
        )
        logger.info("Split summary: %s", bundle.summary())
        return bundle
