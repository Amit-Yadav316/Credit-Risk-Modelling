"""Common contract for every candidate model in the champion-challenger setup."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.config import AppConfig


class RiskModel(ABC):
    """A model is a full sklearn Pipeline: features in, calibrated PD out.

    Keeping preprocessing inside the pipeline means the MLflow artifact is the
    entire decision function, not just the estimator, so serving cannot drift
    from training.
    """

    name: str = "base"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.pipeline: Pipeline | None = None
        self.best_params: dict[str, Any] = {}
        self.true_prior: float | None = None

    def set_prior(self, prior: float) -> RiskModel:
        """Population bad rate, used to undo the effect of resampling."""
        self.true_prior = float(prior)
        return self

    @abstractmethod
    def build(self, params: dict[str, Any] | None = None) -> Pipeline:
        """Return an unfitted pipeline for the given hyperparameters."""

    @abstractmethod
    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Search space for Optuna."""

    def fit(self, X: pd.DataFrame, y: pd.Series, params: dict[str, Any] | None = None) -> RiskModel:
        if self.true_prior is None:
            self.set_prior(float(y.mean()))
        self.best_params = params or self.best_params
        self.pipeline = self.build(self.best_params)
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted")
        return self.pipeline.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["feature", "importance"])
