"""Serving layer: pull the champion from the MLflow registry and score."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import mlflow
import pandas as pd

from credit_risk.config import AppConfig
from credit_risk.scoring.explainer import ReasonCodeExplainer
from credit_risk.scoring.scorecard import ScoreScaler

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    probability_of_default: float
    score: int
    band: str
    decision: str
    model_name: str
    model_version: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScoringService:
    """Loads `models:/<name>@champion` once and serves batch or single requests.

    The alias, not a hard-coded run id, is the contract between training and
    serving. Promoting a challenger is a registry operation, not a redeploy.
    """

    def __init__(self, config: AppConfig, alias: str = "champion") -> None:
        self.config = config
        self.alias = alias
        self.scaler = ScoreScaler(config.scorecard)
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        self._model = None
        self._explainer: ReasonCodeExplainer | None = None
        self._version = "unloaded"
        self._model_name = config.mlflow.registered_model

    @property
    def model(self):
        if self._model is None:
            uri = f"models:/{self._model_name}@{self.alias}"
            logger.info("Loading %s", uri)
            self._model = mlflow.sklearn.load_model(uri)
            self._version = self._resolve_version()
        return self._model

    def _resolve_version(self) -> str:
        try:
            client = mlflow.MlflowClient()
            mv = client.get_model_version_by_alias(self._model_name, self.alias)
            return str(mv.version)
        except Exception:  # registry unreachable in local dev
            return "unknown"

    def score_batch(self, applications: pd.DataFrame) -> pd.DataFrame:
        pd_hat = self.model.predict_proba(applications)[:, 1]
        scores = self.scaler.pd_to_score(pd_hat)
        bands = [self.scaler.band(float(s)) for s in scores]
        return pd.DataFrame(
            {
                "probability_of_default": pd_hat,
                "score": scores.astype(int),
                "band": [b.label for b in bands],
                "decision": [b.decision for b in bands],
                "model_version": self._version,
            }
        )

    def score_one(self, application: dict[str, Any]) -> ScoreResult:
        frame = pd.DataFrame([application])
        row = self.score_batch(frame).iloc[0]
        return ScoreResult(
            probability_of_default=float(row["probability_of_default"]),
            score=int(row["score"]),
            band=str(row["band"]),
            decision=str(row["decision"]),
            model_name=self._model_name,
            model_version=self._version,
            reasons=self.reason_codes(frame),
        )

    @property
    def explainer(self) -> ReasonCodeExplainer:
        """Built once against the loaded pipeline, reused for every request."""
        if self._explainer is None:
            self._explainer = ReasonCodeExplainer(self.model)
        return self._explainer

    def reason_codes(self, frame: pd.DataFrame, top_n: int = 3) -> list[str]:
        """Adverse-action style reasons for the first row of `frame`."""
        return self.explainer.top_reasons(frame, top_n=top_n)


@lru_cache(maxsize=1)
def get_scoring_service() -> ScoringService:
    return ScoringService(AppConfig.load())
