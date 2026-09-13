"""PSI-based stability checks on features and on the score distribution."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from credit_risk.config import AppConfig
from credit_risk.evaluation.metrics import RiskMetrics


@dataclass
class DriftReport:
    score_psi: float
    feature_psi: pd.DataFrame
    threshold: float

    @property
    def is_stable(self) -> bool:
        return self.score_psi < self.threshold

    @property
    def unstable_features(self) -> list[str]:
        return self.feature_psi.loc[self.feature_psi["psi"] >= self.threshold, "feature"].tolist()

    def summary(self) -> dict[str, object]:
        return {
            "score_psi": round(self.score_psi, 4),
            "stable": self.is_stable,
            "unstable_features": self.unstable_features,
        }


class DriftMonitor:
    """Compares a reference window against a current window.

    Convention used across the industry: PSI under 0.10 is no shift, 0.10 to
    0.25 needs watching, above 0.25 means the population has moved enough that
    the scorecard should be re-fitted or re-calibrated.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.threshold = config.monitoring.psi_threshold
        self.bins = config.monitoring.psi_bins

    def compare(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        reference_scores,
        current_scores,
    ) -> DriftReport:
        rows = []
        for col in reference.select_dtypes(include="number").columns:
            if col in current.columns:
                rows.append(
                    {
                        "feature": col,
                        "psi": RiskMetrics.psi(
                            reference[col].dropna().to_numpy(),
                            current[col].dropna().to_numpy(),
                            self.bins,
                        ),
                    }
                )
        feature_psi = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
        score_psi = RiskMetrics.psi(reference_scores, current_scores, self.bins)
        return DriftReport(score_psi=score_psi, feature_psi=feature_psi, threshold=self.threshold)
