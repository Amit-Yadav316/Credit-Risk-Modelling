"""PD to 300-900 score translation using points-to-double-the-odds."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from credit_risk.config import ScorecardConfig


@dataclass(frozen=True)
class ScoreBand:
    label: str
    decision: str
    lower: int
    upper: int


class ScoreScaler:
    """Score = offset + factor * ln(good:bad odds).

    factor = PDO / ln(2) so every `pdo` points doubles the odds of being good.
    With base_score 600 at 50:1 odds and PDO 20, a 620 applicant is twice as
    likely to stay current as a 600 applicant. That property is the whole reason
    lenders use this scale instead of showing a raw probability.
    """

    def __init__(self, config: ScorecardConfig) -> None:
        self.cfg = config
        self.factor = config.pdo / math.log(2)
        self.offset = config.base_score - self.factor * math.log(config.base_odds)

    def pd_to_score(self, pd_values: np.ndarray | float) -> np.ndarray:
        p = np.clip(np.asarray(pd_values, dtype=float), 1e-6, 1 - 1e-6)
        odds = (1 - p) / p
        score = self.offset + self.factor * np.log(odds)
        return np.clip(np.round(score), self.cfg.min_score, self.cfg.max_score)

    def score_to_pd(self, score: np.ndarray | float) -> np.ndarray:
        s = np.asarray(score, dtype=float)
        odds = np.exp((s - self.offset) / self.factor)
        return 1.0 / (1.0 + odds)

    def band(self, score: float) -> ScoreBand:
        approve = self.cfg.policy["approve_above"]
        refer = self.cfg.policy["refer_above"]
        if score >= approve:
            return ScoreBand("Low risk", "Approve", approve, self.cfg.max_score)
        if score >= refer:
            return ScoreBand("Moderate risk", "Refer to underwriter", refer, approve - 1)
        return ScoreBand("High risk", "Decline", self.cfg.min_score, refer - 1)
