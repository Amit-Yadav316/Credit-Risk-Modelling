"""Back-test of the approve / refer / decline policy on a held-out slice.

The training leaderboard says how well the model ranks. This says whether the
cutoffs sitting on top of the model actually separate risk in production terms:
observed bad rates have to rise from Approve through Refer to Decline. A model
with a good AUC and cutoffs in the wrong place is still a bad policy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from credit_risk.config import AppConfig
from credit_risk.data import DataLoader
from credit_risk.scoring.scorecard import ScoreScaler
from credit_risk.scoring.service import ScoringService

logger = logging.getLogger(__name__)

# Riskiest last. Bad rates are expected to increase down this list.
DECISION_ORDER = ["Approve", "Refer to underwriter", "Decline"]


@dataclass(frozen=True)
class PolicyReport:
    """Per-band volumes and observed bad rates at one pair of cutoffs."""

    table: pd.DataFrame
    approve_above: int
    refer_above: int

    @property
    def is_monotonic(self) -> bool:
        """Observed bad rate strictly increasing across Approve, Refer, Decline."""
        rates = [r for r in self.table["bad_rate"].tolist() if not np.isnan(r)]
        return all(a < b for a, b in zip(rates, rates[1:], strict=False))

    @property
    def approval_rate(self) -> float:
        total = self.table["count"].sum()
        approved = self.table.loc[self.table["decision"] == "Approve", "count"].sum()
        return float(approved / total) if total else 0.0

    def summary(self) -> dict[str, object]:
        return {
            "approve_above": self.approve_above,
            "refer_above": self.refer_above,
            "approval_rate": round(self.approval_rate, 4),
            "monotonic": self.is_monotonic,
        }


class PolicyBacktest:
    """Scores a holdout slice with the registry champion and measures each band.

    Scoring is done once. Cutoffs are applied afterwards to the cached scores,
    so re-banding at a different policy costs nothing and the model is never
    re-run just to move a threshold.
    """

    def __init__(self, config: AppConfig, service: ScoringService | None = None) -> None:
        self.config = config
        self.service = service or ScoringService(config)
        self.dc = config.data

    def holdout(self, fraction: float = 0.2) -> pd.DataFrame:
        """The most recent `fraction` of the book, in disbursal-date order."""
        df = DataLoader(self.config).load_raw().sort_values(self.dc.date_col)
        cut = int(round(len(df) * (1.0 - fraction)))
        slice_ = df.iloc[cut:].reset_index(drop=True)
        logger.info(
            "Holdout: %s rows, %s to %s, bad rate %.4f",
            f"{len(slice_):,}",
            slice_[self.dc.date_col].min().date(),
            slice_[self.dc.date_col].max().date(),
            slice_[self.dc.target].mean(),
        )
        return slice_

    def scores(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        """Champion PDs translated to the 300-900 scale, with the outcome."""
        scored = self.service.score_batch(df)
        return scored["score"].to_numpy(), df[self.dc.target].reset_index(drop=True)

    def band_table(
        self,
        scores: np.ndarray,
        y: pd.Series,
        approve_above: int | None = None,
        refer_above: int | None = None,
    ) -> PolicyReport:
        policy = self.config.scorecard.policy
        approve = int(approve_above if approve_above is not None else policy["approve_above"])
        refer = int(refer_above if refer_above is not None else policy["refer_above"])
        scaler = ScoreScaler(
            replace(
                self.config.scorecard,
                policy={"approve_above": approve, "refer_above": refer},
            )
        )

        decisions = pd.Series([scaler.band(float(s)).decision for s in scores], name="decision")
        frame = pd.DataFrame({"decision": decisions, "bad": np.asarray(y, dtype=float)})
        grouped = (
            frame.groupby("decision", observed=True)
            .agg(count=("bad", "size"), bads=("bad", "sum"), bad_rate=("bad", "mean"))
            .reindex(DECISION_ORDER)
            .reset_index()
        )
        grouped["count"] = grouped["count"].fillna(0).astype(int)
        grouped["bads"] = grouped["bads"].fillna(0).astype(int)
        grouped["share_of_book"] = grouped["count"] / max(grouped["count"].sum(), 1)
        return PolicyReport(table=grouped, approve_above=approve, refer_above=refer)

    def run(
        self,
        fraction: float = 0.2,
        approve_above: int | None = None,
        refer_above: int | None = None,
    ) -> PolicyReport:
        scores, y = self.scores(self.holdout(fraction))
        return self.band_table(scores, y, approve_above, refer_above)

    @staticmethod
    def score_quantiles(scores: np.ndarray, step: float = 0.1) -> pd.DataFrame:
        """Score distribution, to pick cutoffs against a target approval rate."""
        qs = np.arange(step, 1.0, step)
        return pd.DataFrame(
            {"quantile": np.round(qs, 2), "score": np.quantile(scores, qs).round(0).astype(int)}
        )
