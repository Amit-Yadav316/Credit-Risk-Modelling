"""Deployability gate, driven by configs/config.yaml rather than literals in code."""
from __future__ import annotations

from credit_risk.config import AppConfig


class DeployabilityGate:
    """Checks one candidate against the promotion thresholds in the config.

    The thresholds are config, not code, because what counts as deployable is a
    portfolio decision: a book with a 15% bad rate and no bureau-grade features
    does not separate risk as sharply as a prime book, and a gate written for one
    silently rejects every sound model on the other. Hard-coding them also hides
    the most useful fact about a rejected candidate, which is *which* threshold it
    missed and by how much, so `failures` returns that rather than a bare bool.

    Set any threshold to null in the config to switch that check off.
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.promotion

    def failures(
        self, *, test: dict[str, float], oot: dict[str, float], score_psi: float
    ) -> tuple[str, ...]:
        """Human-readable reasons this candidate cannot be promoted. Empty means it can."""
        gap = abs(test["roc_auc"] - oot["roc_auc"])
        checks = (
            ("oot_auc", oot["roc_auc"], self.cfg.min_oot_auc, "min"),
            ("oot_ks", oot["ks"], self.cfg.min_oot_ks, "min"),
            ("score_psi", score_psi, self.cfg.max_score_psi, "max"),
            ("test_oot_auc_gap", gap, self.cfg.max_test_oot_auc_gap, "max"),
        )
        reasons: list[str] = []
        for name, value, limit, kind in checks:
            if limit is None:
                continue
            if kind == "min" and value < limit:
                reasons.append(f"{name} {value:.4f} < {limit}")
            elif kind == "max" and value >= limit:
                reasons.append(f"{name} {value:.4f} >= {limit}")
        return tuple(reasons)

    def passes(self, *, test: dict[str, float], oot: dict[str, float], score_psi: float) -> bool:
        return not self.failures(test=test, oot=oot, score_psi=score_psi)
