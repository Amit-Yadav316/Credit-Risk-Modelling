"""Risk-specific evaluation: KS, Gini, PSI, decile lift, capture rates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class RiskMetrics:
    """Stateless metric helpers. All probabilities are P(default)."""

    @staticmethod
    def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
        df = df.sort_values("p", ascending=False)
        bads, goods = df["y"].sum(), (1 - df["y"]).sum()
        if bads == 0 or goods == 0:
            return 0.0
        cum_bad = df["y"].cumsum() / bads
        cum_good = (1 - df["y"]).cumsum() / goods
        return float((cum_bad - cum_good).abs().max())

    @staticmethod
    def gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(2 * roc_auc_score(y_true, y_prob) - 1)

    @staticmethod
    def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index between a reference and a current sample."""
        expected, actual = np.asarray(expected, float), np.asarray(actual, float)
        edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        e_pct = np.histogram(expected, bins=edges)[0] / len(expected)
        a_pct = np.histogram(actual, bins=edges)[0] / len(actual)
        e_pct = np.clip(e_pct, 1e-6, None)
        a_pct = np.clip(a_pct, 1e-6, None)
        return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

    @staticmethod
    def decile_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
        df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
        ranks = df["p"].rank(method="first", ascending=False)
        df["decile"] = pd.qcut(ranks, n_bins, labels=range(1, n_bins + 1))
        out = (
            df.groupby("decile", observed=True)
            .agg(accounts=("y", "size"), bads=("y", "sum"), min_pd=("p", "min"), max_pd=("p", "max"))
            .reset_index()
        )
        out["bad_rate"] = out["bads"] / out["accounts"]
        out["cum_bads_pct"] = out["bads"].cumsum() / out["bads"].sum()
        out["cum_accounts_pct"] = out["accounts"].cumsum() / out["accounts"].sum()
        out["lift"] = out["bad_rate"] / (out["bads"].sum() / out["accounts"].sum())
        return out

    @classmethod
    def full_report(
        cls, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
    ) -> dict[str, float]:
        y_pred = (np.asarray(y_prob) >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "gini": cls.gini(y_true, y_prob),
            "ks": cls.ks_statistic(y_true, y_prob),
            "brier": float(brier_score_loss(y_true, y_prob)),
            "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
            "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
            "approval_rate": float(1 - y_pred.mean()),
            "threshold": float(threshold),
        }

    @staticmethod
    def threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float) -> float:
        """Lowest PD cutoff that still captures `target_recall` of defaults.

        Reporting recall without the cutoff that produced it is the mistake most
        credit-risk CVs make, so the cutoff is carried through explicitly.
        """
        order = np.argsort(-np.asarray(y_prob))
        y_sorted = np.asarray(y_true)[order]
        p_sorted = np.asarray(y_prob)[order]
        cum_recall = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
        idx = np.searchsorted(cum_recall, target_recall)
        idx = min(idx, len(p_sorted) - 1)
        return float(p_sorted[idx])
