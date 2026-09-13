import numpy as np
import pandas as pd

from credit_risk.config import AppConfig
from credit_risk.evaluation import PolicyBacktest
from credit_risk.evaluation.policy import DECISION_ORDER


def _backtest() -> PolicyBacktest:
    """ScoringService loads the champion lazily, so no registry is touched here."""
    return PolicyBacktest(AppConfig.load())


def _scores_and_outcomes(rng, approve=680, refer=585, decline=480):
    """Three cohorts whose bad rates rise as the score falls."""
    scores, bad = [], []
    for score, rate, n in [(approve, 0.05, 600), (refer, 0.15, 300), (decline, 0.35, 200)]:
        scores.extend([score] * n)
        bad.extend(rng.binomial(1, rate, n).tolist())
    return np.array(scores, dtype=float), pd.Series(bad)


def test_bands_are_reported_riskiest_last():
    bt = _backtest()
    rng = np.random.default_rng(0)
    scores, y = _scores_and_outcomes(rng)
    report = bt.band_table(scores, y)
    assert report.table["decision"].tolist() == DECISION_ORDER


def test_counts_match_the_cutoffs():
    bt = _backtest()
    rng = np.random.default_rng(1)
    scores, y = _scores_and_outcomes(rng)
    report = bt.band_table(scores, y, approve_above=620, refer_above=550)
    counts = dict(zip(report.table["decision"], report.table["count"], strict=True))
    assert counts["Approve"] == 600
    assert counts["Refer to underwriter"] == 300
    assert counts["Decline"] == 200
    assert report.table["count"].sum() == len(scores)
    assert abs(report.approval_rate - 600 / 1100) < 1e-9


def test_rising_bad_rates_are_monotonic():
    bt = _backtest()
    rng = np.random.default_rng(2)
    scores, y = _scores_and_outcomes(rng)
    report = bt.band_table(scores, y)
    assert report.is_monotonic
    rates = report.table["bad_rate"].tolist()
    assert rates[0] < rates[1] < rates[2]


def test_inverted_bad_rates_are_flagged():
    """A policy that approves the riskiest applicants must not look monotonic."""
    bt = _backtest()
    scores = np.array([700.0] * 200 + [500.0] * 200)
    y = pd.Series([1] * 200 + [0] * 200)
    report = bt.band_table(scores, y)
    assert not report.is_monotonic


def test_cutoffs_move_the_bands():
    bt = _backtest()
    rng = np.random.default_rng(3)
    scores, y = _scores_and_outcomes(rng)
    loose = bt.band_table(scores, y, approve_above=560, refer_above=470)
    tight = bt.band_table(scores, y, approve_above=700, refer_above=600)
    assert loose.approval_rate > tight.approval_rate
    assert tight.approval_rate == 0.0
    assert loose.summary()["approve_above"] == 560


def test_score_quantiles_are_ordered():
    scores = np.linspace(350, 880, 500)
    q = PolicyBacktest.score_quantiles(scores, step=0.25)
    assert q["score"].tolist() == sorted(q["score"].tolist())
    assert len(q) == 3
