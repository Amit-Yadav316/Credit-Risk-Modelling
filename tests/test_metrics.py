import numpy as np

from credit_risk.evaluation.metrics import RiskMetrics


def test_ks_is_one_for_perfect_separation():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.9] * 50)
    assert RiskMetrics.ks_statistic(y, p) > 0.99


def test_psi_is_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert RiskMetrics.psi(x, x) < 1e-6


def test_psi_detects_a_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(1.2, 1, 5000)
    assert RiskMetrics.psi(a, b) > 0.25


def test_threshold_for_recall_is_achievable():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.2, 4000)
    p = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.2, 4000), 0.01, 0.99)
    cutoff = RiskMetrics.threshold_for_recall(y, p, 0.85)
    achieved = ((p >= cutoff) & (y == 1)).sum() / y.sum()
    assert achieved >= 0.84
