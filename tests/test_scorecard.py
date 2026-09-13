import numpy as np

from credit_risk.config import ScorecardConfig
from credit_risk.scoring.scorecard import ScoreScaler

CFG = ScorecardConfig(
    base_score=600, base_odds=50.0, pdo=20, min_score=300, max_score=900,
    policy={"approve_above": 700, "refer_above": 620},
)


def test_pdo_doubles_the_odds():
    scaler = ScoreScaler(CFG)
    pd_600 = float(scaler.score_to_pd(600))
    pd_620 = float(scaler.score_to_pd(620))
    odds_600 = (1 - pd_600) / pd_600
    odds_620 = (1 - pd_620) / pd_620
    assert np.isclose(odds_620 / odds_600, 2.0, rtol=1e-6)


def test_score_is_monotone_decreasing_in_pd():
    scaler = ScoreScaler(CFG)
    pds = np.linspace(0.01, 0.9, 50)
    scores = scaler.pd_to_score(pds)
    assert np.all(np.diff(scores) <= 0)


def test_scores_are_clipped_to_band():
    scaler = ScoreScaler(CFG)
    scores = scaler.pd_to_score(np.array([1e-9, 0.5, 1 - 1e-9]))
    assert scores.min() >= 300 and scores.max() <= 900


def test_policy_bands():
    scaler = ScoreScaler(CFG)
    assert scaler.band(750).decision == "Approve"
    assert scaler.band(650).decision == "Refer to underwriter"
    assert scaler.band(500).decision == "Decline"


def test_prior_correction_recovers_the_true_base_rate():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from credit_risk.models.calibration import PriorCorrectedClassifier

    rng = np.random.default_rng(7)
    n = 20_000
    x = rng.normal(size=(n, 1))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-2.5 + 1.4 * x[:, 0]))))
    true_prior = y.mean()

    # Simulate resampling: keep every bad, keep a third of the goods
    bad_idx = np.flatnonzero(y == 1)
    good_idx = rng.choice(np.flatnonzero(y == 0), size=len(bad_idx) * 2, replace=False)
    keep = np.concatenate([bad_idx, good_idx])

    naive = LogisticRegression().fit(x[keep], y[keep])
    corrected = PriorCorrectedClassifier(LogisticRegression(), true_prior=true_prior).fit(x[keep], y[keep])

    naive_mean = naive.predict_proba(x)[:, 1].mean()
    corrected_mean = corrected.predict_proba(x)[:, 1].mean()
    assert naive_mean > true_prior * 1.8
    assert abs(corrected_mean - true_prior) < 0.02
