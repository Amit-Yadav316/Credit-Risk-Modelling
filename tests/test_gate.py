from dataclasses import replace

from credit_risk.config import AppConfig, PromotionConfig
from credit_risk.training import DeployabilityGate

STRONG = {"test": {"roc_auc": 0.71}, "oot": {"roc_auc": 0.70, "ks": 0.31}, "score_psi": 0.05}


def _gate(**thresholds) -> DeployabilityGate:
    cfg = AppConfig.load()
    return DeployabilityGate(replace(cfg, promotion=PromotionConfig(**thresholds)))


def test_a_strong_candidate_passes():
    gate = _gate(min_oot_auc=0.68, min_oot_ks=0.28, max_score_psi=0.25, max_test_oot_auc_gap=0.05)
    assert gate.passes(**STRONG)
    assert gate.failures(**STRONG) == ()


def test_each_threshold_can_reject_and_says_why():
    gate = _gate(min_oot_auc=0.68, min_oot_ks=0.28, max_score_psi=0.25, max_test_oot_auc_gap=0.05)

    weak_auc = {**STRONG, "oot": {"roc_auc": 0.60, "ks": 0.31}}
    assert not gate.passes(**weak_auc)
    assert "oot_auc" in gate.failures(**weak_auc)[0]

    weak_ks = {**STRONG, "oot": {"roc_auc": 0.70, "ks": 0.11}}
    assert "oot_ks" in gate.failures(**weak_ks)[0]

    drifted = {**STRONG, "score_psi": 0.40}
    assert "score_psi" in gate.failures(**drifted)[0]

    overfit = {**STRONG, "test": {"roc_auc": 0.90}}
    assert "test_oot_auc_gap" in gate.failures(**overfit)[0]


def test_every_failed_threshold_is_reported_not_just_the_first():
    gate = _gate(min_oot_auc=0.68, min_oot_ks=0.28, max_score_psi=0.25, max_test_oot_auc_gap=0.05)
    hopeless = {"test": {"roc_auc": 0.90}, "oot": {"roc_auc": 0.55, "ks": 0.10}, "score_psi": 0.9}
    assert len(gate.failures(**hopeless)) == 4


def test_a_null_threshold_switches_that_check_off():
    gate = _gate(min_oot_auc=None, min_oot_ks=None, max_score_psi=None, max_test_oot_auc_gap=None)
    hopeless = {"test": {"roc_auc": 0.90}, "oot": {"roc_auc": 0.55, "ks": 0.10}, "score_psi": 0.9}
    assert gate.passes(**hopeless)


def test_thresholds_come_from_the_shipped_config():
    """The config must carry real thresholds, not leave the gate inert by accident."""
    cfg = AppConfig.load()
    assert cfg.promotion.min_oot_auc is not None
    assert cfg.promotion.min_oot_ks is not None
    assert cfg.promotion.max_score_psi is not None
