"""Typed configuration loaded once from configs/config.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    target: str
    date_col: str
    id_col: str
    oot_months: int
    test_size: float
    random_state: int


@dataclass(frozen=True)
class FeatureConfig:
    numeric: list[str]
    categorical: list[str]
    derived: list[str]

    @property
    def all_numeric(self) -> list[str]:
        return list(dict.fromkeys(self.numeric + [c for c in self.derived if c != "utilisation_band"]))

    @property
    def all_categorical(self) -> list[str]:
        extra = ["utilisation_band"] if "utilisation_band" in self.derived else []
        return list(dict.fromkeys(self.categorical + extra))


@dataclass(frozen=True)
class ScorecardConfig:
    base_score: int
    base_odds: float
    pdo: int
    min_score: int
    max_score: int
    policy: dict[str, int]


@dataclass(frozen=True)
class TrainingConfig:
    n_trials: int
    cv_folds: int
    primary_metric: str
    models: list[str]
    tune_sample_size: int | None = None


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment: str
    registered_model: str


@dataclass(frozen=True)
class PromotionConfig:
    """Deployability thresholds. None switches a check off."""

    min_oot_auc: float | None = None
    min_oot_ks: float | None = None
    max_score_psi: float | None = None
    max_test_oot_auc_gap: float | None = None


@dataclass(frozen=True)
class MonitoringConfig:
    psi_threshold: float
    psi_bins: int


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    features: FeatureConfig
    scorecard: ScorecardConfig
    training: TrainingConfig
    mlflow: MLflowConfig
    monitoring: MonitoringConfig
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(
            data=DataConfig(**cfg["data"]),
            features=FeatureConfig(**cfg["features"]),
            scorecard=ScorecardConfig(**cfg["scorecard"]),
            training=TrainingConfig(**cfg["training"]),
            mlflow=MLflowConfig(**cfg["mlflow"]),
            monitoring=MonitoringConfig(**cfg["monitoring"]),
            promotion=PromotionConfig(**(cfg.get("promotion") or {})),
            raw=cfg,
        )

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p
