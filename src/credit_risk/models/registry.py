from __future__ import annotations

from credit_risk.config import AppConfig
from credit_risk.models.base import RiskModel
from credit_risk.models.catboost_model import CatBoostChallenger
from credit_risk.models.lightgbm_model import LightGBMChallenger
from credit_risk.models.logistic import LogisticScorecard
from credit_risk.models.xgboost_model import XGBoostChallenger

MODEL_REGISTRY: dict[str, type[RiskModel]] = {
    LogisticScorecard.name: LogisticScorecard,
    LightGBMChallenger.name: LightGBMChallenger,
    XGBoostChallenger.name: XGBoostChallenger,
    CatBoostChallenger.name: CatBoostChallenger,
}


def build_model(name: str, config: AppConfig) -> RiskModel:
    try:
        return MODEL_REGISTRY[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}") from exc
