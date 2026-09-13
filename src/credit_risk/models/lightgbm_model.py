"""Tree challenger: LightGBM with isotonic calibration on top."""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from credit_risk.features import FeatureBuilder
from credit_risk.models.base import RiskModel
from credit_risk.models.calibration import PriorCorrectedClassifier


class LightGBMChallenger(RiskModel):
    """Gradient-boosted challenger to the scorecard.

    Two deliberate choices worth defending in an interview:

    1. No SMOTE. Boosted trees handle skew through `scale_pos_weight`, and
       synthetic minority points distort the split gains and wreck calibration.
    2. Isotonic calibration wraps the booster, because raw LightGBM scores rank
       well but are not probabilities, and the 300-900 scale is a log-odds
       transform that only means anything on a calibrated PD.
    """

    name = "lightgbm"

    def _preprocessor(self) -> ColumnTransformer:
        f = self.config.features
        return ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), f.all_numeric),
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "ord",
                                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                            ),
                        ]
                    ),
                    f.all_categorical,
                ),
            ],
            remainder="drop",
        )

    def build(self, params: dict[str, Any] | None = None) -> Pipeline:
        params = dict(params or {})
        calibrate = params.pop("calibrate", True)
        booster = LGBMClassifier(
            objective="binary",
            n_estimators=params.pop("n_estimators", 400),
            random_state=self.config.data.random_state,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        estimator = (
            CalibratedClassifierCV(booster, method="isotonic", cv=3) if calibrate else booster
        )
        # Isotonic already re-maps to the observed rate of the data it is fitted
        # on, so the prior correction only applies to the uncalibrated branch
        if not calibrate:
            estimator = PriorCorrectedClassifier(estimator, true_prior=self.true_prior)
        return Pipeline(
            steps=[
                ("features", FeatureBuilder.from_config(self.config)),
                ("prep", self._preprocessor()),
                ("clf", estimator),
            ]
        )

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 12.0),
        }

    def feature_importance(self) -> pd.DataFrame:
        if self.pipeline is None:
            return super().feature_importance()
        names = list(self.pipeline.named_steps["prep"].get_feature_names_out())
        clf = self.pipeline.named_steps["clf"]
        boosters = (
            [c.estimator for c in clf.calibrated_classifiers_]
            if isinstance(clf, CalibratedClassifierCV)
            else [clf]
        )
        gains = np.mean([b.booster_.feature_importance(importance_type="gain") for b in boosters], axis=0)
        return (
            pd.DataFrame({"feature": names, "importance": gains})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
