"""Second tree challenger: XGBoost with isotonic calibration on top.

Included as a control rather than an expectation. LightGBM and XGBoost differ
mainly in how they grow trees -- LightGBM leaf-wise, XGBoost level-wise with
`grow_policy="depthwise"` -- and at the shallow depths this book supports that
difference is small. If it wins by a few ten-thousandths of AUC, that is noise,
not a finding; the leaderboard exists to make that judgement possible instead of
crowning whichever model was run last.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from credit_risk.features import FeatureBuilder
from credit_risk.models.base import RiskModel
from credit_risk.models.calibration import PriorCorrectedClassifier


class XGBoostChallenger(RiskModel):
    """Level-wise boosted trees, calibrated, on ordinal-encoded categoricals."""

    name = "xgboost"

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
        booster = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=params.pop("n_estimators", 400),
            tree_method="hist",
            random_state=self.config.data.random_state,
            n_jobs=-1,
            verbosity=0,
            **params,
        )
        estimator = (
            CalibratedClassifierCV(booster, method="isotonic", cv=3) if calibrate else booster
        )
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
        """Deliberately the same shape and ranges as the LightGBM space.

        Giving the two boosters equivalent budgets is what makes the comparison
        mean anything: a win driven by one search space being richer than the
        other says nothing about the algorithms.
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
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
        # get_score returns a sparse dict keyed "f0", "f1", ... and omits features
        # the booster never split on, so missing keys default to zero gain.
        gains = np.mean(
            [
                np.asarray(
                    [
                        b.get_booster().get_score(importance_type="gain").get(f"f{i}", 0.0)
                        for i in range(len(names))
                    ]
                )
                for b in boosters
            ],
            axis=0,
        )
        return (
            pd.DataFrame({"feature": names, "importance": gains})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
