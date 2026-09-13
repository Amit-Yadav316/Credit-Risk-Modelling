"""Regulator-friendly baseline: WOE-free logistic on scaled and encoded inputs."""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.features import FeatureBuilder
from credit_risk.models.base import RiskModel
from credit_risk.models.calibration import PriorCorrectedClassifier


class LogisticScorecard(RiskModel):
    """Champion-by-default model: monotone, explainable, easy to sign off.

    Trained on the real book with no resampling and no class reweighting. At a
    14.8% bad rate the imbalance is not severe, and every alternative was measured
    on the Lending Club training window against an untouched training set: 150k
    rows, identical preprocessor, solver and C, scored on the in-time test and
    out-of-time slices.

        sampler                  fit     OOT AUC   OOT KS   Brier   mean PD
        none                     5.2s     0.6903   0.2772   0.1197   0.1249
        class_weight=balanced    5.2s     0.6910   0.2788   0.1962   0.4202
        RandomUnderSampler 0.5   3.1s     0.6896   0.2773   0.1198   0.1249
        RandomOverSampler 0.5    4.7s     0.6911   0.2790   0.1197   0.1243
        SMOTE 0.5                8.9s     0.6861   0.2713   0.1199   0.1238
        SMOTETomek 0.5         258.0s     0.6861   0.2711   0.1199   0.1237

    Interpolating minority neighbours ranked *worse* than leaving the book alone,
    for fifty times the compute, so it is gone. class_weight="balanced" ranks a
    hair higher but predicts a mean PD of 0.42 against an observed 0.148. The
    300-900 scale is a log-odds transform of PD, so tuning on AUC alone would have
    bought four ten-thousandths of AUC by wrecking the score scale; it is out of
    the search space for that reason.

    Imbalance is handled where it belongs, at the decision cutoff, which
    RiskMetrics.threshold_for_recall sets explicitly on the test slice.
    """

    name = "logistic"

    def _preprocessor(self) -> ColumnTransformer:
        f = self.config.features
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                    ),
                    f.all_numeric,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
                        ]
                    ),
                    f.all_categorical,
                ),
            ],
            remainder="drop",
        )

    def build(self, params: dict[str, Any] | None = None) -> Pipeline:
        params = dict(params or {})
        return Pipeline(
            steps=[
                ("features", FeatureBuilder.from_config(self.config)),
                ("prep", self._preprocessor()),
                (
                    "clf",
                    PriorCorrectedClassifier(
                        LogisticRegression(
                            max_iter=2000,
                            # liblinear, not saga. Both cover the l1/l2 search space
                            # this model tunes over, but on 470k scaled rows saga
                            # grinds through ~1400 stochastic epochs to land on the
                            # same coefficients liblinear reaches by coordinate
                            # descent in seconds. Measured on the Lending Club book:
                            # 95.3s and AUC 0.6782 against 0.6s and AUC 0.6783.
                            solver="liblinear",
                            random_state=self.config.data.random_state,
                            **params,
                        ),
                        # Nothing resamples this pipeline now, so the offset
                        # computes to zero and this is a pass-through. It stays as
                        # the guard it was written to be: if a sampler is ever put
                        # back, the PD scale gets corrected instead of silently
                        # inflating.
                        true_prior=self.true_prior,
                    ),
                ),
            ]
        )

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Regularisation only.

        class_weight and sampling_strategy are deliberately absent; the class
        docstring carries the measurements. Both traded a calibrated PD for AUC
        noise, and the scorecard is built on PD.
        """
        return {
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        }

    def feature_importance(self) -> pd.DataFrame:
        if self.pipeline is None:
            return super().feature_importance()
        names = self.pipeline.named_steps["prep"].get_feature_names_out()
        coefs = self.pipeline.named_steps["clf"].coef_.ravel()
        return (
            pd.DataFrame({"feature": names, "importance": np.abs(coefs), "coefficient": coefs})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
