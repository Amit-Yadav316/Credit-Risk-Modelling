"""Adverse-action reason codes, with the explainer matched to the model.

Under ECOA a declined US applicant must be told the principal reasons, so a
scorecard needs per-application attributions, not just global importance. Getting
that right is mostly about choosing the explainer correctly:

* A tree model is explained exactly and in microseconds by `shap.TreeExplainer`,
  which needs no background sample at all under its default path-dependent
  perturbation.
* A linear model needs no sampling either. After a StandardScaler the training
  mean is zero, so the exact SHAP value for feature j is simply `coef_j * x_j`.
* Only when neither applies is a sampling explainer required, and that one does
  need a real background distribution.

The previous implementation called `shap.Explainer(predict_proba, row)` with the
single row being scored as its own background. Explaining a point against itself
returns exactly zero for every feature, so the reason list was always empty and
the failure was invisible: the caller caught the exception that never came.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReasonCodeExplainer:
    """Turns one scored application into the handful of features driving its risk.

    Attributions are on the model's log-odds margin. Isotonic calibration is
    monotone, so it cannot reorder contributions, which is why explaining the
    margin is sound even though the served PD is calibrated.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.names = [
            self._humanise(n) for n in pipeline.named_steps["prep"].get_feature_names_out()
        ]
        self.estimator = self._unwrap(pipeline.named_steps["clf"])

    @staticmethod
    def _humanise(name: str) -> str:
        """`num__debt_to_income` -> `debt_to_income`, `cat__state_CA` -> `state = CA`."""
        bare = name.split("__", 1)[-1]
        for prefix in ("loan_purpose_", "employment_type_", "residence_type_", "state_", "utilisation_band_"):
            if bare.startswith(prefix):
                return f"{prefix.rstrip('_')} = {bare[len(prefix):]}"
        return bare

    @staticmethod
    def _unwrap(clf):
        """Strip the prior-correction wrapper, which only shifts the intercept."""
        return getattr(clf, "estimator_", clf)

    def contributions(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-feature push towards default for the first row of `frame`."""
        transformed = self.pipeline[:-1].transform(frame)
        # One-hot encoding leaves the logistic pipeline sparse; every path below
        # wants a dense row.
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed)

        models = self._inner_models(self.estimator)
        backends = [self._tree_backend(m) for m in models]

        if any(backends):
            # Average across the calibration folds: each fold fitted its own model
            values = [
                self._tree_contributions(m, backend, transformed)
                for m, backend in zip(models, backends, strict=True)
                if backend
            ]
            return np.mean(values, axis=0)

        coef = getattr(self.estimator, "coef_", None)
        if coef is not None:
            # Exact SHAP for a linear model on standardised inputs
            return np.ravel(coef) * transformed[0].astype(float)

        raise TypeError(f"No reason-code strategy for {type(self.estimator).__name__}")

    @staticmethod
    def _inner_models(est) -> list:
        """Every fitted base model inside `est`, unwrapping a calibration wrapper."""
        inner = getattr(est, "calibrated_classifiers_", None)
        if inner is not None:
            return [getattr(c, "estimator", None) for c in inner]
        return [est]

    @staticmethod
    def _tree_backend(model) -> str | None:
        """Which tree library this is, or None if it is not a tree model.

        Each library exposes a different handle, and dispatching on the handle
        rather than on an isinstance check keeps this working whether or not the
        model arrived wrapped in calibration.
        """
        if model is None:
            return None
        if hasattr(model, "model_") and hasattr(model, "cat_features"):
            return "catboost"
        if hasattr(model, "booster_"):
            return "lightgbm"
        if hasattr(model, "get_booster"):
            return "xgboost"
        return None

    @staticmethod
    def _tree_contributions(model, backend: str, transformed: np.ndarray) -> np.ndarray:
        if backend == "catboost":
            # CatBoost needs a Pool carrying the categorical positions, and its
            # ShapValues output appends the expected value as a final column.
            from catboost import Pool

            pool = Pool(transformed, cat_features=list(model.cat_features or []))
            shap_values = np.asarray(
                model.model_.get_feature_importance(type="ShapValues", data=pool)
            )
            return shap_values[0, :-1]

        import shap

        dense = transformed.astype(float)
        values = shap.TreeExplainer(model).shap_values(dense)
        return np.asarray(values).reshape(-1, dense.shape[1])[0]

    def top_reasons(self, frame: pd.DataFrame, top_n: int = 3) -> list[str]:
        """The `top_n` features pushing this application towards default.

        Only positive contributions are returned. An application that nothing
        pushes towards default has no adverse reasons, and inventing some would be
        worse than returning none.
        """
        try:
            values = self.contributions(frame)
        except Exception as exc:
            logger.warning("Reason codes unavailable: %s", exc)
            return []

        order = np.argsort(-values)[:top_n]
        return [f"{self.names[i]} pushed risk up" for i in order if values[i] > 0]
