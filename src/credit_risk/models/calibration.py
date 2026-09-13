"""Prior correction for resampled training sets."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class PriorCorrectedClassifier(ClassifierMixin, BaseEstimator):
    """Shifts predicted log-odds back to the true population bad rate.

    Any resampling (SMOTE, Tomek, class weights) changes the base rate the
    classifier sees, so its output is a probability under the *resampled* prior,
    not the book. Ranking is unaffected, which is why AUC and KS look fine and the
    problem goes unnoticed, but every PD is inflated and the 300-900 scale sits on
    top of PD, so the whole scorecard shifts down.

    The fix is the standard offset used in choice-based sampling:

        logit_corrected = logit_model + ln(p_true / (1 - p_true))
                                      - ln(p_sample / (1 - p_sample))

    where p_sample is the bad rate the estimator actually trained on.
    """

    def __init__(self, estimator, true_prior: float | None = None) -> None:
        self.estimator = estimator
        self.true_prior = true_prior

    def fit(self, X, y) -> PriorCorrectedClassifier:
        y = np.asarray(y)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        self.sample_prior_ = float(np.mean(y))
        if self.true_prior is None:
            self.offset_ = 0.0
        else:
            p_t = float(np.clip(self.true_prior, 1e-6, 1 - 1e-6))
            p_s = float(np.clip(self.sample_prior_, 1e-6, 1 - 1e-6))
            self.offset_ = np.log(p_t / (1 - p_t)) - np.log(p_s / (1 - p_s))
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.clip(self.estimator_.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        logit = np.log(p / (1 - p)) + self.offset_
        corrected = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1 - corrected, corrected])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def coef_(self):
        return self.estimator_.coef_
