"""Reason codes must be real attributions, not a silently empty list.

The bug this guards against: passing the row being explained as its own SHAP
background returns exactly zero for every feature, so the reason list came back
empty for every application and nothing raised.
"""
import numpy as np
import pandas as pd
import pytest

from credit_risk.config import AppConfig
from credit_risk.data import SyntheticBookGenerator
from credit_risk.models import build_model
from credit_risk.scoring.explainer import ReasonCodeExplainer

CFG = AppConfig.load()
BOOK = SyntheticBookGenerator(n=2_000, seed=3).generate()
X = BOOK.drop(columns=[CFG.data.target, CFG.data.date_col, CFG.data.id_col])
y = BOOK[CFG.data.target]

RISKY = X.assign(
    loan_amount=34_000, annual_income=26_000, interest_rate=26.5,
    credit_utilisation=1.4, bureau_score=625, debt_to_income=38.0,
    months_since_delinq=2.0, num_delinquent_accounts=4, public_records=2, enquiries_6m=6,
).head(1)


SMALL_PARAMS = {
    "logistic": {"C": 0.1, "penalty": "l2"},
    "lightgbm": {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 8, "max_depth": 3},
    "xgboost": {"n_estimators": 40, "learning_rate": 0.1, "max_depth": 3},
    "catboost": {"iterations": 40, "learning_rate": 0.1, "depth": 3},
}
TREE_MODELS = ["lightgbm", "xgboost", "catboost"]
ALL_MODELS = ["logistic", *TREE_MODELS]


def _fitted(name: str):
    return build_model(name, CFG).set_prior(float(y.mean())).fit(X, y, SMALL_PARAMS[name]).pipeline


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_contributions_are_not_all_zero(model_name):
    """The exact symptom of the degenerate-background bug."""
    explainer = ReasonCodeExplainer(_fitted(model_name))
    values = explainer.contributions(RISKY)
    assert np.abs(values).max() > 0, "every attribution is zero, background is degenerate"


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_one_contribution_per_model_feature(model_name):
    explainer = ReasonCodeExplainer(_fitted(model_name))
    assert len(explainer.contributions(RISKY)) == len(explainer.names)


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_a_risky_application_gets_reasons(model_name):
    explainer = ReasonCodeExplainer(_fitted(model_name))
    reasons = explainer.top_reasons(RISKY, top_n=3)
    assert 0 < len(reasons) <= 3
    assert all(r.endswith("pushed risk up") for r in reasons)


@pytest.mark.parametrize("model_name", TREE_MODELS)
def test_every_tree_backend_is_recognised(model_name):
    """An unrecognised backend silently falls through to an empty reason list."""
    explainer = ReasonCodeExplainer(_fitted(model_name))
    models = explainer._inner_models(explainer.estimator)
    assert models and all(explainer._tree_backend(m) is not None for m in models)


def test_reasons_are_ordered_by_contribution():
    explainer = ReasonCodeExplainer(_fitted("lightgbm"))
    values = explainer.contributions(RISKY)
    reasons = explainer.top_reasons(RISKY, top_n=3)
    ranked = [explainer.names[i] for i in np.argsort(-values) if values[i] > 0][: len(reasons)]
    assert [r.replace(" pushed risk up", "") for r in reasons] == ranked


def test_feature_names_are_readable():
    explainer = ReasonCodeExplainer(_fitted("logistic"))
    assert not any(n.startswith(("num__", "cat__")) for n in explainer.names)


def test_an_unexplainable_model_returns_no_reasons_instead_of_raising():
    """Serving must degrade to no reasons, never fail the score."""

    class Opaque:
        pass

    explainer = ReasonCodeExplainer(_fitted("logistic"))
    explainer.estimator = Opaque()
    with pytest.raises(TypeError):
        explainer.contributions(RISKY)
    assert explainer.top_reasons(RISKY) == []


def test_explaining_a_batch_uses_the_first_row():
    explainer = ReasonCodeExplainer(_fitted("lightgbm"))
    batch = pd.concat([RISKY, X.head(3)], ignore_index=True)
    assert len(explainer.contributions(batch)) == len(explainer.names)
