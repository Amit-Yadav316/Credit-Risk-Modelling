"""The synthetic fixture must satisfy the same contract as the real extract.

Without this, changing features.numeric in the config leaves the generator behind
and CI only finds out minutes into a training run, with a KeyError from inside a
fitted pipeline.
"""
import pandas as pd

from credit_risk.config import AppConfig
from credit_risk.data import DataLoader, SyntheticBookGenerator
from credit_risk.features import FeatureBuilder

CFG = AppConfig.load()
BOOK = SyntheticBookGenerator(n=3_000, seed=7).generate()


def test_every_raw_feature_the_config_asks_for_is_generated():
    derived = set(CFG.features.derived)
    required = [
        c
        for c in CFG.features.numeric + CFG.features.categorical
        if c not in derived
    ]
    missing = [c for c in required if c not in BOOK.columns]
    assert not missing, f"generator is missing configured features: {missing}"


def test_split_columns_are_generated():
    for col in (CFG.data.target, CFG.data.date_col, CFG.data.id_col):
        assert col in BOOK.columns, f"generator is missing {col}"


def test_the_feature_pipeline_transforms_the_fixture():
    """The real check: FeatureBuilder must run on this book without a KeyError."""
    out = FeatureBuilder.from_config(CFG).fit_transform(BOOK)
    assert list(out.columns) == CFG.features.all_numeric + CFG.features.all_categorical
    assert len(out) == len(BOOK)
    assert out[CFG.features.all_numeric].notna().all().all()


def test_no_protected_characteristic_is_generated():
    """Age is barred from the scorecard, so it is not in the fixture either."""
    for banned in ("age", "sex", "gender", "race", "marital_status"):
        assert banned not in BOOK.columns


def test_the_book_carries_signal_and_a_plausible_bad_rate():
    assert 0.05 < BOOK.default_flag.mean() < 0.30
    # A riskier half must actually default more, or the fixture teaches nothing
    high_lti = BOOK.loan_amount / BOOK.annual_income > (BOOK.loan_amount / BOOK.annual_income).median()
    assert BOOK.loc[high_lti, "default_flag"].mean() > BOOK.loc[~high_lti, "default_flag"].mean()


def test_delinquency_recency_covers_both_the_sentinel_and_recent_cases():
    never = (BOOK.months_since_delinq == 999).mean()
    assert 0.3 < never < 0.7, f"sentinel share {never:.2f} is not representative"
    assert (BOOK.months_since_delinq < 24).any()


def test_the_fixture_survives_the_out_of_time_split():
    bundle = DataLoader(CFG).split(BOOK)
    assert len(bundle.X_oot) > 0
    assert len(bundle.X_train) > len(bundle.X_test)
    for frame in (bundle.X_train, bundle.X_test, bundle.X_oot):
        assert CFG.data.target not in frame.columns
        assert CFG.data.date_col not in frame.columns


def test_generation_is_reproducible():
    a = SyntheticBookGenerator(n=500, seed=11).generate()
    b = SyntheticBookGenerator(n=500, seed=11).generate()
    pd.testing.assert_frame_equal(a, b)


def test_the_fixture_is_separable_enough_for_the_gate_to_promote():
    """CI trains on this book and then asserts the champion alias loads.

    That only works if a model fitted on the fixture clears the promotion floors
    in the config. If this fails, CI does not have a pipeline bug, it has an
    unrealistic fixture: raise SIGNAL_SCALE in credit_risk.data.synthetic.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    book = SyntheticBookGenerator(n=20_000, seed=5).generate()
    y = book[CFG.data.target]
    X = FeatureBuilder.from_config(CFG).fit_transform(book)[CFG.features.all_numeric]

    cut = int(len(X) * 0.7)
    model = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, solver="liblinear"))]
    ).fit(X.iloc[:cut], y.iloc[:cut])
    auc = roc_auc_score(y.iloc[cut:], model.predict_proba(X.iloc[cut:])[:, 1])

    floor = CFG.promotion.min_oot_auc
    assert floor is not None
    assert auc >= floor, f"fixture yields AUC {auc:.4f}, below the promotion floor {floor}"
