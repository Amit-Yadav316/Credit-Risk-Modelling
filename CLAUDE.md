# Working notes for Claude Code

## What this repo is

A credit risk underwriting engine: probability-of-default model, champion-challenger
setup, MLflow tracking and registry, FastAPI scoring service, Streamlit console.
Scores are on a 300-900 scale via a points-to-double-the-odds transform.

It runs end to end as shipped. `make data && make train` works on the synthetic
book. Tests pass, ruff is clean. Do not restructure it before you have run it.

## Conventions to follow

- Class-based services, one responsibility each. New logic goes in a class under
  `src/credit_risk/<area>/`, not in a script and not in the pipeline entry point
- Config lives in `configs/config.yaml` and is read through `AppConfig`. No magic
  numbers in code, no hard-coded column names outside `FeatureBuilder`
- Anything that touches features must live inside the sklearn pipeline so the
  MLflow artifact is the whole decision function. Never recompute a feature by
  hand in the API or the Streamlit app
- Every model subclasses `RiskModel` and implements `build` and `suggest_params`
- Run `ruff check src api app tests scripts` and `pytest -q` before declaring done

## Commands

```
make setup     # install deps + editable install
make data      # generate the 50,000-row synthetic book
make mlflow    # tracking server on :5000, needs its own shell
make train     # tune, log, evaluate, promote
make api       # FastAPI on :8000
make ui        # Streamlit on :8501
make test lint
```

## Data: Lending Club (done)

The switch is complete. `data/raw/loans.csv` is the adapted Lending Club
accepted-loans file, not the synthetic book.

```
python scripts/adapters/lending_club.py --input <path>/accepted_2007_to_2018Q4.csv
```

672,373 loans, 2007-06 to 2015-12, bad rate 14.81%, after dropping loans whose
outcome is still unknown and vintages too recent to have matured. The adapter
streams the 1.6 GB file in chunks: reading it whole needs several GB of resident
memory and dies on a normal laptop.

Features are `months_since_delinq`, `public_records` and `debt_to_income` in
place of `mean_dpd`/`max_dpd`, with `delinq_recency = 100 / (1 + months_since_delinq)`
in place of `dpd_severity`. `age` is gone too: Lending Club does not publish it,
and age is a protected characteristic under ECOA / Regulation B, so the scorecard
must not use one.

A later features pass widened the adapter from 20 source columns to 26, adding
origination-time fields it had been leaving in the raw file: `credit_history_months`
(from `earliest_cr_line`), `revolving_balance`, `total_accounts`,
`installment_amount`, `income_verification` and `months_since_public_record`, plus
the derived `installment_to_income`, `revol_bal_to_income`, `open_to_total_accounts`
and `public_record_recency`. It bought roughly +0.002 OOT AUC and +0.002 KS on both
families, which was enough to move the logistic from failing the KS floor to
passing it.

`zip_code` stays out. It is in the raw file and it predicts, but a zip prefix is a
documented proxy for race in the US and the disparate-impact exposure is not worth
the gain.

`scorecard.base_odds` is 5.75, which is `(1 - bad_rate) / bad_rate` on the
training window. Policy cutoffs are 620 / 550 and produce monotonic observed bad
rates on the holdout: Approve 7.43%, Refer 20.63%, Decline 40.39%.

## Things that are deliberate, do not "fix" them

- **Nothing is resampled.** This was measured rather than assumed, on 150k real
  rows with the preprocessor, solver and C held fixed:

  | sampler | fit | OOT AUC | OOT KS | Brier | mean PD |
  |---|---|---|---|---|---|
  | none | 5.2s | **0.6903** | **0.2772** | 0.1197 | 0.1249 |
  | class_weight=balanced | 5.2s | 0.6910 | 0.2788 | 0.1962 | 0.4202 |
  | RandomUnderSampler 0.5 | 3.1s | 0.6896 | 0.2773 | 0.1198 | 0.1249 |
  | RandomOverSampler 0.5 | 4.7s | 0.6911 | 0.2790 | 0.1197 | 0.1243 |
  | SMOTE 0.5 | 8.9s | 0.6861 | 0.2713 | 0.1199 | 0.1238 |
  | SMOTETomek 0.5 | 258.0s | 0.6861 | 0.2711 | 0.1199 | 0.1237 |

  At a 14.8% bad rate the imbalance is mild. SMOTE and SMOTE-Tomek both ranked
  *worse* than the untouched book for fifty times the compute, so they are gone
  and `imbalanced-learn` is no longer a dependency. Do not reintroduce either
  without a table like this showing it pays.
- **`class_weight` is out of the logistic search space.** It buys 0.0007 AUC and
  costs the score scale: mean predicted PD 0.42 against an observed 0.148.
  `PriorCorrectedClassifier` cannot catch it, because reweighting does not change
  `np.mean(y)`, so the correction computes to zero while the predictions inflate
- `PriorCorrectedClassifier` stays in the pipeline even though it is inert at zero
  offset today. It is the guard that makes reintroducing any sampler safe
- LightGBM uses `scale_pos_weight` plus isotonic calibration. The calibration is
  load-bearing: the 300-900 scale is a log-odds transform of PD, and an
  uncalibrated booster ranks fine while producing meaningless probabilities
- `solver="liblinear"`, not `saga`. Same l1/l2 search space, same answer: on 470k
  scaled rows saga took 95.3s and AUC 0.6782 where liblinear took 0.6s for 0.6783
- `penalty=` is kept rather than migrated to sklearn 1.9's `l1_ratio`, because
  `requirements.txt` declares `scikit-learn>=1.5` where that spelling does not
  exist. The deprecation warning is expected
- Reason codes pick the explainer to match the model: `TreeExplainer` for the
  boosters inside the calibration wrapper, `coef * x` for the linear model. Do not
  replace this with `shap.Explainer(predict_proba, row)` — passing the row as its
  own background returns exactly zero for every feature and the reason list is
  silently always empty
- `FeatureBuilder` takes plain lists, not `AppConfig`, so the pickled artifact
  does not carry the whole config
- The out-of-time window is carved by date before any random split
- `serialization_format="cloudpickle"` on `log_model` is required, skops refuses
  the custom transformer
- The promotion gate lives in `configs/config.yaml` under `promotion:`, not in
  code. `DeployabilityGate` reports which threshold a candidate missed and by how
  much, so a near miss is visible in the leaderboard's `blocked_by` column
- `training.tune_sample_size` searches hyperparameters on a stratified slice; the
  winner is always refitted on the whole training window. Set it to null to search
  full size, which costs about five hours here, nearly all of it LightGBM
- `SIGNAL_SCALE` in `credit_risk.data.synthetic` is calibrated so the CI fixture
  separates risk about as well as the real book (AUC 0.711 against 0.697). Below
  the promotion floors, CI trains a model and then fails because nothing can be
  promoted, which looks like a pipeline bug and is really an unrealistic fixture
- `CatBoostEstimator` exists because `CatBoostClassifier` normalises `cat_features`
  in its own constructor, so `get_params()` does not round-trip and sklearn refuses
  to clone it. Without the wrapper neither `cross_val_score` nor
  `CalibratedClassifierCV` can touch it. `__init__` assigns and does nothing else;
  that is the whole contract
- CatBoost is implemented and tested but commented out of `training.models`. Its
  full-size refit, tripled by the calibration folds, was killed by the OS for
  memory on a 16 GB machine. `one_hot_max_size=60` is load-bearing if you re-enable
  it: a fixed-budget fit went from 82.4s to 8.4s at an identical AUC of 0.6884,
  because every categorical here has at most 51 levels
- `ReasonCodeExplainer` dispatches per tree backend: `booster_` for LightGBM,
  `get_booster` for XGBoost, a `Pool` and native `ShapValues` for CatBoost. Adding
  a model family without extending that dispatch silently returns no reason codes
  for it, which is the same failure the degenerate-background bug produced

## Known metric expectations

On Lending Club, expect OOT AUC around 0.68 to 0.72 and KS around 0.28 to 0.35
once `grade` and the post-origination columns are excluded. Anything above 0.85
means a leakage column got back in. Check that first.

Where the bake-off landed, 25 trials per family, tuned on a 25k stratified sample
and refitted on the full 466,964-row training window:

| model | cv_auc | test_auc | oot_auc | oot_ks | oot_recall | cutoff_pd | score_psi | deployable |
|---|---|---|---|---|---|---|---|---|
| lightgbm | 0.6836 | 0.6902 | **0.6987** | **0.2850** | 0.7807 | 0.1036 | 0.0798 | yes |
| xgboost | 0.6831 | 0.6893 | 0.6980 | 0.2849 | 0.7828 | 0.1026 | 0.0860 | yes |
| logistic | 0.6802 | 0.6870 | 0.6926 | 0.2808 | 0.7959 | 0.1031 | 0.0718 | yes |

`credit_risk_scorer@champion` is v2 (LightGBM), `@challenger` is v3 (XGBoost).
Test-to-OOT AUC gap 0.0085.

XGBoost lands 0.0007 behind LightGBM on a deliberately matched search space, which
is noise. That is the useful part: two boosters that grow trees differently agree
to the fourth decimal, so the ceiling on this book is the data and not the
algorithm. Do not read a reordering of those two across reruns as a finding.

## Running it on Windows

There is no `make`. Use the venv interpreter directly:

```
.venv/Scripts/python.exe -m credit_risk.pipelines.train_pipeline
.venv/Scripts/python.exe scripts/policy_backtest.py --fraction 0.2 --quantiles
.venv/Scripts/python.exe scripts/promote_from_runs.py --dry-run
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src api app tests scripts
.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

`mlflow.tracking_uri` is `sqlite:///mlflow.db`, so no tracking server is needed;
the UI command above reads the same file the pipeline writes.

If a training job dies partway, do not rerun it blindly. Candidates are registered
and their metrics logged as each family finishes, so `scripts/promote_from_runs.py`
rebuilds the leaderboard from the tracking store and assigns aliases without
spending the compute again. That is how the current champion was promoted after
the four-family run was killed for memory during CatBoost's final fit.
