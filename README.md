# Credit Risk Underwriting Engine

End-to-end probability-of-default model with a champion-challenger setup, MLflow
tracking and registry, a FastAPI scoring service and a Streamlit underwriting
console. Scores are expressed on a 300-900 scale using a points-to-double-the-odds
transform, so the output sits in the same units a credit policy team already uses.

Trained and evaluated on 672,373 real Lending Club loans issued 2007-06 to 2015-12,
bad rate 14.8%, after dropping loans whose outcome is still unknown. A synthetic
book generator ships alongside it for CI and for running the pipeline without the
download.

## What this adds over a notebook

| Concern | How it is handled |
|---|---|
| Model choice | Logistic scorecard, LightGBM and XGBoost, each tuned with Optuna, promoted to `champion` and `challenger` through a deployability gate |
| Leakage | Every transform lives inside the sklearn pipeline, so nothing is fitted outside a training fold |
| Time robustness | Last 3 months held out as an out-of-time window before any random split |
| Calibration | Isotonic wrapper on the boosters, so the 300-900 scale sits on a PD that means what it says |
| Reproducibility | Every run logs params, metrics, decile table, feature importance, PSI and diagnostic plots |
| Deployment | Registry aliases `champion` and `challenger` are the contract, promotion is not a redeploy |
| Stability | PSI on features and on the score distribution, with a hard gate before promotion |
| Ops | Docker Compose for MLflow plus API plus UI, GitHub Actions runs lint, tests and a smoke training job |

## Results on Lending Club

The out-of-time window is the last three months, October to December 2015, 88,667
loans, carved off by date before any random split. Each family ran 25 Optuna
trials on a 25k stratified sample, and the winner was refitted on the full
466,964-row training window.

| model | cv AUC | test AUC | OOT AUC | OOT KS | OOT recall | approved at cutoff | score PSI | alias |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.6836 | 0.6902 | **0.6987** | **0.2850** | 0.7807 | 45.5% | 0.0798 | champion |
| XGBoost | 0.6831 | 0.6893 | 0.6980 | 0.2849 | 0.7828 | 45.3% | 0.0860 | challenger |
| Logistic | 0.6802 | 0.6870 | 0.6926 | 0.2808 | 0.7959 | 42.9% | 0.0718 | |

Recall and approval are at each model's PD cutoff, about 0.10, chosen to catch 85%
of defaults on the in-time test set. The two boosters grow trees differently and
agree to the fourth decimal, so the ceiling on this book is the data, not the
algorithm.

The champion's policy bands on the same out-of-time window, cutoffs 620 and 550:

| decision | share of loans | share of defaults | default rate |
|---|---|---|---|
| Approve | 50.3% | 26.1% | 7.7% |
| Refer to underwriter | 46.4% | 65.1% | 20.8% |
| Decline | 3.3% | 8.8% | 39.3% |
| Whole window | 100% | 100% | 14.8% |

Lending Club only publishes loans it accepted, so this re-ranks a book that was
already underwritten once. There are no rejected applicants to score.

## Architecture

```
data/raw/loans.csv
      |
      v
DataLoader ---- time-based OOT split ----> SplitBundle
      |
      v
FeatureBuilder (LTI, EMI and instalment burden, delinquency recency, utilisation band)
      |
      v
TrainingService ---- Optuna x {LogisticScorecard, LightGBMChallenger, XGBoostChallenger}
      |                     |
      |                     +--> MLflow runs: params, metrics, artifacts
      v
ModelPromoter ---- deployability gate ----> registry aliases
      |
      v
ScoringService (models:/credit_risk_scorer@champion)
      |
      +--> FastAPI /score, /score/batch
      +--> Streamlit console with gauge, bands and batch upload
```

## Run it

```bash
make setup          # install deps and the package in editable mode

# data, pick one
python scripts/adapters/lending_club.py --input <path>/accepted_2007_to_2018Q4.csv
make data           # 50,000-row synthetic book, no download needed

make train          # tunes each family in training.models, logs to MLflow, promotes
make api            # FastAPI on :8000
make ui             # Streamlit on :8501
make mlflow         # optional: MLflow UI on :5000 over the same sqlite store
```

Or the whole stack at once:

```bash
docker compose up --build
```

## The two calibration traps

**Resampling moves the base rate, so this book is not resampled.** Any resampler
trains the classifier on a book whose bad rate is nothing like the real one.
Ranking survives, so AUC and KS look healthy and nobody notices, but every PD comes
out inflated, and the 300-900 scale is a transform of PD, so the whole scorecard
shifts down and the approve cutoff stops meaning what the policy says it means.

Measured on this book at a 14.8% bad rate, resampling did not even pay for itself
on ranking: SMOTE and SMOTE-Tomek both scored OOT AUC 0.6861 against 0.6903 for
the untouched training set, and SMOTE-Tomek cost fifty times the compute. So the
model trains on real rows only and imbalance is handled at the decision cutoff.
`PriorCorrectedClassifier` stays in the pipeline as a guard: it applies the
standard choice-based-sampling offset in log-odds space, and is inert at zero
offset while nothing resamples.

**The anchor has to come from the book.** `base_score` 600 at `base_odds` 5.75 with
`pdo` 40 says: a 600 applicant is 5.75 to 1 to stay current, and every 40 points
doubles those odds. 5.75 is `(1 - bad_rate) / bad_rate` at the 14.8% bad rate of
the training window, so 600 sits on the average borrower. Anchor at a textbook 50 to 1 instead and almost every applicant scores below 600,
the approve cutoff declines the entire portfolio, and the scale carries no
information across the range you actually lend in. Re-anchor whenever the book's
bad rate moves materially.

## The deployability gate

A candidate is only eligible for promotion when all four hold:

- OOT AUC at or above 0.68
- OOT KS at or above 0.28
- Score PSI between train and OOT below 0.25
- Test-to-OOT AUC gap below 0.05

The thresholds live in `configs/config.yaml` under `promotion:`, not in code. The
floors sit at the bottom of the band this book supports once `grade` and the
post-origination columns are excluded, OOT AUC 0.68 to 0.72, so a sound model
promotes and a broken one does not. `DeployabilityGate` reports which threshold a
candidate missed and by how much.

The last one matters most. A model that wins on the in-time test set but loses
five points on the out-of-time window has learned the vintage, not the borrower.

## Metric discipline

Recall is meaningless without the cutoff that produced it, so the pipeline picks
the PD threshold that captures 85% of defaults on the test set, then applies that
same threshold unchanged to the OOT window and reports precision and approval
rate alongside it. Expect recall to fall between the two windows: the champion
catches 85.0% of defaults on the test set and 78.1% out of time at the same cutoff.
That drop is the honest number.

Sanity check on the metric pair: when scores are close to normal with a similar
spread in both classes, AUC pins down KS as `KS = 2 * Phi(Phi^-1(AUC) / sqrt(2)) - 1`.
The champion's OOT AUC of 0.6987 implies a KS of 0.287 against 0.285 observed, so
the two numbers agree. The common shortcut `KS = 2 * (AUC - 0.5)` overstates it
badly, 0.397 here. A KS far from what the AUC implies usually means the two came
from different runs or different splits.

## Repository layout

```
configs/config.yaml              single source of truth for features, cutoffs, promotion gate, MLflow
src/credit_risk/config.py        typed config loader
src/credit_risk/data/            loading, out-of-time splitting, synthetic book generator
src/credit_risk/features/        FeatureBuilder, shared by training and serving
src/credit_risk/models/          RiskModel base, logistic, LightGBM, XGBoost, CatBoost, prior correction, registry
src/credit_risk/training/        TrainingService, DeployabilityGate, ModelPromoter
src/credit_risk/evaluation/      KS, Gini, PSI, decile lift, recall thresholding, policy back-test
src/credit_risk/monitoring/      DriftMonitor
src/credit_risk/scoring/         ScoreScaler, ScoringService, reason codes (SHAP for boosters)
src/credit_risk/pipelines/       train_pipeline entry point
scripts/adapters/lending_club.py raw Lending Club file to the pipeline schema
scripts/policy_backtest.py       approve, refer and decline bad rates on a holdout
scripts/promote_from_runs.py     rebuild the leaderboard from MLflow without retraining
api/main.py                      FastAPI service
app/streamlit_app.py             underwriting console
```

## Swapping in your own data

Point `data.raw_path` at your extract and update `features.numeric` and
`features.categorical` in the config. `FeatureBuilder` expects `loan_amount`,
`annual_income`, `interest_rate`, `tenure_months`, `months_since_delinq`,
`months_since_public_record`, `installment_amount`, `revolving_balance`,
`total_accounts`, `num_open_accounts` and `credit_utilisation` to exist, since the
derived features are built from them. `tests/test_synthetic.py` asserts that the
config, the generator and the feature builder still agree, so a mismatch fails
there rather than inside a training run.
Nothing else in the pipeline is hard-coded to a column name.
