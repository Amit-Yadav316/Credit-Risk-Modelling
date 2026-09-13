# Credit Risk Underwriting Engine

End-to-end probability-of-default model with a champion-challenger setup, MLflow
tracking and registry, a FastAPI scoring service and a Streamlit underwriting
console. Scores are expressed on a 300-900 scale using a points-to-double-the-odds
transform, so the output sits in the same units a credit policy team already uses.

## What this adds over a notebook

| Concern | How it is handled |
|---|---|
| Model choice | Logistic scorecard as champion, LightGBM as challenger, both tuned with Optuna |
| Leakage | Every transform lives inside the sklearn pipeline, so nothing is fitted outside a training fold |
| Time robustness | Last 3 months held out as an out-of-time window before any random split |
| Calibration | Isotonic wrapper on LightGBM, so the 300-900 scale sits on a PD that means what it says |
| Reproducibility | Every run logs params, metrics, decile table, feature importance, PSI and diagnostic plots |
| Deployment | Registry aliases `champion` and `challenger` are the contract, promotion is not a redeploy |
| Stability | PSI on features and on the score distribution, with a hard gate before promotion |
| Ops | Docker Compose for MLflow plus API plus UI, GitHub Actions runs lint, tests and a smoke training job |

## Architecture

```
data/raw/loans.csv
      |
      v
DataLoader ---- time-based OOT split ----> SplitBundle
      |
      v
FeatureBuilder (LTI, EMI burden, DPD severity, utilisation band)
      |
      v
TrainingService ---- Optuna x {LogisticScorecard, LightGBMChallenger}
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
make data           # 50,000-row synthetic loan book with a real signal structure
make mlflow         # tracking server on :5000, leave running in another shell
make train          # tunes both models, logs everything, promotes the winner
make api            # FastAPI on :8000
make ui             # Streamlit on :8501
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

**The anchor has to come from the book.** `base_score` 600 at `base_odds` 10 with
`pdo` 40 says: a 600 applicant is roughly 10 to 1 to stay current, and every 40
points doubles those odds. Ten-to-one is near the 11% bad rate of the sample book.
Anchor at a textbook 50 to 1 instead and almost every applicant scores below 600,
the approve cutoff declines the entire portfolio, and the scale carries no
information across the range you actually lend in. Re-anchor whenever the book's
bad rate moves materially.

## The deployability gate

A candidate is only eligible for promotion when all four hold:

- OOT AUC at or above 0.70
- OOT KS at or above 0.30
- Score PSI between train and OOT below 0.25
- Test-to-OOT AUC gap below 0.05

The last one matters most. A model that wins on the in-time test set but loses
five points on the out-of-time window has learned the vintage, not the borrower.

## Metric discipline

Recall is meaningless without the cutoff that produced it, so the pipeline picks
the PD threshold that captures 85% of defaults on the test set, then applies that
same threshold unchanged to the OOT window and reports precision and approval
rate alongside it. Expect recall to fall between the two windows. That drop is the
honest number, and being able to explain it is worth more in an interview than a
higher headline figure.

Sanity check on the metric pair you quote: for a typical retail book, KS and AUC
move together roughly as KS around 2 x (AUC - 0.5) when scores are close to
normal. An AUC of 0.83 usually lands KS near 0.50 to 0.55, not 0.66. If your run
genuinely produced both, keep the numbers and be ready to show the decile table.
If they came from different runs or different splits, re-derive both from the same
OOT window before the numbers go on a CV.

## Repository layout

```
configs/config.yaml              single source of truth for features, cutoffs, MLflow
src/credit_risk/config.py        typed config loader
src/credit_risk/data/            loading and out-of-time splitting
src/credit_risk/features/        FeatureBuilder, shared by training and serving
src/credit_risk/models/          RiskModel base, logistic, LightGBM, prior correction, registry
src/credit_risk/training/        TrainingService, ModelPromoter
src/credit_risk/evaluation/      KS, Gini, PSI, decile lift, recall thresholding
src/credit_risk/monitoring/      DriftMonitor
src/credit_risk/scoring/         ScoreScaler, ScoringService
src/credit_risk/pipelines/       train_pipeline entry point
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
