"""Optuna tuning, MLflow tracking, champion-challenger evaluation."""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from credit_risk.config import AppConfig
from credit_risk.data import SplitBundle
from credit_risk.evaluation.metrics import RiskMetrics
from credit_risk.models import build_model
from credit_risk.monitoring import DriftMonitor
from credit_risk.scoring.scorecard import ScoreScaler
from credit_risk.training.gate import DeployabilityGate

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    name: str
    run_id: str
    params: dict[str, Any]
    cv_auc: float
    test: dict[str, float]
    oot: dict[str, float]
    score_psi: float
    gate_failures: tuple[str, ...] = ()

    @property
    def is_deployable(self) -> bool:
        """True when every threshold in the config's promotion block held.

        The thresholds themselves live in configs/config.yaml and are applied by
        DeployabilityGate, so changing risk appetite is a config change that shows
        up in the logged run, not an edit buried in a dataclass.
        """
        return not self.gate_failures

    @property
    def blocked_by(self) -> str:
        return "; ".join(self.gate_failures) if self.gate_failures else "-"


class TrainingService:
    """Runs the full experiment for every model named in the config.

    One parent MLflow run per training job, one child run per candidate model,
    one nested run per Optuna trial. That structure is what makes the experiment
    readable three months later when someone asks why version 7 was promoted.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.scaler = ScoreScaler(config.scorecard)
        self.monitor = DriftMonitor(config)
        self.gate = DeployabilityGate(config)
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)

    def run(self, data: SplitBundle) -> list[CandidateResult]:
        results: list[CandidateResult] = []
        with mlflow.start_run(run_name="training-job") as parent:
            mlflow.log_params({f"data.{k}": v for k, v in data.summary().items()})
            mlflow.log_dict(self.config.raw, "config.yaml")
            for model_name in self.config.training.models:
                results.append(self._train_candidate(model_name, data))
            best = max(results, key=lambda r: r.oot["roc_auc"])
            mlflow.set_tag("best_model", best.name)
            mlflow.log_metric("best_oot_auc", best.oot["roc_auc"])
            logger.info("Parent run %s complete. Best: %s", parent.info.run_id, best.name)
        return results

    def _train_candidate(self, model_name: str, data: SplitBundle) -> CandidateResult:
        with mlflow.start_run(run_name=model_name, nested=True) as run:
            model = build_model(model_name, self.config).set_prior(data.y_train.mean())
            best_params, cv_auc = self._tune(model, data)
            model.fit(data.X_train, data.y_train, best_params)

            train_pd = model.predict_proba(data.X_train)
            test_pd = model.predict_proba(data.X_test)
            oot_pd = model.predict_proba(data.X_oot)

            # Operating point is set on test, then applied unchanged to OOT
            cutoff = RiskMetrics.threshold_for_recall(data.y_test, test_pd, target_recall=0.85)
            test_metrics = RiskMetrics.full_report(data.y_test, test_pd, cutoff)
            oot_metrics = RiskMetrics.full_report(data.y_oot, oot_pd, cutoff)

            drift = self.monitor.compare(
                data.X_train,
                data.X_oot,
                self.scaler.pd_to_score(train_pd),
                self.scaler.pd_to_score(oot_pd),
            )

            mlflow.log_params({f"hp.{k}": v for k, v in best_params.items()})
            mlflow.log_param("tune.sample_size", self.config.training.tune_sample_size)
            mlflow.log_metric("cv_auc", cv_auc)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_metrics({f"oot_{k}": v for k, v in oot_metrics.items()})
            mlflow.log_metric("score_psi_train_vs_oot", drift.score_psi)
            gate_failures = self.gate.failures(
                test=test_metrics, oot=oot_metrics, score_psi=drift.score_psi
            )
            mlflow.set_tags(
                {
                    "model_family": model_name,
                    "stage": "candidate",
                    "deployable": not gate_failures,
                    "gate_failures": "; ".join(gate_failures) or "none",
                }
            )

            self._log_artifacts(model, data, test_pd, oot_pd, drift)

            # Cast numerics to float64 before inferring the signature. An int64
            # schema rejects any request where the field arrives as a float,
            # which is what every JSON client will send.
            example = data.X_test.head(5).copy()
            num_cols = example.select_dtypes(include="number").columns
            example[num_cols] = example[num_cols].astype("float64")
            signature = infer_signature(example, test_pd[:5])
            mlflow.sklearn.log_model(
                sk_model=model.pipeline,
                name="model",
                signature=signature,
                input_example=example.head(3),
                serialization_format="cloudpickle",
                registered_model_name=self.config.mlflow.registered_model,
            )

            return CandidateResult(
                name=model_name,
                run_id=run.info.run_id,
                params=best_params,
                cv_auc=cv_auc,
                test=test_metrics,
                oot=oot_metrics,
                score_psi=drift.score_psi,
                gate_failures=gate_failures,
            )

    def _tuning_sample(self, data: SplitBundle) -> tuple[pd.DataFrame, pd.Series]:
        """The stratified slice the hyperparameter search runs on.

        A 40-trial search at full fold size costs about five hours on this book,
        almost all of it LightGBM refitting 467k rows two hundred times. The search
        only has to *rank* hyperparameters against each other, which a stratified
        slice does faithfully at a fraction of the cost. The winning configuration is then
        refitted on the entire training window by `_train_candidate`, so the model
        that gets registered has still seen every row, and every reported metric
        is measured full size.

        Set training.tune_sample_size to null to search on the whole window.
        """
        size = self.config.training.tune_sample_size
        if not size or size >= len(data.X_train):
            return data.X_train, data.y_train
        X, _, y, _ = train_test_split(
            data.X_train,
            data.y_train,
            train_size=int(size),
            random_state=self.config.data.random_state,
            stratify=data.y_train,
        )
        logger.info(
            "Tuning on a stratified %s-row sample of the %s-row training window (bad rate %.4f)",
            f"{len(X):,}",
            f"{len(data.X_train):,}",
            y.mean(),
        )
        return X.reset_index(drop=True), y.reset_index(drop=True)

    def _tune(self, model, data: SplitBundle) -> tuple[dict[str, Any], float]:
        X_tune, y_tune = self._tuning_sample(data)
        cv = StratifiedKFold(
            n_splits=self.config.training.cv_folds,
            shuffle=True,
            random_state=self.config.data.random_state,
        )

        def objective(trial: optuna.Trial) -> float:
            params = model.suggest_params(trial)
            pipeline = model.build(params)
            scores = cross_val_score(pipeline, X_tune, y_tune, cv=cv, scoring="roc_auc", n_jobs=1)
            trial.set_user_attr("cv_std", float(scores.std()))
            return float(scores.mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.data.random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(objective, n_trials=self.config.training.n_trials, show_progress_bar=False)
        logger.info("%s best CV AUC %.4f", model.name, study.best_value)
        return study.best_params, float(study.best_value)

    def _log_artifacts(self, model, data: SplitBundle, test_pd, oot_pd, drift) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            deciles = RiskMetrics.decile_table(data.y_oot, oot_pd)
            deciles.to_csv(tmp / "oot_decile_table.csv", index=False)
            model.feature_importance().to_csv(tmp / "feature_importance.csv", index=False)
            drift.feature_psi.to_csv(tmp / "feature_psi.csv", index=False)

            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            for label, y, p in [("test", data.y_test, test_pd), ("oot", data.y_oot, oot_pd)]:
                fpr, tpr, _ = roc_curve(y, p)
                axes[0].plot(fpr, tpr, label=f"{label} AUC {roc_auc_score(y, p):.3f}")
            axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
            axes[0].set_title("ROC")
            axes[0].legend()

            axes[1].bar(deciles["decile"].astype(str), deciles["bad_rate"])
            axes[1].set_title("OOT bad rate by decile")
            axes[1].set_xlabel("risk decile (1 = riskiest)")

            axes[2].hist(self.scaler.pd_to_score(test_pd), bins=40, alpha=0.6, label="test")
            axes[2].hist(self.scaler.pd_to_score(oot_pd), bins=40, alpha=0.6, label="oot")
            axes[2].set_title(f"Score distribution, PSI {drift.score_psi:.3f}")
            axes[2].legend()

            fig.tight_layout()
            fig.savefig(tmp / "diagnostics.png", dpi=130)
            plt.close(fig)

            mlflow.log_artifacts(str(tmp), artifact_path="diagnostics")
