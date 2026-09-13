"""Rebuild the leaderboard from logged MLflow runs and assign aliases.

    python scripts/promote_from_runs.py

Training already registers a model version per candidate and logs its metrics
before moving to the next family. Promotion, though, happens once at the end of
the job, so a run that dies partway leaves fully trained and fully evaluated
models in the registry with no alias pointing at any of them.

That is recoverable without retraining: every number the gate needs is already in
the tracking store. This reads the candidate runs back, applies the same
DeployabilityGate the pipeline would have, and hands the survivors to the same
ModelPromoter. Use it after an interrupted job, or to re-promote under changed
thresholds without spending the compute again.
"""
from __future__ import annotations

import argparse
import logging

import mlflow
import pandas as pd
from mlflow import MlflowClient

from credit_risk.config import AppConfig
from credit_risk.training import DeployabilityGate, ModelPromoter
from credit_risk.training.trainer import CandidateResult

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("promote_from_runs")

# Metric keys as TrainingService logs them
TEST_PREFIX, OOT_PREFIX = "test_", "oot_"


def _candidates(config: AppConfig, client: MlflowClient) -> list[CandidateResult]:
    experiment = client.get_experiment_by_name(config.mlflow.experiment)
    if experiment is None:
        raise SystemExit(f"No experiment named {config.mlflow.experiment!r}")

    gate = DeployabilityGate(config)
    results: list[CandidateResult] = []
    for run in client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"]):
        family = run.data.tags.get("model_family")
        if not family or run.info.status != "FINISHED":
            continue
        metrics = run.data.metrics
        if f"{OOT_PREFIX}roc_auc" not in metrics:
            continue  # died before it finished evaluating

        test = {k[len(TEST_PREFIX):]: v for k, v in metrics.items() if k.startswith(TEST_PREFIX)}
        oot = {k[len(OOT_PREFIX):]: v for k, v in metrics.items() if k.startswith(OOT_PREFIX)}
        score_psi = metrics.get("score_psi_train_vs_oot", float("nan"))
        results.append(
            CandidateResult(
                name=family,
                run_id=run.info.run_id,
                params={k: v for k, v in run.data.params.items() if k.startswith("hp.")},
                cv_auc=metrics.get("cv_auc", float("nan")),
                test=test,
                oot=oot,
                score_psi=score_psi,
                gate_failures=gate.failures(test=test, oot=oot, score_psi=score_psi),
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote from already-logged runs")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true", help="show the leaderboard, assign nothing")
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    client = MlflowClient()

    results = _candidates(config, client)
    if not results:
        raise SystemExit("No finished candidate runs found to promote from.")

    table = pd.DataFrame(
        [
            {
                "model": r.name,
                "cv_auc": round(r.cv_auc, 4),
                "test_auc": round(r.test["roc_auc"], 4),
                "oot_auc": round(r.oot["roc_auc"], 4),
                "oot_ks": round(r.oot["ks"], 4),
                "oot_recall": round(r.oot["recall_at_threshold"], 4),
                "oot_precision": round(r.oot["precision_at_threshold"], 4),
                "cutoff_pd": round(r.oot["threshold"], 4),
                "score_psi": round(r.score_psi, 4),
                "deployable": r.is_deployable,
                "blocked_by": r.blocked_by,
            }
            for r in sorted(results, key=lambda r: r.oot["roc_auc"], reverse=True)
        ]
    )
    logger.info("Leaderboard, rebuilt from %s logged runs\n%s", len(results), table.to_string(index=False))

    if args.dry_run:
        logger.info("Dry run, aliases unchanged.")
        return
    assigned = ModelPromoter(config).promote(results)
    logger.info("Aliases assigned: %s", assigned or "none")


if __name__ == "__main__":
    main()
