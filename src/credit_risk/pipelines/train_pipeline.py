"""Entry point: python -m credit_risk.pipelines.train_pipeline"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from credit_risk.config import AppConfig
from credit_risk.data import DataLoader
from credit_risk.training import ModelPromoter, TrainingService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
logger = logging.getLogger("train_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register credit risk models")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    loader = DataLoader(config)
    data = loader.split(loader.load_raw())

    results = TrainingService(config).run(data)

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
            for r in results
        ]
    )
    logger.info("Leaderboard\n%s", table.to_string(index=False))

    if not args.no_promote:
        assigned = ModelPromoter(config).promote(results)
        logger.info("Aliases assigned: %s", assigned or "none")


if __name__ == "__main__":
    main()
