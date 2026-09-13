"""Alias-based promotion in the MLflow Model Registry."""
from __future__ import annotations

import logging

import mlflow
from mlflow import MlflowClient

from credit_risk.config import AppConfig

logger = logging.getLogger(__name__)


class ModelPromoter:
    """Assigns `champion` and `challenger` aliases to registered versions.

    Promotion rule: the deployable candidate with the highest out-of-time AUC
    becomes champion, the runner-up becomes challenger. A candidate that fails
    the deployability gate is never promoted no matter how good its AUC looks,
    because an unstable model that beats the incumbent on one window is a
    liability on the next one.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        self.client = MlflowClient()
        self.model_name = config.mlflow.registered_model

    def _version_for_run(self, run_id: str) -> str | None:
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        for v in versions:
            if v.run_id == run_id:
                return v.version
        return None

    def promote(self, results) -> dict[str, str]:
        deployable = sorted(
            [r for r in results if r.is_deployable], key=lambda r: r.oot["roc_auc"], reverse=True
        )
        if not deployable:
            logger.warning("No candidate passed the deployability gate. Aliases unchanged.")
            return {}

        assigned: dict[str, str] = {}
        for alias, result in zip(["champion", "challenger"], deployable, strict=False):
            version = self._version_for_run(result.run_id)
            if version is None:
                continue
            self.client.set_registered_model_alias(self.model_name, alias, version)
            self.client.set_model_version_tag(self.model_name, version, "model_family", result.name)
            self.client.set_model_version_tag(
                self.model_name, version, "oot_auc", f"{result.oot['roc_auc']:.4f}"
            )
            self.client.update_model_version(
                name=self.model_name,
                version=version,
                description=(
                    f"{result.name} | OOT AUC {result.oot['roc_auc']:.4f} "
                    f"| OOT KS {result.oot['ks']:.4f} | score PSI {result.score_psi:.4f}"
                ),
            )
            assigned[alias] = version
            logger.info("Alias %s -> %s v%s", alias, result.name, version)
        return assigned
