"""Scoring package.

`ScoringService` is exported lazily so that importing the pure score algebra does
not drag in MLflow. Training and unit tests only need `ScoreScaler`.
"""
from __future__ import annotations

from typing import Any

from .scorecard import ScoreBand, ScoreScaler

__all__ = ["ScoreScaler", "ScoreBand", "ScoringService"]


def __getattr__(name: str) -> Any:
    if name == "ScoringService":
        from .service import ScoringService

        return ScoringService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
