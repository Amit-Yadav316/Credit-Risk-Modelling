"""Score the tail of the book with the registry champion and report each band.

    python scripts/policy_backtest.py --fraction 0.2

Optional --approve-above / --refer-above try a different policy without editing
the config, which is how the cutoffs get tuned before they are committed.
"""
from __future__ import annotations

import argparse
import logging

from credit_risk.config import AppConfig
from credit_risk.evaluation import PolicyBacktest

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("policy_backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Back-test the decision policy")
    parser.add_argument("--config", default=None)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--approve-above", type=int, default=None)
    parser.add_argument("--refer-above", type=int, default=None)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--quantiles", action="store_true", help="also print the score distribution")
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    from credit_risk.scoring.service import ScoringService

    backtest = PolicyBacktest(config, ScoringService(config, alias=args.alias))
    scores, y = backtest.scores(backtest.holdout(args.fraction))

    if args.quantiles:
        logger.info("Score distribution\n%s", backtest.score_quantiles(scores).to_string(index=False))

    report = backtest.band_table(scores, y, args.approve_above, args.refer_above)
    logger.info(
        "Policy back-test, approve_above=%s refer_above=%s\n%s",
        report.approve_above,
        report.refer_above,
        report.table.to_string(index=False),
    )
    logger.info("Summary: %s", report.summary())
    if not report.is_monotonic:
        logger.warning("Bad rates are NOT monotonic across the bands. Move the cutoffs.")


if __name__ == "__main__":
    main()
