"""Write the synthetic loan book used as the CI fixture and the no-download path.

    python scripts/make_synthetic_data.py

The generator itself is `credit_risk.data.SyntheticBookGenerator`; this is only a
command line around it. Swap data.raw_path in configs/config.yaml, or run
scripts/adapters/lending_club.py, when you have the real extract.
"""
from __future__ import annotations

import argparse
import logging

from credit_risk.data import SyntheticBookGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("make_synthetic_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic loan book")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bad-rate", type=float, default=0.148)
    parser.add_argument("--output", default="data/raw/loans.csv")
    args = parser.parse_args()

    from pathlib import Path

    df = SyntheticBookGenerator(
        n=args.rows, seed=args.seed, target_bad_rate=args.bad_rate
    ).generate()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(
        "Wrote %s rows to %s | bad rate %.2f%% | %s to %s",
        f"{len(df):,}",
        out,
        100 * df.default_flag.mean(),
        df.disbursal_date.min().date(),
        df.disbursal_date.max().date(),
    )


if __name__ == "__main__":
    main()
