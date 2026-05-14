"""Generate synthetic Franco spelling variants for the training split only.

For every row in ``data/splits/train.csv``, generates up to N spelling
variants and emits ``(variant_franco, arabic)`` pairs. The original row is
also kept. Output is written to ``data/splits/train_augmented.csv``.

Val and test splits are NEVER augmented — that would invalidate evaluation.

Usage:
    python scripts/augment_train.py
    python scripts/augment_train.py --n-variants 6 --seed 7
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.augmentation import generate_variants  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "data" / "splits" / "train.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "splits" / "train_augmented.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--n-variants", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(
            f"Train split not found: {in_path}. Run scripts/prepare_splits.py first."
        )

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    logging.info("Loaded %d training rows from %s", len(df), in_path)

    rows: list[dict] = []
    n_variants_total = 0
    n_with_variants = 0

    for _, row in df.iterrows():
        rows.append(
            {
                "id": row["id"],
                "franco": row["franco"],
                "arabic": row["arabic"],
                "source": "original",
            }
        )
        variants = generate_variants(
            row["franco"], n=args.n_variants, seed=args.seed
        )
        if variants:
            n_with_variants += 1
        for v in variants:
            rows.append(
                {
                    "id": row["id"],
                    "franco": v,
                    "arabic": row["arabic"],
                    "source": "synthetic",
                }
            )
            n_variants_total += 1

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    multiplier = len(out_df) / len(df) if len(df) else 0
    avg_variants = n_variants_total / max(n_with_variants, 1)

    logging.info("─" * 60)
    logging.info("Wrote %d rows to %s", len(out_df), out_path)
    logging.info("  Original rows           : %d", len(df))
    logging.info("  Rows that got variants  : %d", n_with_variants)
    logging.info("  Synthetic variants added: %d", n_variants_total)
    logging.info("  Avg variants per source : %.2f", avg_variants)
    logging.info("  Dataset multiplier      : ×%.2f", multiplier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
