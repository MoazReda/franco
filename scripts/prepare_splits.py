"""Generate deterministic train/val/test splits from the annotation sheet.

Output goes to data/splits/{train,val,test}.csv. Each row has columns
id, franco, arabic. Splits are gitignored (since they derive from the
gitignored annotation_sheet.xlsx).

Usage:
    python scripts/prepare_splits.py
    python scripts/prepare_splits.py --seed 7 --train 0.7 --val 0.15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python scripts/X.py` to import from src/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_utils import load_annotated, split_dataset, write_splits  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "data" / "annotated" / "annotation_sheet.xlsx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "splits"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    test_frac = round(1.0 - args.train - args.val, 6)
    if test_frac <= 0:
        raise SystemExit(f"Test fraction must be > 0, got {test_frac}")

    df = load_annotated(Path(args.input))
    logging.info("Loaded %d annotated rows from %s", len(df), args.input)

    splits = split_dataset(df, (args.train, args.val, test_frac), seed=args.seed)
    paths = write_splits(splits, Path(args.output_dir))

    logging.info(
        "Wrote splits: train=%d  val=%d  test=%d  (seed=%d)",
        len(splits[0]),
        len(splits[1]),
        len(splits[2]),
        args.seed,
    )
    logging.info("  %s", paths.train)
    logging.info("  %s", paths.val)
    logging.info("  %s", paths.test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
