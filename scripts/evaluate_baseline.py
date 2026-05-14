"""Evaluate the rule-based Franco↔Arabic baseline on the test split.

Reports BLEU, chrF, and exact-match for both directions and prints a few
sample predictions. Writes the full predictions to
data/reports/baseline_predictions.csv for error analysis.

Usage:
    python scripts/evaluate_baseline.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import evaluate_predictions  # noqa: E402
from src.translator import arabic_to_franco, franco_to_arabic  # noqa: E402

DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "reports" / "baseline_predictions.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    p.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    p.add_argument("--n-examples", type=int, default=5)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    test_path = Path(args.splits_dir) / "test.csv"
    if not test_path.exists():
        raise SystemExit(
            f"Test split not found at {test_path}. Run scripts/prepare_splits.py first."
        )

    test = pd.read_csv(test_path, encoding="utf-8-sig")
    logging.info("Loaded test split: %d rows", len(test))
    logging.info("─" * 60)

    logging.info("DIRECTION 1: Franco → Arabic")
    fa_to_ar = evaluate_predictions(
        sources=test["franco"].tolist(),
        references=test["arabic"].tolist(),
        predict_fn=franco_to_arabic,
        n_examples=args.n_examples,
    )
    logging.info("  %s", fa_to_ar.summary())
    for src, ref, pred in fa_to_ar.sample_predictions:
        logging.info("    src : %s", src)
        logging.info("    ref : %s", ref)
        logging.info("    pred: %s", pred)
        logging.info("")

    logging.info("─" * 60)
    logging.info("DIRECTION 2: Arabic → Franco")
    ar_to_fa = evaluate_predictions(
        sources=test["arabic"].tolist(),
        references=test["franco"].tolist(),
        predict_fn=arabic_to_franco,
        n_examples=args.n_examples,
    )
    logging.info("  %s", ar_to_fa.summary())
    for src, ref, pred in ar_to_fa.sample_predictions:
        logging.info("    src : %s", src)
        logging.info("    ref : %s", ref)
        logging.info("    pred: %s", pred)
        logging.info("")

    # Save full predictions for error analysis
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in test.iterrows():
        rows.append(
            {
                "id": row["id"],
                "franco": row["franco"],
                "arabic": row["arabic"],
                "pred_arabic": franco_to_arabic(row["franco"]),
                "pred_franco": arabic_to_franco(row["arabic"]),
            }
        )
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig")
    logging.info("─" * 60)
    logging.info("Saved predictions to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
