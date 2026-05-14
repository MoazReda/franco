"""Evaluate a fine-tuned model checkpoint on the test split.

Reports BLEU, chrF, and exact-match for both directions, prints a few
sample predictions per direction, and writes the full predictions to
``data/reports/model_predictions.csv`` for error analysis.

Usage:
    python scripts/evaluate_model.py --model models/franco-translator-v1
    python scripts/evaluate_model.py --model MoazReda/franco-translator-v1
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
from src.training import predict  # noqa: E402

DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "reports" / "model_predictions.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local path or HuggingFace Hub id")
    p.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    p.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    p.add_argument("--n-examples", type=int, default=5)
    p.add_argument("--device", default="auto", help="'cpu', 'cuda', or 'auto'")
    return p.parse_args()


def resolve_device(spec: str) -> str:
    import torch

    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = resolve_device(args.device)
    logging.info("Loading model from %s on %s", args.model, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    test_path = Path(args.splits_dir) / "test.csv"
    if not test_path.exists():
        raise SystemExit(
            f"Test split not found at {test_path}. Run scripts/prepare_splits.py first."
        )
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    logging.info("Loaded test split: %d rows", len(test))

    def predict_fa_to_ar(src: str) -> str:
        return predict(src, "fa2ar", tokenizer, model)

    def predict_ar_to_fa(src: str) -> str:
        return predict(src, "ar2fa", tokenizer, model)

    logging.info("─" * 60)
    logging.info("DIRECTION 1: Franco → Arabic")
    fa_to_ar = evaluate_predictions(
        sources=test["franco"].tolist(),
        references=test["arabic"].tolist(),
        predict_fn=predict_fa_to_ar,
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
        predict_fn=predict_ar_to_fa,
        n_examples=args.n_examples,
    )
    logging.info("  %s", ar_to_fa.summary())
    for src, ref, pred in ar_to_fa.sample_predictions:
        logging.info("    src : %s", src)
        logging.info("    ref : %s", ref)
        logging.info("    pred: %s", pred)
        logging.info("")

    rows = []
    for _, row in test.iterrows():
        rows.append(
            {
                "id": row["id"],
                "franco": row["franco"],
                "arabic": row["arabic"],
                "pred_arabic": predict_fa_to_ar(row["franco"]),
                "pred_franco": predict_ar_to_fa(row["arabic"]),
            }
        )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig")

    logging.info("─" * 60)
    logging.info("Baseline (rule-based) — Franco→AR BLEU=3.94 chrF=33.90")
    logging.info("Baseline (rule-based) — AR→Franco BLEU=15.38 chrF=42.52")
    logging.info("Model deltas to beat the floor:")
    logging.info(
        "  Franco→AR : ΔBLEU=%+.2f  ΔchrF=%+.2f",
        fa_to_ar.bleu - 3.94,
        fa_to_ar.chrf - 33.90,
    )
    logging.info(
        "  AR→Franco : ΔBLEU=%+.2f  ΔchrF=%+.2f",
        ar_to_fa.bleu - 15.38,
        ar_to_fa.chrf - 42.52,
    )
    logging.info("Saved predictions to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
