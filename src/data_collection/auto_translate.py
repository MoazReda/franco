"""Auto-translate Arabic annotations to English using a HuggingFace model.

Produces a 3-column parallel CSV (franco, arabic, english) plus an
``english_source`` column flagging whether each English value came from
a human annotator or this auto-translation pass.

Designed to be safe to re-run: existing rows in the output CSV are loaded
and skipped, so a crash mid-job loses at most one checkpoint window.

Example
-------
    python -m src.data_collection.auto_translate \
        --input data/annotated/annotation_sheet.xlsx \
        --output data/annotated/annotated_500.csv \
        --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

LOGGER = logging.getLogger("franco.auto_translate")

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-ar-en"
DEFAULT_INPUT = Path("data/annotated/annotation_sheet.xlsx")
DEFAULT_OUTPUT = Path("data/annotated/annotated_500.csv")
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LEN = 256
CHECKPOINT_EVERY_N_BATCHES = 5


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    model_name: str
    batch_size: int
    max_length: int
    col_id: str = "id"
    col_franco: str = "franco"
    col_arabic: str = "arabic"
    col_english: str = "english"
    col_source: str = "english_source"


# ── I/O helpers ────────────────────────────────────────────────────────


def load_annotations(path: Path, cfg: Config) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)

    missing = {cfg.col_id, cfg.col_franco, cfg.col_arabic} - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    if cfg.col_english not in df.columns:
        df[cfg.col_english] = pd.NA
    if cfg.col_source not in df.columns:
        # Existing English values are assumed manual unless we overwrite them.
        df[cfg.col_source] = df[cfg.col_english].apply(
            lambda v: "manual" if isinstance(v, str) and v.strip() else pd.NA
        )
    return df


def load_existing_output(path: Path, cfg: Config) -> dict[int, tuple[str, str]]:
    """Return {id: (english, source)} from a prior run, for resume."""
    if not path.exists():
        return {}
    prior = pd.read_csv(path, encoding="utf-8-sig")
    needed = {cfg.col_id, cfg.col_english}
    if not needed.issubset(prior.columns):
        return {}
    if cfg.col_source not in prior.columns:
        prior[cfg.col_source] = "manual"
    done = prior[prior[cfg.col_english].notna() & (prior[cfg.col_english].astype(str).str.strip() != "")]
    return {
        row[cfg.col_id]: (row[cfg.col_english], row[cfg.col_source])
        for _, row in done.iterrows()
    }


def merge_prior_translations(
    df: pd.DataFrame, prior: dict[int, tuple[str, str]], cfg: Config
) -> pd.DataFrame:
    if not prior:
        return df
    for row_id, (en, src) in prior.items():
        idx = df.index[df[cfg.col_id] == row_id]
        if len(idx) == 0:
            continue
        if not isinstance(df.at[idx[0], cfg.col_english], str) or not df.at[idx[0], cfg.col_english].strip():
            df.at[idx[0], cfg.col_english] = en
            df.at[idx[0], cfg.col_source] = src
    return df


def rows_needing_translation(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    has_ar = df[cfg.col_arabic].notna() & (df[cfg.col_arabic].astype(str).str.strip() != "")
    no_en = df[cfg.col_english].isna() | (df[cfg.col_english].astype(str).str.strip() == "")
    return df[has_ar & no_en].copy()


# ── Translation core ───────────────────────────────────────────────────


def build_translator(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    LOGGER.info("Loading model %s on %s", model_name, "GPU" if device == 0 else "CPU")
    return pipeline("translation", model=model_name, device=device)


def translate_batch(translator, texts: list[str], max_length: int) -> list[str]:
    if not texts:
        return []
    try:
        results = translator(texts, max_length=max_length, batch_size=len(texts))
        return [r["translation_text"] for r in results]
    except Exception:
        LOGGER.exception("Batch failed; returning empty strings for %d rows", len(texts))
        return [""] * len(texts)


# ── Output ─────────────────────────────────────────────────────────────


def build_output_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    mask = (
        df[cfg.col_franco].notna()
        & df[cfg.col_arabic].notna()
        & df[cfg.col_english].notna()
        & (df[cfg.col_english].astype(str).str.strip() != "")
    )
    cols = [cfg.col_id, cfg.col_franco, cfg.col_arabic, cfg.col_english, cfg.col_source]
    return df.loc[mask, cols].reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ── Orchestration ──────────────────────────────────────────────────────


def run(cfg: Config) -> int:
    df = load_annotations(cfg.input_path, cfg)
    LOGGER.info(
        "Loaded %d rows | arabic=%d english=%d",
        len(df),
        df[cfg.col_arabic].notna().sum(),
        df[cfg.col_english].notna().sum(),
    )

    prior = load_existing_output(cfg.output_path, cfg)
    if prior:
        LOGGER.info("Resuming: %d rows already in %s", len(prior), cfg.output_path)
        df = merge_prior_translations(df, prior, cfg)

    todo = rows_needing_translation(df, cfg)
    LOGGER.info("Rows to translate this run: %d", len(todo))
    if todo.empty:
        write_csv(build_output_frame(df, cfg), cfg.output_path)
        LOGGER.info("Nothing to translate. Output rewritten to %s", cfg.output_path)
        return 0

    translator = build_translator(cfg.model_name)

    start_time = time.perf_counter()
    total = len(todo)
    for batch_idx, start in enumerate(range(0, total, cfg.batch_size)):
        chunk = todo.iloc[start : start + cfg.batch_size]
        texts = chunk[cfg.col_arabic].astype(str).tolist()
        outputs = translate_batch(translator, texts, cfg.max_length)

        for row_id, en in zip(chunk[cfg.col_id], outputs):
            mask = df[cfg.col_id] == row_id
            df.loc[mask, cfg.col_english] = en
            df.loc[mask, cfg.col_source] = "auto" if en else pd.NA

        done = min(start + cfg.batch_size, total)
        LOGGER.info("Progress: %d/%d (%.1f%%)", done, total, done / total * 100)

        if batch_idx % CHECKPOINT_EVERY_N_BATCHES == 0:
            write_csv(build_output_frame(df, cfg), cfg.output_path)
            LOGGER.debug("Checkpointed at batch %d", batch_idx)

    elapsed = time.perf_counter() - start_time
    final = build_output_frame(df, cfg)
    write_csv(final, cfg.output_path)

    LOGGER.info("─" * 60)
    LOGGER.info("Done in %.1fs. Wrote %d rows to %s", elapsed, len(final), cfg.output_path)
    auto_count = (final[cfg.col_source] == "auto").sum()
    manual_count = (final[cfg.col_source] == "manual").sum()
    LOGGER.info("Breakdown: auto=%d, manual=%d", auto_count, manual_count)
    for _, row in final.head(3).iterrows():
        LOGGER.info("  franco : %s", str(row[cfg.col_franco])[:70])
        LOGGER.info("  arabic : %s", str(row[cfg.col_arabic])[:70])
        LOGGER.info("  english: %s", str(row[cfg.col_english])[:70])
        LOGGER.info("")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-translate Arabic annotations to English.")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    cfg = Config(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
