"""Data loading and deterministic train/val/test splits.

The annotated sheet contains 489 rows but only 445 have Arabic translations.
This module loads only the annotated rows, drops the now-unused English
column, and produces reproducible splits keyed on row id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
DEFAULT_SPLITS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class SplitPaths:
    train: Path
    val: Path
    test: Path


def load_annotated(path: Path) -> pd.DataFrame:
    """Return the annotated rows with two columns: franco, arabic."""
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    required = {"id", "franco", "arabic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    mask = (
        df["franco"].notna()
        & df["arabic"].notna()
        & (df["franco"].astype(str).str.strip() != "")
        & (df["arabic"].astype(str).str.strip() != "")
    )
    out = df.loc[mask, ["id", "franco", "arabic"]].copy()
    out["franco"] = out["franco"].astype(str).str.strip()
    out["arabic"] = out["arabic"].astype(str).str.strip()
    return out.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = DEFAULT_SPLITS,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not abs(sum(fractions) - 1.0) < 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))

    n = len(df)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def write_splits(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    out_dir: Path,
) -> SplitPaths:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = SplitPaths(
        train=out_dir / "train.csv",
        val=out_dir / "val.csv",
        test=out_dir / "test.csv",
    )
    splits[0].to_csv(paths.train, index=False, encoding="utf-8-sig")
    splits[1].to_csv(paths.val, index=False, encoding="utf-8-sig")
    splits[2].to_csv(paths.test, index=False, encoding="utf-8-sig")
    return paths


def load_splits(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(out_dir / "train.csv", encoding="utf-8-sig"),
        pd.read_csv(out_dir / "val.csv", encoding="utf-8-sig"),
        pd.read_csv(out_dir / "test.csv", encoding="utf-8-sig"),
    )
