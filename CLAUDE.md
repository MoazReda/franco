# CLAUDE.md — Franco Translator

Project memory loaded automatically by Claude Code. Update this file when conventions, scope, or stack change.

## What this project is

Bidirectional Egyptian-Arabic translator that converts between two forms of the same language:

- **Franco** — Egyptian Arabic written in Latin script with digit substitutions (`3yzak tigi bukra`)
- **Arabic** — standard Egyptian Arabic script (`عايزك تيجي بكره`)

A **single Seq2Seq model** handles both directions via prefix tokens (`<2ar>` for Franco→Arabic, `<2franco>` for Arabic→Franco).

> **Scope note (2026-05-14):** English translation was originally planned as a third target but was dropped after an empirical baseline (`experiments/01_arabic_to_english_baseline/`) showed pretrained MSA→EN models fail on ~40% of Egyptian colloquial sentences. See that folder's README for the full writeup.

## Why bidirectional matters

- **Franco → Arabic**: serves people who can read Arabic but receive messages written in Franco.
- **Arabic → Franco**: serves the inverse — typing aid for users who can read Arabic but type faster in Latin script. Also the foundation for a Chrome-extension product angle for the MENA market.

## Stack

| Layer | Library |
| --- | --- |
| Python | 3.11.9 (venv) |
| Data | pandas, numpy, openpyxl |
| EDA | matplotlib, seaborn |
| NLP/ML | transformers, torch, sentencepiece, datasets, accelerate, sacrebleu, scikit-learn |
| Scraping | google-api-python-client (YouTube), requests (Reddit) |
| API | FastAPI (Phase 2f) |
| Frontend | React (Phase 2f) |
| Deploy | HuggingFace Spaces (free GPU) |

## Repository layout

```
franco/
├── data/
│   ├── raw/              ← personal data — gitignored
│   ├── processed/        ← franco_dataset_v1.csv (13,373 cleaned sentences)
│   └── annotated/        ← annotation sheets — gitignored
├── notebooks/
│   └── 01_eda.ipynb      ← Phase 1 EDA
├── src/
│   ├── data_collection/  ← scrapers + parsers
│   ├── translator.py     ← rule-based Franco↔Arabic baseline
│   ├── evaluation.py     ← BLEU / chrF / exact-match
│   ├── data_utils.py     ← load + deterministic train/val/test split
│   ├── augmentation.py   ← Franco spelling-variant generator
│   └── training.py       ← AraT5 fine-tuning helpers
├── scripts/
│   ├── prepare_splits.py
│   ├── evaluate_baseline.py
│   ├── augment_train.py
│   └── evaluate_model.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 03_finetune_arat5.ipynb  ← Kaggle-ready training notebook
├── tests/
│   ├── test_translator.py
│   └── test_augmentation.py
├── experiments/
│   └── 01_arabic_to_english_baseline/  ← archived English experiment + writeup
├── docs/
│   └── annotation_guidelines.md
├── data/
│   ├── splits/           ← gitignored, derived from annotation_sheet.xlsx
│   └── reports/          ← gitignored, baseline predictions
├── .env                  ← gitignored — API keys live here
└── requirements.txt
```

## Conventions

- **Source files** are modules under `src/` — `python -m src.data_collection.X` is the canonical way to run a script (not `python src/data_collection/X.py`).
- **Logging not print**: use the `logging` module for any script that will live longer than a one-off cell. `print` is fine inside notebooks only.
- **Configs over hardcoding**: paths, model names, batch sizes belong in a config object or YAML, not literals scattered in code.
- **Checkpoint long jobs**: any script that processes >100 rows must support resume — write intermediate progress to disk so a crash at row 400/500 doesn't restart from zero.
- **Save with `utf-8-sig`** when writing CSVs that Excel will open (preserves Arabic).
- **PEP 8** + run `ruff check .` before committing. Type hints on public functions.
- **Tests**: pytest under `tests/`. Critical pure functions (e.g. `franco_ratio`) must have at least one test.

## Data conventions

- **Two-column parallel format**: `franco | arabic` is the working training/eval set.
- **No PII in any committed file**. WhatsApp data lives only in `data/raw/`.
- Augmented / synthetic rows must carry a flag column (e.g. `source=synthetic`) so they can be filtered out for evaluation.

## Franco digit map (canonical)

| Digit | Arabic | Notes |
| --- | --- | --- |
| 2 | ء / أ / إ | Hamza family |
| 3 | ع | |
| 5 | خ | |
| 6 | ط | |
| 7 | ح | |
| 8 | غ / ق | Context-dependent |
| 9 | ص / ق | Context-dependent |

## Roadmap

- [x] Phase 1 — Data collection (18,125 raw → 13,373 cleaned)
- [x] Phase 1.5 — Manual annotation (445 / 489 sentences translated to Arabic)
- [x] Phase 2a — Tested Arabic→English baseline; dropped from scope (see `experiments/01_arabic_to_english_baseline/`)
- [x] Phase 2c — Rule-based Franco↔Arabic baseline + BLEU/chrF benchmark (see below)
- [x] Phase 2b — Data augmentation (Franco spelling variants, ×5 multiplier)
- [ ] **Phase 2d** — Fine-tune AraT5v2-base on Kaggle GPU (notebook ready, training pending)
- [ ] Phase 2e — Publish model card to HuggingFace Hub
- [ ] Phase 2f — FastAPI backend + React frontend + HF Spaces deploy

## Baseline numbers (rule-based, test split n=45, seed=42)

These are the floor that the fine-tuned Seq2Seq must beat:

| Direction | BLEU | chrF | Exact-match |
|-----------|------|------|-------------|
| Franco → Arabic | 3.94 | 33.90 | 0% |
| Arabic → Franco | 15.38 | 42.52 | 0% |

Regenerate after any rule change:
```
python scripts/prepare_splits.py --input <abs-path-to-xlsx> --output-dir <abs-path>
python scripts/evaluate_baseline.py --splits-dir <abs-path>
```

## Augmented training set

`scripts/augment_train.py` generates spelling variants for the **train split only**.
Val / test are never augmented — that would invalidate evaluation.

Current run: 356 training rows → **1,779 rows (×5 multiplier)**, 4 synthetic variants per source row. Transformations include digit↔letter swaps (3↔a, 7↔h), word-medial vowel drops, definite-article swap (el↔al), word joining/splitting, and final-vowel doubling. See `src/augmentation.py` for the full list.

## Training (Phase 2d) — Kaggle workflow

The trained model is too big for git (~1GB). Training happens on Kaggle GPU; the artifact lives on the HuggingFace Hub.

**Setup once:**
1. Upload `data/splits/{train_augmented.csv, val.csv, test.csv}` as a private Kaggle dataset called `franco-translator-data`.
2. (Optional) Add a Kaggle Secret named `HF_TOKEN` with a HuggingFace write token.

**Run:**
1. Open `notebooks/03_finetune_arat5.ipynb` on Kaggle.
2. Settings → Accelerator → GPU T4 x2 (or P100).
3. Run All.
4. Download `franco-translator-v1` from `/kaggle/working/` or set `PUSH_TO_HUB=True` to publish.

**Architecture:** single AraT5v2-base model, bidirectional via prefix tokens:
- `<2ar>` prepended → Franco→Arabic
- `<2franco>` prepended → Arabic→Franco

Both directions train jointly, doubling the effective training set per source row.

**Eval after training** (locally with the downloaded checkpoint):
```
python scripts/evaluate_model.py --model models/franco-translator-v1 \
    --splits-dir <abs-path-to-data/splits>
```
The script reports BLEU/chrF for both directions and the Δ against the rule-based baseline.

## Working agreements with Claude

- **Don't over-engineer.** No premature abstraction. Three-line repetition beats a bad helper.
- **Don't add features beyond what the task asks.** Bug-fix means bug-fix; refactor is a separate task.
- **Egyptian Arabic + English mix** in responses, technical terms stay English.
- **Connect work to portfolio/business value** in 2-3 lines, not lectures.
- **No mock data in training.** Train and evaluate on real annotated data only — synthetic data must be flagged.
- **Never commit `data/raw/`, `data/annotated/`, `.env`, or `venv/`.**

## Common commands

```bash
# Environment
python -m venv venv
.\venv\Scripts\activate              # Windows PowerShell
pip install -r requirements.txt

# Run a script (module form)
python -m src.data_collection.youtube_scraper

# HuggingFace cache must live on an ASCII-only path on this machine.
# The username folder contains non-ASCII characters which break model file
# resolution. Set this env var before running any transformers code:
$env:HF_HOME = "C:\hf_cache"

# Tests
pytest tests/ -v

# Lint
ruff check .
ruff format .
```
