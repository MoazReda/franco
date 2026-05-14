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
│   └── data_utils.py     ← load + deterministic train/val/test split
├── scripts/
│   ├── prepare_splits.py
│   └── evaluate_baseline.py
├── tests/
│   └── test_translator.py
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
- [ ] **Phase 2b** — Data augmentation (Franco spelling variants, rule-based AR→Franco generator)
- [ ] Phase 2d — Fine-tune AR-aware Seq2Seq (AraT5v2 or AraBART) with `<2ar>`/`<2franco>` prefix tokens
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
