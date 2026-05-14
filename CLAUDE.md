# CLAUDE.md — Franco Translator

Project memory loaded automatically by Claude Code. Update this file when conventions, scope, or stack change.

## What this project is

Bidirectional Egyptian-Arabic translator that converts between three forms of the same language:

- **Franco** — Egyptian Arabic written in Latin script with digit substitutions (`3yzak tigi bukra`)
- **Arabic** — standard Egyptian Arabic script (`عايزك تيجي بكره`)
- **English** — natural meaning translation (`I want you to come tomorrow`)

A **single multilingual Seq2Seq model** handles all six direction-pairs via prefix tokens (`<2ar>`, `<2en>`, `<2franco>`).

## Why bidirectional matters

- **Franco → AR/EN**: serves people who can read Arabic but get messages in Franco.
- **AR → Franco**: serves the inverse — and exposes a Chrome-extension / typing-aid product angle for the MENA market.

## Stack

| Layer | Library |
| --- | --- |
| Python | 3.11.9 (venv) |
| Data | pandas, numpy, openpyxl |
| EDA | matplotlib, seaborn |
| NLP/ML | transformers, torch, sentencepiece, datasets, accelerate, evaluate, sacrebleu |
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
│   └── data_collection/  ← scrapers + parsers + auto_translate
├── docs/
│   └── annotation_guidelines.md
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

- **Three-column parallel format**: `franco | arabic | english` — every annotated row carries all three forms after Phase 2a.
- **Quality flags**: rows produced by auto-translation get `auto_en=true` so we can spot-check them separately from manually-verified rows.
- **No PII in any committed file**. WhatsApp data lives only in `data/raw/`.

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
- [ ] **Phase 2a** — Auto-translate Arabic → English (in progress)
- [ ] Phase 2b — Data augmentation (Franco spelling variants, rule-based AR→Franco generator, back-translation)
- [ ] Phase 2c — Rule-based baseline + BLEU/chrF benchmark
- [ ] Phase 2d — Fine-tune multilingual Seq2Seq (AraT5v2 or mT5) with prefix-token direction control
- [ ] Phase 2e — Publish model card to HuggingFace Hub
- [ ] Phase 2f — FastAPI backend + React frontend + HF Spaces deploy

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
python -m src.data_collection.auto_translate

# Tests
pytest tests/ -v

# Lint
ruff check .
ruff format .
```
