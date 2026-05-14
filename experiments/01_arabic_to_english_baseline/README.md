# Experiment 01 — Arabic → English baseline (Helsinki opus-mt-ar-en)

**Date:** 2026-05-14
**Decision:** English direction dropped from project scope. Kept here as documented baseline.

## Hypothesis

A pretrained Modern Standard Arabic → English translation model (`Helsinki-NLP/opus-mt-ar-en`) can produce usable English translations for the 445 manually-annotated Egyptian Arabic sentences in our dataset. If quality is acceptable, we'd auto-fill the English column to make the dataset trilingual (Franco / Arabic / English).

## Method

- **Model:** `Helsinki-NLP/opus-mt-ar-en` (~300 MB, MSA-trained MarianMT)
- **Input:** 445 manually-translated Egyptian Arabic sentences from `data/annotated/annotation_sheet.xlsx`
- **Setup:** CPU inference, batch size 16, max length 256
- **Runtime:** ~5 min 25 s for all 445 rows
- **Output:** `data/annotated/annotated_500.csv` (gitignored — contains personal data)

Run:

```bash
$env:HF_HOME = "C:\hf_cache"      # Required: ASCII cache path on this machine
python experiments/01_arabic_to_english_baseline/run.py
```

## Results — random 20-sample manual review

| Quality bucket | Count | Notes |
|----------------|-------|-------|
| ✅ Good | ~8 | Short, MSA-like sentences worked: `"ايوة احنا شغالين عليها"` → `"Yeah, we're working on it."` |
| ⚠️ Acceptable | ~5 | Meaning preserved but dropped clauses or details |
| ❌ Bad | ~7 | Slang, code-switching, and idioms broke completely |

### Examples of failure modes

| Franco | Arabic | Model output | Issue |
|--------|--------|--------------|-------|
| `Msh nazlin el term d` | `مش نازلين الترم ده` | "not going down that apron" | Egyptian `ترم` = semester; MSA = apron |
| `Yadeen omy 3la el shabora yaged3an` | `يا دين أمي على الشوبورة يا جدعان` | "Hey, Dean, Mom's on the soup, Gideon" | Slang, vocatives, and proper-name confusion |
| `Ah de al set bta3t AI` | `اه دي ال set بتاعة AI` | "A.D.L. set, A.I." | Code-switching tokens dropped |
| `Ana shoft el molash beta3o 3la YouTube` | `انا شوفت المولاش بتاعه على YouTube` | "I've seen the mulash on a regular YouTube" | OOV slang term left untranslated |

## Why it failed (root cause)

`opus-mt-ar-en` is trained predominantly on Modern Standard Arabic (news, religious texts, parliamentary records). Our dataset is **Egyptian colloquial**, which differs from MSA in:

- **Vocabulary** — many Egyptian words don't exist in MSA (`مولاش`, `شوبورة`, `جدعان`)
- **Grammar** — different verb conjugations, negation patterns (`مش` vs MSA `لا`/`ليس`)
- **Code-switching** — Egyptians routinely mix Arabic and English in the same sentence
- **False friends** — words that exist in both registers but mean different things (`ترم` = apron in MSA, semester in Egyptian)

## Decision

**English translation direction is dropped from the project scope.**

Rationale:
1. ~40-50% of auto translations would need manual fixing — expensive and yields only mediocre data.
2. The target user is an Egyptian who wants to convert between Franco and Arabic. English isn't part of the core value.
3. Removing English simplifies the model (2 directions instead of 6 pair-combinations), shrinks the training set requirements, and produces a cleaner product story.

The output CSV (`data/annotated/annotated_500.csv`) is preserved in personal data only — not committed — as a reference snapshot.

## Portfolio takeaway

This experiment is a **useful failure**: it validates the project's reason-to-exist. Generic MSA NLP models don't work for Egyptian dialect, which is exactly the gap the custom fine-tuned model is meant to fill.
