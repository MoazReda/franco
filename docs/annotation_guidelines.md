# Franco Annotation Guidelines

## What is Franco?

Franco (فرانكو) is Egyptian Arabic written using Latin characters and digits.
There is no single standard — this document defines the conventions used in
this dataset to ensure consistency across all annotators.

## Digit–Letter Mapping

| Digit | Arabic letter | Example Franco | Arabic |
| ----- | ------------- | -------------- | ------ |
| 2     | ء / أ / إ     | 2anta          | أنت    |
| 3     | ع             | 3ayzak         | عايزك   |
| 5     | خ             | 5alas          | خلاص   |
| 6     | ط             | 6ab            | طب     |
| 7     | ح             | 7elw           | حلو    |
| 8     | غ / ق         | 8alat          | غلط    |
| 9     | ص / ق         | 9a7            | صح     |

## Annotation Rules

### Rule 1 — Preserve the original Franco

Never correct spelling. Keep the text exactly as written.

### Rule 2 — Standard Arabic

Write clean Arabic text (عايزك تيجي بكره).

### Rule 3 — Natural English

Translate meaning, not word-for-word.

### Rule 4 — One sentence per row

Each row should be one full sentence (5–20 words).

### Rule 5 — Variants are allowed

Different spellings are valid and should be included.

## Dataset Format

franco,arabic,english,source,annotator,verified
"3yzak tigi bukra","عايزك تيجي بكره","I want you to come tomorrow","manual","moaz","true"

## Quality Checklist (before marking verified=true)
- [ ] Franco is the exact original text
- [ ] Arabic is grammatically correct
- [ ] English conveys the same meaning naturally
- [ ] No personally identifiable information
- [ ] Row length is between 5–20 words
