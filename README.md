# Franco Translator 🇪🇬

> Bidirectional translator between Egyptian Franco (Arabic written in Latin script)
> and standard Arabic, powered by a fine-tuned Seq2Seq model.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## What is Franco?

Franco (فرانكو) is how millions of Egyptians write Arabic online —
using Latin letters and digits to represent Arabic sounds:

| Franco | Arabic |
|--------|--------|
| `3yzak tigi bukra` | عايزك تيجي بكره |
| `el mawdo3 7elw awy` | الموضوع حلو أوي |
| `5alas mesh moshkela` | خلاص مش مشكلة |

No two people write it the same way. That's what makes it an
interesting NLP challenge.

> An earlier scope included English translation as a third target. After an empirical
> baseline (`experiments/01_arabic_to_english_baseline/`) showed that off-the-shelf
> MSA→EN models fail on ~40% of Egyptian colloquial sentences, English was dropped
> to focus the product on the actual user need: converting between Franco and Arabic.

## Project Architecture

```
Franco text  ──┐
                ├──►  Fine-tuned Seq2Seq (AraT5 / AraBART)  ──►  Arabic text
Arabic text  ──┘                  ▲                                │
                                  └────────  same model  ◄─────────┘
                                  (prefix tokens: <2ar> / <2franco>)
                                                                 │
                                                                 ▼
                                              FastAPI backend  →  React web app
```

## Roadmap

- [x] Project setup & annotation guidelines
- [x] Dataset collection (~18k raw → 13,373 cleaned Franco sentences)
- [x] EDA + manual annotation (445 Franco↔Arabic pairs)
- [x] Arabic→English baseline (dropped from scope after testing)
- [ ] Data augmentation (Franco spelling variants + rule-based AR→Franco generator)
- [ ] Rule-based Franco↔Arabic baseline + BLEU/chrF benchmark
- [ ] Fine-tune Seq2Seq model
- [ ] Publish to HuggingFace Hub with model card
- [ ] FastAPI backend
- [ ] React web app + HuggingFace Spaces deploy

## Dataset

Built from real Egyptian Franco found on Twitter, YouTube comments, 
and Reddit. Annotated following strict guidelines in 
[`docs/annotation_guidelines.md`](docs/annotation_guidelines.md).

Current size: 🔄 in progress

## Local Setup
```bash
git clone https://github.com/MoazReda/franco.git
cd franco
pip install -r requirements.txt
```

## Contributing

See [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md) 
for dataset contribution rules.

---

Built by [@MoazReda](https://github.com/MoazReda)