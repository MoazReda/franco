# Franco Translator 🇪🇬

> Automatically convert Egyptian Franco (Arabic written in Latin script) 
> to Arabic and English using fine-tuned NLP models.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## What is Franco?

Franco (فرانكو) is how millions of Egyptians write Arabic online — 
using Latin letters and digits to represent Arabic sounds:

| Franco | Arabic | English |
|--------|--------|---------|
| `3yzak tigi bukra` | عايزك تيجي بكره | I want you to come tomorrow |
| `el mawdo3 7elw awy` | الموضوع حلو أوي | The topic is really great |
| `5alas mesh moshkela` | خلاص مش مشكلة | It's fine, no problem |

No two people write it the same way. That's what makes it an 
interesting NLP challenge.

## Project Architecture
Franco text (input)
↓
Preprocessing & normalization
↓
Fine-tuned Seq2Seq model (HuggingFace)
↓
Arabic output  +  English output
↓
FastAPI backend  →  React web app

## Roadmap

- [x] Project setup & annotation guidelines
- [ ] Dataset collection (target: 50,000 sentences)
- [ ] EDA & baseline model
- [ ] Fine-tuning transformer model
- [ ] FastAPI backend
- [ ] React web app
- [ ] Docker & deployment

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