"""Fine-tuning utilities for the Franco↔Arabic Seq2Seq model.

Reusable building blocks called from both the Kaggle notebook
(``notebooks/03_finetune_arat5.ipynb``) and any local training script.

Design choices:
- A single model handles both directions via prefix tokens ``<2ar>`` (for
  Franco→Arabic) and ``<2franco>`` (for Arabic→Franco). The tokens are
  added to the tokenizer and the model's input embeddings are resized.
- Training data is built by emitting two rows per source pair — one for
  each direction. This doubles the effective training set.
- The augmented CSV (``train_augmented.csv``) is the training source.
  Val and test are never augmented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PREFIX_TO_AR = "<2ar>"
PREFIX_TO_FRANCO = "<2franco>"
SPECIAL_PREFIXES = [PREFIX_TO_AR, PREFIX_TO_FRANCO]


@dataclass(frozen=True)
class TrainingConfig:
    """All knobs in one place. Sensible defaults for AraT5v2-base on a T4."""

    model_name: str = "UBC-NLP/AraT5v2-base-1024"
    output_dir: str = "models/franco-translator-v1"

    # Data
    train_csv: str = "data/splits/train_augmented.csv"
    val_csv: str = "data/splits/val.csv"
    test_csv: str = "data/splits/test.csv"
    max_input_length: int = 128
    max_target_length: int = 128

    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    label_smoothing_factor: float = 0.1

    # Eval / logging
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "chrf"
    greater_is_better: bool = True
    logging_steps: int = 50

    # Generation (for eval)
    predict_with_generate: bool = True
    generation_max_length: int = 128
    generation_num_beams: int = 4

    # Misc
    seed: int = 42
    fp16: bool = True  # T4 supports fp16; set False on CPU
    report_to: tuple[str, ...] = field(default_factory=tuple)  # set ("wandb",) to log


# ── Data preparation ──────────────────────────────────────────────────


def build_bidirectional_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Emit two training rows per source: Franco→Arabic and Arabic→Franco.

    Each output row has columns ``input_text`` (with prefix) and
    ``target_text``.
    """
    rows = []
    for _, row in df.iterrows():
        franco = str(row["franco"]).strip()
        arabic = str(row["arabic"]).strip()
        if not franco or not arabic:
            continue
        rows.append(
            {
                "input_text": f"{PREFIX_TO_AR} {franco}",
                "target_text": arabic,
                "direction": "fa2ar",
            }
        )
        rows.append(
            {
                "input_text": f"{PREFIX_TO_FRANCO} {arabic}",
                "target_text": franco,
                "direction": "ar2fa",
            }
        )
    return pd.DataFrame(rows)


def load_split_as_bidirectional(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"franco", "arabic"}
    if not required.issubset(raw.columns):
        raise ValueError(
            f"{csv_path} is missing required columns {required - set(raw.columns)}"
        )
    return build_bidirectional_dataset(raw)


# ── Tokenizer / model setup ───────────────────────────────────────────


def setup_tokenizer_and_model(model_name: str):
    """Load tokenizer + model, add prefix tokens, resize embeddings.

    Returns ``(tokenizer, model)``. Imports are local so this module
    stays importable without ``transformers`` (e.g. in tests).
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_PREFIXES}
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def build_tokenized_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_input_length: int,
    max_target_length: int,
):
    """Tokenize input/target columns; return a HuggingFace Dataset."""
    from datasets import Dataset

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


# ── Metrics ───────────────────────────────────────────────────────────


def make_compute_metrics(tokenizer):
    """Return a ``compute_metrics`` callable suitable for Seq2SeqTrainer."""
    import sacrebleu
    import numpy as np

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        bleu = sacrebleu.corpus_bleu(
            decoded_preds, list(zip(*decoded_labels))
        ).score
        chrf = sacrebleu.corpus_chrf(
            decoded_preds, list(zip(*decoded_labels))
        ).score
        return {"bleu": bleu, "chrf": chrf}

    return compute_metrics


# ── Trainer factory ───────────────────────────────────────────────────


def build_trainer(cfg: TrainingConfig, tokenizer, model, train_ds, val_ds):
    """Construct a ``Seq2SeqTrainer`` from a ``TrainingConfig``."""
    from transformers import (
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        label_smoothing_factor=cfg.label_smoothing_factor,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        logging_steps=cfg.logging_steps,
        predict_with_generate=cfg.predict_with_generate,
        generation_max_length=cfg.generation_max_length,
        generation_num_beams=cfg.generation_num_beams,
        seed=cfg.seed,
        fp16=cfg.fp16,
        report_to=list(cfg.report_to),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )


# ── Inference helper ──────────────────────────────────────────────────


def predict(text: str, direction: str, tokenizer, model, max_new_tokens: int = 128) -> str:
    """Run a single prediction; ``direction`` ∈ {'fa2ar', 'ar2fa'}."""
    import torch

    prefix = PREFIX_TO_AR if direction == "fa2ar" else PREFIX_TO_FRANCO
    inputs = tokenizer(f"{prefix} {text}", return_tensors="pt", truncation=True).to(
        model.device
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()
