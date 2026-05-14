"""Translation evaluation utilities.

Wraps sacrebleu so that the rest of the codebase always computes metrics
the same way. Two metrics:

- **BLEU**: standard machine-translation metric, n-gram precision.
  Harsh on short sentences and small test sets.
- **chrF**: character-level F-score. More forgiving for morphologically
  rich languages like Arabic and for short references.

Use ``evaluate_predictions`` from scripts; use ``bleu`` / ``chrf`` directly
in notebooks for quick checks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import sacrebleu


@dataclass(frozen=True)
class EvalReport:
    n: int
    bleu: float
    chrf: float
    exact_match: float
    sample_predictions: list[tuple[str, str, str]]  # (source, reference, prediction)

    def summary(self) -> str:
        return (
            f"n={self.n} | BLEU={self.bleu:.2f} | chrF={self.chrf:.2f} "
            f"| exact-match={self.exact_match * 100:.1f}%"
        )


def bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    refs = [list(references)]  # sacrebleu wants list-of-lists
    return sacrebleu.corpus_bleu(list(predictions), refs).score


def chrf(predictions: Sequence[str], references: Sequence[str]) -> float:
    refs = [list(references)]
    return sacrebleu.corpus_chrf(list(predictions), refs).score


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    hits = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return hits / len(predictions)


def evaluate_predictions(
    sources: Sequence[str],
    references: Sequence[str],
    predict_fn: Callable[[str], str],
    n_examples: int = 5,
) -> EvalReport:
    """Run a prediction function over sources and score the outputs.

    Parameters
    ----------
    sources : sequence of input strings
    references : ground-truth target strings, same length as ``sources``
    predict_fn : callable mapping one source string to one prediction string
    n_examples : number of (source, ref, pred) tuples to keep for inspection
    """
    if len(sources) != len(references):
        raise ValueError(
            f"sources/references length mismatch: {len(sources)} vs {len(references)}"
        )

    predictions = [predict_fn(s) for s in sources]

    sample = [
        (sources[i], references[i], predictions[i])
        for i in range(min(n_examples, len(predictions)))
    ]

    return EvalReport(
        n=len(predictions),
        bleu=bleu(predictions, references),
        chrf=chrf(predictions, references),
        exact_match=exact_match(predictions, references),
        sample_predictions=sample,
    )


def iterate_pairs(
    sources: Iterable[str], references: Iterable[str]
) -> Iterable[tuple[str, str]]:
    """Tiny helper for the notebooks — strict-zip Python 3.10+ style."""
    for s, r in zip(sources, references):
        yield s, r
