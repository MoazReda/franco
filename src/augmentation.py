"""Realistic Franco spelling-variant augmentation.

Each transformation models a real way Egyptians spell Franco differently.
The goal is to expand a small annotated dataset by generating plausible
alternative spellings for the same Arabic reference, so a downstream
Seq2Seq model sees more surface variation during training.

Important: only the Franco side is augmented. The Arabic reference stays
the same — same meaning, different spelling on the source side.

Determinism: passing the same ``seed`` and same input always yields the
same set of variants. The per-row RNG mixes the seed with the input text.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

# Digit → letter alternatives that real Egyptians use. The digit form is
# unambiguously Arabic-derived, so swapping it for the Latin equivalent is
# almost always safe.
DIGIT_TO_LETTER: dict[str, str] = {
    "3": "a",
    "7": "h",
    "2": "a",
    "5": "kh",
    "8": "gh",
    "9": "s",
    "6": "t",
    "4": "sh",
}

# Letter → digit is restricted: the Latin letters appear in English words
# too, so blanket swaps produce noise like ``discu99ion``. Only swap the
# letters that are most often actually representing Arabic emphatic /
# guttural sounds. Skip ``s``→9, ``t``→6, ``a``→2, etc. — too lossy.
LETTER_TO_DIGIT: dict[str, str] = {
    "a": "3",  # ayn (when in Arabic-Franco context)
    "h": "7",  # ha'
}

VOWELS = set("aeiou")

ARABIC_RE = re.compile(r"[؀-ۿ]")


# ── Atomic transformations ────────────────────────────────────────────


def digit_to_letter(text: str, rng: random.Random, prob: float = 0.5) -> str:
    """Probabilistically replace 3/7/2/5/8 with their letter equivalents."""
    out: list[str] = []
    for ch in text:
        if ch in DIGIT_TO_LETTER and rng.random() < prob:
            out.append(DIGIT_TO_LETTER[ch])
        else:
            out.append(ch)
    return "".join(out)


def letter_to_digit(text: str, rng: random.Random, prob: float = 0.25) -> str:
    """The inverse: swap ``a`` for ``3`` etc.

    Conservative by design (see LETTER_TO_DIGIT comment): only operates on
    a small whitelist of letters, skips word-initial positions, and uses a
    lower default probability than its sibling ``digit_to_letter``.
    """
    out: list[str] = []
    for i, ch in enumerate(text):
        if i == 0 or text[i - 1].isspace():
            out.append(ch)
            continue
        low = ch.lower()
        if low in LETTER_TO_DIGIT and rng.random() < prob:
            out.append(LETTER_TO_DIGIT[low])
        else:
            out.append(ch)
    return "".join(out)


def drop_short_vowels(text: str, rng: random.Random, prob: float = 0.5) -> str:
    """Drop word-medial short vowels — ``3ayez`` → ``3yz``.

    Skips the first character of each word (people rarely drop it) and
    avoids dropping when it would create empty words.
    """
    out: list[str] = []
    prev_was_boundary = True
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            out.append(ch)
            prev_was_boundary = True
            continue
        if prev_was_boundary:
            out.append(ch)
            prev_was_boundary = False
            continue
        if ch.lower() in VOWELS and rng.random() < prob:
            continue
        out.append(ch)
    return "".join(out)


def double_final_vowel(text: str, rng: random.Random, prob: float = 0.5) -> str:
    """``aywa`` → ``aywaa``. Only doubles a final vowel on a word."""
    tokens = re.split(r"(\s+)", text)
    out: list[str] = []
    for tok in tokens:
        if tok and tok[-1].lower() in VOWELS and rng.random() < prob:
            out.append(tok + tok[-1])
        else:
            out.append(tok)
    return "".join(out)


def swap_definite_article(text: str) -> str:
    """``el`` ↔ ``al`` — both are common."""
    if re.search(r"\bel\b", text, re.IGNORECASE):
        return re.sub(r"\bel\b", "al", text, flags=re.IGNORECASE)
    if re.search(r"\bal\b", text, re.IGNORECASE):
        return re.sub(r"\bal\b", "el", text, flags=re.IGNORECASE)
    return text


def join_definite_article(text: str) -> str:
    """``el shar7`` → ``elshar7``."""
    return re.sub(r"\b(el|al|il)\s+(\w)", r"\1\2", text, flags=re.IGNORECASE)


def split_definite_article(text: str) -> str:
    """``elshar7`` → ``el shar7`` — only when followed by a non-vowel."""
    return re.sub(
        r"\b(el|al|il)([bcdfghjklmnpqrstvwxz])",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )


def sh_to_4(text: str, rng: random.Random, prob: float = 0.5) -> str:
    """``sh`` ↔ ``4`` — niche but real."""
    if "sh" in text.lower():
        if rng.random() < prob:
            return re.sub(r"sh", "4", text, flags=re.IGNORECASE)
    if "4" in text and rng.random() < prob:
        return text.replace("4", "sh")
    return text


# ── Composition ───────────────────────────────────────────────────────


_TRANSFORMS: list[Callable[[str, random.Random], str]] = [
    lambda s, rng: digit_to_letter(s, rng),
    lambda s, rng: letter_to_digit(s, rng),
    lambda s, rng: drop_short_vowels(s, rng),
    lambda s, rng: double_final_vowel(s, rng),
    lambda s, rng: swap_definite_article(s),
    lambda s, rng: join_definite_article(s),
    lambda s, rng: split_definite_article(s),
    lambda s, rng: sh_to_4(s, rng),
]


def generate_variants(franco: str, n: int = 4, *, seed: int = 42) -> list[str]:
    """Return up to ``n`` distinct synthetic variants of the input.

    Does not include the original text. Empty / unchanged outputs are
    dropped silently. May return fewer than ``n`` if the input has few
    augmentable features (e.g. a single short word with no digits).
    """
    if not franco or not franco.strip():
        return []
    if ARABIC_RE.search(franco):
        # Don't augment Arabic-heavy / code-switched text — the rules
        # are tuned for Latin-script Franco.
        return []

    rng = random.Random(f"{seed}::{franco}")
    variants: list[str] = []
    seen: set[str] = {franco}

    # First pass: each transformation alone
    for transform in _TRANSFORMS:
        v = transform(franco, rng)
        if v and v not in seen:
            variants.append(v)
            seen.add(v)
        if len(variants) >= n:
            return variants[:n]

    # Second pass: stack 2-3 transformations
    indices = list(range(len(_TRANSFORMS)))
    for _ in range(n * 3):
        k = rng.randint(2, 3)
        picks = rng.sample(indices, k=k)
        v = franco
        for idx in picks:
            v = _TRANSFORMS[idx](v, rng)
        if v and v not in seen:
            variants.append(v)
            seen.add(v)
        if len(variants) >= n:
            break

    return variants[:n]
