"""Tests for the Franco spelling-variant augmentation."""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.augmentation import (
    digit_to_letter,
    double_final_vowel,
    drop_short_vowels,
    generate_variants,
    join_definite_article,
    letter_to_digit,
    sh_to_4,
    split_definite_article,
    swap_definite_article,
)


def test_digit_to_letter_swaps_3_to_a() -> None:
    out = digit_to_letter("3ayez", random.Random(0), prob=1.0)
    assert out == "aayez"


def test_digit_to_letter_swaps_5_to_kh() -> None:
    out = digit_to_letter("5alas", random.Random(0), prob=1.0)
    assert out == "khalas"


def test_digit_to_letter_zero_prob_is_identity() -> None:
    assert digit_to_letter("3ayez", random.Random(0), prob=0.0) == "3ayez"


def test_letter_to_digit_skips_word_initial() -> None:
    # 'ana' starts with 'a' — should NOT be swapped to '3' at position 0
    out = letter_to_digit("ana", random.Random(0), prob=1.0)
    assert out.startswith("a"), f"expected leading 'a' preserved, got: {out}"


def test_drop_short_vowels_skips_first_char() -> None:
    out = drop_short_vowels("ana", random.Random(0), prob=1.0)
    assert out.startswith("a")  # never drop initial


def test_double_final_vowel_appends() -> None:
    out = double_final_vowel("aywa", random.Random(0), prob=1.0)
    assert out == "aywaa"


def test_double_final_vowel_skips_consonant_ending() -> None:
    out = double_final_vowel("msh", random.Random(0), prob=1.0)
    assert out == "msh"  # ends in consonant — unchanged


def test_swap_definite_article_el_to_al() -> None:
    assert swap_definite_article("el shar7") == "al shar7"


def test_swap_definite_article_al_to_el() -> None:
    assert swap_definite_article("al kitab") == "el kitab"


def test_swap_definite_article_neither_present() -> None:
    assert swap_definite_article("3ayez") == "3ayez"


def test_join_definite_article() -> None:
    assert join_definite_article("el shar7") == "elshar7"


def test_split_definite_article() -> None:
    assert split_definite_article("elshar7") == "el shar7"


def test_sh_to_4_swaps_one_way() -> None:
    out = sh_to_4("shokran", random.Random(0), prob=1.0)
    assert "4" in out


def test_generate_variants_returns_distinct() -> None:
    variants = generate_variants("3ayez el shar7", n=5)
    assert len(variants) == len(set(variants)), "variants must be distinct"
    assert "3ayez el shar7" not in variants, "must not include original"


def test_generate_variants_deterministic() -> None:
    a = generate_variants("3ayez el shar7", n=5, seed=42)
    b = generate_variants("3ayez el shar7", n=5, seed=42)
    assert a == b


def test_generate_variants_different_seeds_differ() -> None:
    a = generate_variants("3ayez el shar7 da", n=5, seed=42)
    b = generate_variants("3ayez el shar7 da", n=5, seed=7)
    # Should differ on at least one element (probabilistic transforms)
    assert a != b


def test_generate_variants_skips_arabic_input() -> None:
    assert generate_variants("عايز الشرح", n=5) == []


def test_generate_variants_empty_input() -> None:
    assert generate_variants("", n=5) == []
    assert generate_variants("   ", n=5) == []


def test_generate_variants_caps_at_n() -> None:
    out = generate_variants("3ayez el shar7 da kollo", n=3)
    assert len(out) <= 3
