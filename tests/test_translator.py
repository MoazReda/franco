"""Tests for the rule-based translator.

These don't aim for translation quality — they pin down the algorithm so
refactors don't silently regress the baseline number.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

from src.translator import arabic_to_franco, franco_to_arabic


# ── Franco → Arabic ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "franco,expected",
    [
        ("ana", "أنا"),
        ("msh", "مش"),
        ("3ayez", "عايز"),
        ("el", "الـ"),
        ("shokran", "شكرا"),
    ],
)
def test_franco_word_dict_hits(franco: str, expected: str) -> None:
    assert franco_to_arabic(franco) == expected


def test_franco_digit_substitutions() -> None:
    # 'ze7laneen' has no dict entry — bigrams + chars apply
    assert "ح" in franco_to_arabic("ze7laneen")
    assert "ع" in franco_to_arabic("a3taqed")
    assert "خ" in franco_to_arabic("5ales")


def test_franco_preserves_arabic_chunks() -> None:
    # Code-switched input: Arabic word + Franco word → Arabic word kept verbatim
    out = franco_to_arabic("ana عايز")
    assert "أنا" in out
    assert "عايز" in out


def test_franco_empty_input() -> None:
    assert franco_to_arabic("") == ""
    assert franco_to_arabic("   ") == ""


def test_franco_bigram_beats_chars() -> None:
    # 'sh' must map to ش, not س+ه
    out = franco_to_arabic("shokr")  # 'shokran' is in dict; 'shokr' is not
    assert out.startswith("ش")


# ── Arabic → Franco ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "arabic,expected",
    [
        ("أنا", "ana"),
        ("مش", "msh"),
        ("شكرا", "shokran"),
        ("في", "fi"),
    ],
)
def test_arabic_word_dict_hits(arabic: str, expected: str) -> None:
    assert arabic_to_franco(arabic) == expected


def test_arabic_definite_article_to_el() -> None:
    # Unknown word starting with ال should emit "el" prefix
    assert arabic_to_franco("المدرسة").startswith("el")


def test_arabic_char_mapping_covers_emphatic_consonants() -> None:
    assert arabic_to_franco("حلو") == "7elw"  # dict hit, but verifies fall-through too
    # Unknown word with 7/3/5
    out = arabic_to_franco("صحيح")
    assert "9" in out  # ص → 9
    assert "7" in out  # ح → 7


def test_arabic_drops_diacritics() -> None:
    # Diacritics must not appear in output
    out = arabic_to_franco("شُكْرًا")
    for d in "َُِْٰٓ":
        assert d not in out


def test_arabic_empty_input() -> None:
    assert arabic_to_franco("") == ""
    assert arabic_to_franco("   ") == ""


# ── Round-trip sanity ────────────────────────────────────────────────


def test_round_trip_preserves_dictionary_words() -> None:
    # Words in both dictionaries should survive a round trip exactly
    for word in ["ana", "msh", "shokran", "fi"]:
        ar = franco_to_arabic(word)
        back = arabic_to_franco(ar)
        assert back == word, f"{word} → {ar} → {back}"
