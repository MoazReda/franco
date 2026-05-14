"""Rule-based bidirectional Franco ↔ Arabic transliteration.

This is the baseline that the fine-tuned Seq2Seq model must beat. It uses:

1. A small Egyptian-Arabic word dictionary for the most common words
2. Bigram char rules (sh, kh, gh, etc.) applied with longest-match
3. Single-char fallback rules

It deliberately doesn't handle context, ambiguity, or code-switching well —
the point is to establish a measurable floor.
"""

from __future__ import annotations

import re

# ── Franco → Arabic mappings ──────────────────────────────────────────


WORD_MAP_FA_TO_AR: dict[str, str] = {
    # Pronouns
    "ana": "أنا",
    "enta": "انت",
    "enti": "انتي",
    "enty": "انتي",
    "ehna": "احنا",
    "homma": "هما",
    "homa": "هما",
    # Particles / common short words
    "el": "الـ",
    "al": "الـ",
    "msh": "مش",
    "mesh": "مش",
    "fi": "في",
    "fy": "في",
    "fe": "في",
    "men": "من",
    "ma3": "مع",
    "3ala": "على",
    "3la": "على",
    "we": "و",
    "w": "و",
    "law": "لو",
    "lw": "لو",
    "bs": "بس",
    "bas": "بس",
    "ya": "يا",
    "leh": "ليه",
    "eh": "ايه",
    "fen": "فين",
    "fein": "فين",
    "emta": "امتى",
    "ezay": "ازاي",
    "ezzay": "ازاي",
    # Demonstratives
    "da": "ده",
    "de": "دي",
    "dol": "دول",
    # Verbs / adjectives (common)
    "3ayz": "عايز",
    "3yz": "عايز",
    "3aiz": "عايز",
    "3ayez": "عايز",
    "3ayza": "عايزة",
    "7elw": "حلو",
    "7elwa": "حلوة",
    "kwayes": "كويس",
    "kwayesa": "كويسة",
    "5alas": "خلاص",
    "kollo": "كله",
    "tayeb": "طيب",
    "6ab": "طب",
    "tab": "طب",
    "bukra": "بكره",
    "bokra": "بكره",
    "embare7": "امبارح",
    "delwa2ty": "دلوقتي",
    "delwa2ti": "دلوقتي",
    # Greetings / closures
    "shokran": "شكرا",
    "shukran": "شكرا",
    "ok": "اوكي",
    "okay": "اوكي",
    "yala": "يلا",
    "yalla": "يلا",
    "aywa": "ايوة",
    "aywaa": "ايوه",
    "la": "لا",
    "laa": "لا",
}

# Multi-character (bigram/trigram) rules — longest match first
BIGRAMS_FA_TO_AR: list[tuple[str, str]] = [
    ("sh", "ش"),
    ("kh", "خ"),
    ("th", "ث"),
    ("gh", "غ"),
    ("ch", "تش"),
    ("aa", "ا"),
    ("ee", "ي"),
    ("oo", "و"),
    ("ou", "و"),
    ("ai", "اي"),
    ("ay", "اي"),
    ("ei", "اي"),
]

CHAR_MAP_FA_TO_AR: dict[str, str] = {
    "a": "ا",
    "b": "ب",
    "c": "ك",
    "d": "د",
    "e": "ا",
    "f": "ف",
    "g": "ج",
    "h": "ه",
    "i": "ي",
    "j": "ج",
    "k": "ك",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "o": "و",
    "p": "ب",
    "q": "ق",
    "r": "ر",
    "s": "س",
    "t": "ت",
    "u": "و",
    "v": "ف",
    "w": "و",
    "x": "كس",
    "y": "ي",
    "z": "ز",
    # Franco digit-letter substitutions
    "2": "ء",
    "3": "ع",
    "4": "ش",
    "5": "خ",
    "6": "ط",
    "7": "ح",
    "8": "غ",
    "9": "ص",
}


# ── Arabic → Franco mappings ──────────────────────────────────────────


CHAR_MAP_AR_TO_FA: dict[str, str] = {
    "ا": "a",
    "أ": "a",
    "إ": "e",
    "آ": "aa",
    "ء": "2",
    "ب": "b",
    "ت": "t",
    "ث": "th",
    "ج": "g",
    "ح": "7",
    "خ": "5",
    "د": "d",
    "ذ": "z",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "9",
    "ض": "d",
    "ط": "6",
    "ظ": "z",
    "ع": "3",
    "غ": "8",
    "ف": "f",
    "ق": "2",  # Egyptian glottal-stop pronunciation of qaf
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",
    "ي": "y",
    "ى": "a",
    "ة": "a",
    "ؤ": "2",
    "ئ": "2",
    # Drop diacritics
    "َ": "",
    "ُ": "",
    "ِ": "",
    "ً": "",
    "ٌ": "",
    "ٍ": "",
    "ْ": "",
    "ّ": "",
    "ـ": "",
}

WORD_MAP_AR_TO_FA: dict[str, str] = {
    "أنا": "ana",
    "انت": "enta",
    "انتي": "enti",
    "احنا": "ehna",
    "في": "fi",
    "من": "men",
    "مع": "ma3",
    "على": "3la",
    "ده": "da",
    "دي": "de",
    "ايه": "eh",
    "ليه": "leh",
    "فين": "fen",
    "ازاي": "ezay",
    "مش": "msh",
    "بس": "bs",
    "يا": "ya",
    "و": "w",
    "لو": "law",
    "حلو": "7elw",
    "كويس": "kwayes",
    "خلاص": "5alas",
    "شكرا": "shokran",
    "يلا": "yala",
    "ايوة": "aywa",
    "لا": "la",
    "بكره": "bukra",
    "عايز": "3ayez",
    "عايزة": "3ayza",
    "كله": "kollo",
    "طب": "tab",
    "طيب": "tayeb",
}

ARABIC_CHAR_RANGE = re.compile(r"[؀-ۿ]")
ASCII_LETTER = re.compile(r"[A-Za-z]")
WORD_BOUNDARY = re.compile(r"(\s+|[^\w؀-ۿ\d]+)")


# ── Franco → Arabic ───────────────────────────────────────────────────


def _transliterate_word_fa_to_ar(word: str) -> str:
    """Apply bigram-then-char rules to a single Franco word."""
    out: list[str] = []
    i = 0
    while i < len(word):
        # Try bigrams (longest match first)
        matched = False
        for src, tgt in BIGRAMS_FA_TO_AR:
            if word[i : i + len(src)] == src:
                out.append(tgt)
                i += len(src)
                matched = True
                break
        if matched:
            continue
        ch = word[i]
        if ch in CHAR_MAP_FA_TO_AR:
            out.append(CHAR_MAP_FA_TO_AR[ch])
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def franco_to_arabic(text: str) -> str:
    """Convert a Franco (Latin-script Egyptian Arabic) sentence to Arabic.

    Strategy:
      1. Preserve already-Arabic chunks as-is.
      2. Lowercase Latin chunks, split into tokens, try the word dictionary.
      3. For dictionary misses, apply bigram + single-char fallback.
    """
    if not text or not text.strip():
        return ""

    tokens = WORD_BOUNDARY.split(text)
    out: list[str] = []
    for token in tokens:
        if not token:
            continue
        if ARABIC_CHAR_RANGE.search(token):
            out.append(token)
            continue
        if not ASCII_LETTER.search(token) and not any(c in "23456789" for c in token):
            out.append(token)
            continue

        lowered = token.lower()
        if lowered in WORD_MAP_FA_TO_AR:
            out.append(WORD_MAP_FA_TO_AR[lowered])
        else:
            out.append(_transliterate_word_fa_to_ar(lowered))
    return "".join(out)


# ── Arabic → Franco ───────────────────────────────────────────────────


def _transliterate_word_ar_to_fa(word: str) -> str:
    return "".join(CHAR_MAP_AR_TO_FA.get(ch, ch) for ch in word)


def arabic_to_franco(text: str) -> str:
    """Convert an Arabic sentence to a canonical Franco form.

    Always emits one canonical form per Arabic word — does not attempt to
    capture the spelling variation real writers produce.
    """
    if not text or not text.strip():
        return ""

    tokens = WORD_BOUNDARY.split(text)
    out: list[str] = []
    for token in tokens:
        if not token:
            continue
        if not ARABIC_CHAR_RANGE.search(token):
            out.append(token)
            continue
        # Strip the definite article suffix used in WORD_MAP_FA_TO_AR
        lookup = token.replace("الـ", "ال")
        if lookup in WORD_MAP_AR_TO_FA:
            out.append(WORD_MAP_AR_TO_FA[lookup])
        else:
            # Special: leading "ال" → "el"
            if lookup.startswith("ال") and len(lookup) > 2:
                out.append("el" + _transliterate_word_ar_to_fa(lookup[2:]))
            else:
                out.append(_transliterate_word_ar_to_fa(lookup))
    return "".join(out)
