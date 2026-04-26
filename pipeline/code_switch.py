"""Code-switch enrichment for transcript segments and words."""
from __future__ import annotations

import unicodedata
from typing import Iterable, List

from .language_detection import (
    ROMAN_HINDI_MARKERS,
    ROMAN_MARWADI_MARKERS,
    ROMAN_PUNJABI_MARKERS,
)


def script_tag(text: str) -> str:
    scripts = set()
    for ch in text:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "DEVANAGARI" in name:
            scripts.add("Deva")
        elif "GURMUKHI" in name:
            scripts.add("Guru")
        elif "LATIN" in name:
            scripts.add("Latn")
    if len(scripts) > 1:
        return "mixed"
    return next(iter(scripts), "unknown")


def infer_word_language(word: str, segment_language: str) -> tuple[str, float]:
    token = "".join(ch for ch in str(word or "").lower() if ch.isalnum() or ch == "'")
    script = script_tag(token)
    if script == "Deva":
        return "hi", 0.9
    if script == "Guru":
        return "pa", 0.9
    if token in ROMAN_HINDI_MARKERS:
        return "hi-Latn", 0.78
    if token in ROMAN_PUNJABI_MARKERS:
        return "pa-Latn", 0.76
    if token in ROMAN_MARWADI_MARKERS:
        return "mwr-Latn", 0.76
    if script == "Latn":
        return "en", 0.68
    return segment_language or "und", 0.5


_SENT_PUNCT = frozenset(".?!")


def _is_sentence_boundary(words: list, switch_idx: int) -> bool:
    """True when the word immediately before *switch_idx* ends with sentence punctuation."""
    if switch_idx <= 0 or switch_idx > len(words):
        return False
    prev_word = words[switch_idx - 1]
    text = str(getattr(prev_word, "text", "") or "").rstrip()
    # check trailing_punct first (PUNC-02), then last char of text
    punct = str(getattr(prev_word, "trailing_punct", "") or "")
    if punct and punct[-1] in _SENT_PUNCT:
        return True
    return bool(text) and text[-1] in _SENT_PUNCT


def _is_tag_switch(langs: List[str], switch_idx: int) -> bool:
    """True when a single isolated word in a different language (tag-switching).

    A 'tag' is a brief insertion — one word sandwiched between majority-language
    words on both sides, or a sentence-initial/final single-word insertion.
    """
    n = len(langs)
    if n < 3:
        return False
    # The word that triggers the switch is at switch_idx; check its neighbours.
    word_lang = langs[switch_idx]
    left = langs[switch_idx - 1] if switch_idx > 0 else None
    right = langs[switch_idx + 1] if switch_idx + 1 < n else None
    # Same language on both sides → isolated insertion → tag
    if left and right and left == right and word_lang != left:
        return True
    return False


def _switch_type(
    prev_lang: str,
    next_lang: str,
    switch_idx: int,
    langs: List[str],
    words: list,
) -> str:
    """Classify a language switch per SEAME / Miami-Bangor typology.

    switch_type ∈ {inter_sentential, intra_sentential, intra_word, tag}
    """
    # intra_word: base language is the same (e.g. hi ↔ hi-Latn)
    if prev_lang.split("-")[0] == next_lang.split("-")[0]:
        return "intra_word"
    # tag: isolated single-word insertion
    if _is_tag_switch(langs, switch_idx):
        return "tag"
    # inter_sentential: switch happens at a sentence boundary
    if _is_sentence_boundary(words, switch_idx):
        return "inter_sentential"
    # default: switch within a sentence
    return "intra_sentential"


def enrich_code_switch_segments(segments: Iterable) -> None:
    for seg in segments:
        words = list(getattr(seg, "words", []) or [])
        langs: List[str] = []
        for word in words:
            lang, conf = infer_word_language(getattr(word, "text", ""), getattr(seg, "language", "und"))
            word.language = lang
            word.language_confidence = conf
            word.script = script_tag(getattr(word, "text", ""))
            langs.append(lang)

        if not langs:
            setattr(seg, "matrix_language", getattr(seg, "language", "und"))
            setattr(seg, "switch_points", [])
            setattr(seg, "cs_density", 0.0)
            continue

        counts: dict = {}
        for lang in langs:
            counts[lang] = counts.get(lang, 0) + 1
        matrix = max(counts, key=counts.get)
        switches = []
        for idx, (prev_lang, next_lang) in enumerate(zip(langs, langs[1:]), start=1):
            if prev_lang == next_lang:
                continue
            switches.append({
                "word_idx": idx,
                "from_lang": prev_lang,
                "to_lang": next_lang,
                "switch_type": _switch_type(prev_lang, next_lang, idx, langs, words),
            })
        setattr(seg, "matrix_language", matrix)
        setattr(seg, "switch_points", switches)
        setattr(seg, "cs_density", round(len(switches) / max(len(langs), 1) * 100.0, 4))
