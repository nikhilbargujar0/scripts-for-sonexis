"""Punctuation metadata without corrupting word timestamps."""
from __future__ import annotations

import unicodedata
from typing import Iterable


def _major_script(text: str) -> str:
    counts = {"Deva": 0, "Guru": 0, "Latn": 0}
    for ch in text:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "DEVANAGARI" in name:
            counts["Deva"] += 1
        elif "GURMUKHI" in name:
            counts["Guru"] += 1
        elif "LATIN" in name:
            counts["Latn"] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] else "unknown"


def apply_punctuation_metadata(segments: Iterable, enabled: bool = True) -> None:
    """Attach punctuation as Word.trailing_punct only.

    No punctuation token is inserted, so word indices/timestamps stay stable.
    Hindi/Devanagari remains honestly marked as missing for this milestone.
    """
    for seg in segments:
        text = str(getattr(seg, "text", "") or "")
        script = _major_script(text)
        language = str(getattr(seg, "language", "") or "")
        words = list(getattr(seg, "words", []) or [])
        for word in words:
            token = str(getattr(word, "text", "") or "")
            trailing = ""
            while token and unicodedata.category(token[-1]).startswith("P"):
                trailing = token[-1] + trailing
                token = token[:-1]
            word.text = token or word.text
            if trailing:
                word.trailing_punct = trailing

        if not enabled:
            setattr(seg, "punctuation_applied", False)
            setattr(seg, "punct_skipped", "disabled")
            continue
        if language == "hi" or script == "Deva":
            setattr(seg, "punctuation_applied", False)
            setattr(seg, "punct_skipped", "hindi_model_unavailable")
            continue
        if language in {"en", "pa", "mr", "hi-Latn", "mwr-Latn"} and words:
            if not getattr(words[-1], "trailing_punct", ""):
                words[-1].trailing_punct = "."
            setattr(seg, "punctuation_applied", True)
            setattr(seg, "punct_skipped", None)
        else:
            setattr(seg, "punctuation_applied", False)
            setattr(seg, "punct_skipped", "unsupported_language")
