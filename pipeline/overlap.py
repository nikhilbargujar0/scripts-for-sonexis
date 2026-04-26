"""Two-tier speaker overlap detection and enrichment.

OVER-01: Detect overlapping speech regions from speaker turns; emit
         overlap_type, overlap_speakers, overlap_start, overlap_end.
OVER-02: Soft (<0.6 s, one side backchannel) → overlap_backchannel, no
         quality downgrade.  Hard (≥0.6 s, both sides content words) →
         overlap_crosstalk, routes to review queue.
OVER-03: Overlaps <0.2 s discarded (below reliable detection threshold).
OVER-04: Backchannel lexicon is profile-aware — uses per-profile FILLERS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Minimum duration for an overlap to be reported at all (OVER-03).
_MIN_OVERLAP_S = 0.2
# Duration boundary between soft (backchannel) and hard (crosstalk) overlaps (OVER-02).
_SOFT_HARD_BOUNDARY_S = 0.6

# Profile-aware backchannel lexicons (OVER-04).
# Keys match language_profiles profile names and BCP-47 bases.
_BACKCHANNEL_LEXICONS: Dict[str, Set[str]] = {
    "default": {
        "um", "uh", "uhh", "hmm", "ah", "ahh", "ok", "okay", "yeah", "yes",
        "no", "right", "sure", "mhm",
    },
    "hindi": {
        "haan", "ha", "achha", "acha", "theek", "theek hai", "sahi", "bilkul",
        "hmm", "hm", "arre", "oh", "wah", "waah", "bas", "na", "nahi",
        "matlab", "toh", "bhai", "yaar",
    },
    "punjabi": {
        "haan", "ha", "achha", "theek", "theek hai", "sahi", "bilkul", "hmm",
        "oh", "arre", "wah", "waah", "bas", "na", "nahi", "yaar",
    },
    "marwadi": {
        "haan", "ha", "achha", "theek", "hmm", "arre", "wah", "bas", "na",
    },
    "indian_english": {
        "yeah", "yes", "no", "ok", "okay", "right", "sure", "hmm", "uh",
        "um", "mhm", "ah", "oh", "i see", "got it",
    },
    "hinglish": {
        "haan", "ha", "achha", "acha", "theek", "hmm", "ok", "okay", "yeah",
        "yes", "right", "arre", "wah", "bas", "na", "nahi", "matlab", "toh",
    },
}
# BCP-47 base → profile name
_LANG_TO_PROFILE: Dict[str, str] = {
    "hi": "hindi", "pa": "punjabi", "mwr": "marwadi",
    "en": "indian_english", "hinglish": "hinglish",
}


@dataclass
class OverlapRegion:
    """One detected speaker overlap interval."""
    start: float
    end: float
    speakers: List[str]
    overlap_type: str       # "backchannel" | "crosstalk"
    duration: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration = round(max(0.0, self.end - self.start), 4)

    def to_dict(self) -> dict:
        return {
            "overlap_start": round(self.start, 4),
            "overlap_end": round(self.end, 4),
            "overlap_duration": self.duration,
            "overlap_type": self.overlap_type,
            "overlap_speakers": list(self.speakers),
        }


def _backchannel_lexicon(language: str) -> Set[str]:
    """OVER-04: return profile-aware backchannel word set."""
    base = str(language or "").lower().split("-")[0]
    profile = _LANG_TO_PROFILE.get(base, "default")
    lex = set(_BACKCHANNEL_LEXICONS.get("default", set()))
    lex.update(_BACKCHANNEL_LEXICONS.get(profile, set()))
    return lex


def _is_backchannel_text(text: str, language: str) -> bool:
    """True when *text* consists almost entirely of backchannel tokens."""
    lex = _backchannel_lexicon(language)
    tokens = [t.strip(".,!?;:'\"-").lower() for t in str(text or "").split()]
    if not tokens:
        return True   # empty text → treat as backchannel
    bc_count = sum(1 for t in tokens if t in lex)
    return bc_count / len(tokens) >= 0.7   # ≥70% backchannel tokens


def _classify_overlap(
    duration: float,
    seg_a_text: str,
    seg_a_lang: str,
    seg_b_text: str,
    seg_b_lang: str,
) -> str:
    """OVER-02: two-tier overlap classification.

    Soft (<0.6s, at least one side is backchannel) → backchannel.
    Hard (≥0.6s OR both sides are content) → crosstalk.
    """
    if duration < _SOFT_HARD_BOUNDARY_S:
        if _is_backchannel_text(seg_a_text, seg_a_lang) or \
           _is_backchannel_text(seg_b_text, seg_b_lang):
            return "backchannel"
    return "crosstalk"


def detect_overlaps(
    turns: List,
    segments: Optional[List] = None,
) -> List[OverlapRegion]:
    """OVER-01..04: detect and classify overlapping speech regions.

    Args:
        turns:    List of SpeakerTurn objects with .start, .end, .speaker.
        segments: Optional list of transcript segments to look up text/language
                  for overlap classification.

    Returns:
        List of OverlapRegion (OVER-03: filtered to ≥0.2 s).
    """
    if not turns:
        return []

    sorted_turns = sorted(turns, key=lambda t: t.start)

    # Build a segment lookup for text/language (best-effort)
    seg_at: List[Tuple[float, float, str, str]] = []  # (start, end, text, lang)
    for seg in (segments or []):
        seg_at.append((
            float(seg.start),
            float(seg.end),
            str(getattr(seg, "text", "") or ""),
            str(getattr(seg, "language", "") or ""),
        ))

    def _seg_text_lang_at(start: float, end: float) -> Tuple[str, str]:
        mid = (start + end) / 2.0
        best_text, best_lang = "", ""
        for ss, se, st, sl in seg_at:
            if ss <= mid <= se:
                return st, sl
            # fallback: any overlap
            if min(end, se) - max(start, ss) > 0:
                best_text, best_lang = st, sl
        return best_text, best_lang

    overlaps: List[OverlapRegion] = []
    for i, turn_a in enumerate(sorted_turns):
        for turn_b in sorted_turns[i + 1:]:
            if turn_b.start >= turn_a.end:
                break  # sorted → no more overlaps with turn_a
            if turn_a.speaker == turn_b.speaker:
                continue  # same speaker (shouldn't happen after merge, but guard)

            ov_start = max(turn_a.start, turn_b.start)
            ov_end = min(turn_a.end, turn_b.end)
            ov_dur = ov_end - ov_start

            # OVER-03: discard <0.2 s overlaps
            if ov_dur < _MIN_OVERLAP_S:
                continue

            text_a, lang_a = _seg_text_lang_at(turn_a.start, turn_a.end)
            text_b, lang_b = _seg_text_lang_at(turn_b.start, turn_b.end)
            ov_type = _classify_overlap(ov_dur, text_a, lang_a, text_b, lang_b)

            overlaps.append(OverlapRegion(
                start=round(ov_start, 4),
                end=round(ov_end, 4),
                speakers=[turn_a.speaker, turn_b.speaker],
                overlap_type=ov_type,
            ))

    return overlaps


def annotate_segments_with_overlaps(
    segments: Iterable,
    overlaps: List[OverlapRegion],
) -> None:
    """OVER-01: attach overlap metadata fields to each segment.

    Sets on each segment:
      overlap           bool
      overlap_type      "backchannel" | "crosstalk" | None
      overlap_speakers  list[str] | []
      overlap_start     float | None
      overlap_end       float | None
    """
    for seg in segments:
        s_start = float(seg.start)
        s_end = float(seg.end)
        matching: List[OverlapRegion] = [
            ov for ov in overlaps
            if min(s_end, ov.end) - max(s_start, ov.start) > 0
        ]

        if not matching:
            setattr(seg, "overlap", False)
            setattr(seg, "overlap_type", None)
            setattr(seg, "overlap_speakers", [])
            setattr(seg, "overlap_start", None)
            setattr(seg, "overlap_end", None)
            continue

        # Use the longest overlap if multiple hit this segment
        primary = max(matching, key=lambda ov: ov.duration)
        setattr(seg, "overlap", True)
        setattr(seg, "overlap_type", primary.overlap_type)
        setattr(seg, "overlap_speakers", list(primary.speakers))
        setattr(seg, "overlap_start", primary.start)
        setattr(seg, "overlap_end", primary.end)
