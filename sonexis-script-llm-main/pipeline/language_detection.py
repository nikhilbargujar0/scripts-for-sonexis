"""language_detection.py

Language + code-switching detection for Indian conversational data.

Two complementary layers:

1. **Script-based heuristic** (zero-dependency, always available):
   - Detects Devanagari (Hindi/Marwadi), Gurmukhi (Punjabi), Latin.
   - Token-level stats to flag code switching inside a single utterance.

2. **FastText lid.176** (optional, local model file, no API):
   - Called per segment to refine language labels when enough chars present.
   - Graceful fallback to heuristic if model file unavailable.

Segment-level detection uses Whisper TranscriptSegment start/end timestamps
to build `language_segments` with real time boundaries, enabling downstream
tools to map language to speaker turns accurately.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .transcription import TranscriptSegment

log = logging.getLogger(__name__)


# Romanised Hindi / Hinglish markers — deliberately compact, not exhaustive.
# Excludes tokens that are also common English words to avoid false positives.
ROMAN_HINDI_MARKERS = {
    "hai", "nahi", "nahin", "kya", "kyun", "kyunki", "haan", "bhai",
    "acha", "accha", "theek", "thik", "matlab", "yaar", "bas", "abhi",
    "ab", "toh", "mera", "meri", "tera", "teri", "apna", "apni",
    "kaise", "kaisa", "kuch", "kuchh", "sab", "sabhi", "bahut", "bohot",
    "chalo", "batao", "bolo", "suno", "dekho", "karna", "karo",
    "raha", "rahi", "rahe", "tha", "thi", "hoon", "hun",
    "hain", "ji", "yaara", "arre", "arey", "oye",
    "mai", "tum", "aap", "wala", "wali", "wale",
}

ROMAN_PUNJABI_MARKERS = {
    "ki", "kiven", "paaji", "veerji", "sat", "sri", "akaal", "shukriya",
    "bhaji", "tusi", "tussi", "asi", "aapan", "mainu", "tenu",
}

ROMAN_MARWADI_MARKERS = {
    "thara", "tharo", "mhare", "mhaari", "kai", "padharo",
}


@dataclass
class LanguageSegment:
    start: float
    end: float
    language: str
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "language": self.language,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class LanguageReport:
    primary_language: str
    confidence: float
    dominant_language: str = ""      # most common language by duration/segment count
    code_switching: bool = False
    multilingual_flag: bool = False  # True when 2+ distinct languages detected
    switching_frequency: float = 0.0 # language switches per minute of audio
    scripts: List[str] = field(default_factory=list)
    switching_score: float = 0.0     # normalized [0, 1] score for dataset filtering
    switching_pattern: Dict[str, int] = field(default_factory=dict)
    # switching_pattern: e.g. {"hi→en": 3, "en→hi": 2} — observed transition counts
    language_segments: List[LanguageSegment] = field(default_factory=list)
    # Backward-compat: per_segment kept as flat list (text + language only).
    per_segment: List[Dict] = field(default_factory=list)
    method: str = "heuristic"

    def __post_init__(self) -> None:
        if not self.dominant_language:
            self.dominant_language = self.primary_language

    def to_dict(self) -> dict:
        return {
            "primary_language": self.primary_language,
            "confidence": round(float(self.confidence), 3),
            "dominant_language": self.dominant_language,
            "code_switching": bool(self.code_switching),
            "multilingual_flag": bool(self.multilingual_flag),
            "switching_frequency": round(float(self.switching_frequency), 4),
            "switching_score": round(float(self.switching_score), 4),
            "switching_pattern": dict(self.switching_pattern),
            "scripts": list(self.scripts),
            "language_segments": [s.to_dict() for s in self.language_segments],
            "per_segment": self.per_segment,
            "method": self.method,
        }


def _detect_scripts(text: str) -> List[str]:
    scripts = set()
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "DEVANAGARI" in name:
            scripts.add("Devanagari")
        elif "GURMUKHI" in name:
            scripts.add("Gurmukhi")
        elif "LATIN" in name:
            scripts.add("Latin")
        elif "BENGALI" in name:
            scripts.add("Bengali")
        elif "TAMIL" in name:
            scripts.add("Tamil")
        elif "TELUGU" in name:
            scripts.add("Telugu")
    return sorted(scripts)


def _tokenise(text: str) -> List[str]:
    return re.findall(r"[\w']+", text.lower())


def _roman_indic_probe(tokens: List[str]) -> Tuple[str, float]:
    """Return (language_code, strength) if enough Indic markers found."""
    if not tokens:
        return ("en", 0.0)
    hits_hi = sum(1 for t in tokens if t in ROMAN_HINDI_MARKERS)
    hits_pa = sum(1 for t in tokens if t in ROMAN_PUNJABI_MARKERS)
    hits_mw = sum(1 for t in tokens if t in ROMAN_MARWADI_MARKERS)

    total = len(tokens)
    best = max((hits_hi, "hi-Latn"), (hits_pa, "pa-Latn"), (hits_mw, "mwr-Latn"),
               key=lambda x: x[0])
    hits, lang = best
    ratio = hits / max(total, 1)
    if hits >= 2 and ratio >= 0.05:
        return (lang, min(1.0, ratio * 4))
    return ("en", 0.0)


def _classify_segment_language(
    text: str,
    fasttext_lid: "FastTextLID | None",
    roman_indic_classifier,
) -> Tuple[str, float]:
    """Return (language_code, confidence) for a single segment text."""
    seg_scripts = _detect_scripts(text)
    seg_tokens = _tokenise(text)

    if "Devanagari" in seg_scripts:
        return ("hi", 0.9)
    if "Gurmukhi" in seg_scripts:
        return ("pa", 0.9)

    # Latin-only: try ML classifier → lexicon → fasttext → fallback
    seg_lang, seg_conf = "en", 0.6
    ml_used = False

    if roman_indic_classifier is not None and roman_indic_classifier.available():
        pred = roman_indic_classifier.predict(text)
        if pred is not None and pred.confidence >= 0.55:
            seg_lang, seg_conf = pred.language, pred.confidence
            ml_used = True

    if not ml_used:
        probed_lang, strength = _roman_indic_probe(seg_tokens)
        if probed_lang != "en" and strength > 0.1:
            seg_lang = probed_lang
            seg_conf = 0.55 + 0.3 * strength

    if fasttext_lid and fasttext_lid.available() and text.strip() and not ml_used:
        ft_lang, ft_conf = fasttext_lid.predict(text)
        if ft_lang != "und":
            probed_lang, strength = _roman_indic_probe(seg_tokens)
            if probed_lang != "en" and strength > 0.2:
                seg_lang = probed_lang
            else:
                seg_lang, seg_conf = ft_lang, ft_conf

    return seg_lang, seg_conf


class FastTextLID:
    """Wraps the lid.176 fasttext model if the user has downloaded it."""

    DEFAULT_PATH = os.environ.get(
        "FASTTEXT_LID_MODEL",
        os.path.expanduser("~/.cache/sonexis/lid.176.ftz"),
    )

    def __init__(self, path: Optional[str] = None):
        self.path = path or self.DEFAULT_PATH
        self._model = None

    def available(self) -> bool:
        return os.path.isfile(self.path)

    def _load(self):
        if self._model is None:
            import fasttext
            self._model = fasttext.load_model(self.path)
        return self._model

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.available() or not text.strip():
            return ("und", 0.0)
        try:
            model = self._load()
            labels, probs = model.predict(text.replace("\n", " "), k=1)
            lang = labels[0].replace("__label__", "") if labels else "und"
            return (lang, float(probs[0]) if len(probs) else 0.0)
        except Exception as err:
            log.warning("fasttext predict failed: %s", err)
            return ("und", 0.0)


def _compute_switching_frequency(
    language_segments: List[LanguageSegment],
    total_duration_s: float,
) -> float:
    """Return language switches per minute."""
    if len(language_segments) < 2 or total_duration_s <= 0:
        return 0.0
    switches = sum(
        1 for a, b in zip(language_segments, language_segments[1:])
        if a.language != b.language
    )
    return round(switches / (total_duration_s / 60.0), 4)


def _compute_switching_pattern(
    language_segments: List[LanguageSegment],
) -> Dict[str, int]:
    """Count directional language transitions.

    Returns a dict like ``{"hi→en": 3, "en→hi": 2}`` that shows which
    direction code-switching flows and how often.  Only consecutive segments
    with different languages are counted.
    """
    pattern: Dict[str, int] = {}
    for a, b in zip(language_segments, language_segments[1:]):
        if a.language != b.language:
            key = f"{a.language}\u2192{b.language}"
            pattern[key] = pattern.get(key, 0) + 1
    return pattern


def detect_language(
    full_text: str,
    segments_text: Optional[List[str]] = None,
    fasttext_lid: Optional[FastTextLID] = None,
    roman_indic_classifier=None,
    # New: pass TranscriptSegment list directly for timestamp-aware detection.
    transcript_segments: "Optional[List[TranscriptSegment]]" = None,
    total_duration_s: float = 0.0,
) -> LanguageReport:
    """Return a combined language + code-switching report.

    When transcript_segments is provided (preferred), language_segments will
    have real start/end timestamps. Otherwise falls back to segments_text
    (text-only, no timestamps).

    Resolution order for Latin-only text:
      1. roman_indic_classifier (trained ML model) if supplied.
      2. Romanised-Indic lexicon probe.
      3. FastText lid.176 if available.
      4. Fall back to "en" at 0.6 confidence.
    """
    segments_text = segments_text or []
    scripts = _detect_scripts(full_text)
    tokens = _tokenise(full_text)

    # Script-based global baseline.
    if "Devanagari" in scripts and "Latin" not in scripts:
        primary, conf = "hi", 0.9
    elif "Gurmukhi" in scripts and "Latin" not in scripts:
        primary, conf = "pa", 0.9
    elif "Devanagari" in scripts and "Latin" in scripts:
        primary, conf = "hi", 0.75
    elif "Gurmukhi" in scripts and "Latin" in scripts:
        primary, conf = "pa", 0.75
    else:
        primary, conf = "en", 0.6
        ml_used = False
        if roman_indic_classifier is not None and roman_indic_classifier.available():
            pred = roman_indic_classifier.predict(full_text)
            if pred is not None and pred.confidence >= 0.55:
                primary, conf = pred.language, pred.confidence
                ml_used = True
        if not ml_used:
            probed_lang, probe_strength = _roman_indic_probe(tokens)
            if probed_lang != "en":
                primary, conf = probed_lang, 0.55 + 0.3 * probe_strength

    method = "heuristic"
    if roman_indic_classifier is not None and roman_indic_classifier.available():
        method = "ml+heuristic"

    if fasttext_lid and fasttext_lid.available() and full_text.strip():
        ft_lang, ft_conf = fasttext_lid.predict(full_text)
        method = "ml+fasttext+heuristic" if "ml" in method else "fasttext+heuristic"
        if ft_lang != "und":
            if scripts == ["Devanagari"]:
                primary, conf = ("hi", max(conf, ft_conf))
            elif scripts == ["Gurmukhi"]:
                primary, conf = ("pa", max(conf, ft_conf))
            elif scripts == ["Latin"] and "ml" not in method:
                probed_lang, probe_strength = _roman_indic_probe(tokens)
                if probed_lang != "en" and probe_strength > 0.2:
                    primary = probed_lang
                else:
                    primary, conf = ft_lang, ft_conf

    # ------------------------------------------------------------------ #
    #  Per-segment language detection with timestamps.
    # ------------------------------------------------------------------ #
    language_segments: List[LanguageSegment] = []
    per_segment: List[Dict] = []
    seen_langs: Counter = Counter()

    if transcript_segments:
        for seg in transcript_segments:
            if not seg.text.strip():
                continue
            seg_lang, seg_conf = _classify_segment_language(
                seg.text, fasttext_lid, roman_indic_classifier
            )
            language_segments.append(LanguageSegment(
                start=seg.start,
                end=seg.end,
                language=seg_lang,
                confidence=seg_conf,
            ))
            per_segment.append({"text": seg.text, "language": seg_lang})
            seg_dur = max(0.0, seg.end - seg.start)
            seen_langs[seg_lang] += seg_dur if seg_dur > 0 else 1
    else:
        # Fallback: text-only (no timestamps available).
        for text in segments_text:
            if not text.strip():
                continue
            seg_lang, seg_conf = _classify_segment_language(
                text, fasttext_lid, roman_indic_classifier
            )
            per_segment.append({"text": text, "language": seg_lang})
            seen_langs[seg_lang] += 1

    # Dominant language = highest duration/count.
    dominant_language = seen_langs.most_common(1)[0][0] if seen_langs else primary

    # Use transcript_segments total duration if not supplied.
    dur = total_duration_s
    if dur <= 0 and transcript_segments:
        dur = max((s.end for s in transcript_segments), default=0.0)

    switching_freq = _compute_switching_frequency(language_segments, dur)
    switching_score = float(min(1.0, switching_freq / 10.0))
    multilingual_flag = len(seen_langs) >= 2
    code_switching = multilingual_flag or len(scripts) >= 2
    switching_pattern = _compute_switching_pattern(language_segments)

    return LanguageReport(
        primary_language=primary,
        confidence=float(conf),
        dominant_language=dominant_language,
        code_switching=bool(code_switching),
        multilingual_flag=bool(multilingual_flag),
        switching_frequency=switching_freq,
        switching_score=switching_score,
        switching_pattern=switching_pattern,
        scripts=scripts,
        language_segments=language_segments,
        per_segment=per_segment,
        method=method,
    )


def detect_language_per_speaker(
    turns: "List",  # List[SpeakerTurn]
    transcript_segments: "Optional[List[TranscriptSegment]]",
    fasttext_lid: Optional[FastTextLID] = None,
    roman_indic_classifier=None,
) -> Dict[str, LanguageReport]:
    """Detect language for each speaker independently.

    Maps transcript segments to speakers by finding the turn that best
    overlaps each segment, then builds a per-speaker LanguageReport.

    Returns {speaker_id: LanguageReport}.
    """
    if not turns or not transcript_segments:
        return {}

    # Build a quick list of (start, end, speaker) from turns for overlap lookup.
    turn_list = sorted(turns, key=lambda t: t.start)

    def _best_speaker_for_segment(seg_start: float, seg_end: float) -> str:
        """Return the speaker whose turn most overlaps this segment."""
        best_spk = turn_list[0].speaker
        best_overlap = 0.0
        for turn in turn_list:
            if turn.start > seg_end:
                break
            overlap = min(turn.end, seg_end) - max(turn.start, seg_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = turn.speaker
        return best_spk

    # Partition transcript segments by speaker.
    spk_segs: Dict[str, list] = {}
    for seg in transcript_segments:
        if not seg.text.strip():
            continue
        spk = _best_speaker_for_segment(seg.start, seg.end)
        spk_segs.setdefault(spk, []).append(seg)

    reports: Dict[str, LanguageReport] = {}
    for spk, segs in spk_segs.items():
        full_text = " ".join(s.text for s in segs)
        total_dur = max((s.end for s in segs), default=0.0)
        reports[spk] = detect_language(
            full_text=full_text,
            fasttext_lid=fasttext_lid,
            roman_indic_classifier=roman_indic_classifier,
            transcript_segments=segs,
            total_duration_s=total_dur,
        )

    return reports
