"""Shared helpers for premium ASR adapters."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from ...language_detection import (
    ROMAN_HINDI_MARKERS,
    ROMAN_MARWADI_MARKERS,
    ROMAN_PUNJABI_MARKERS,
    detect_language,
)
from ...steps.language import LANG_NAMES, build_code_switch_report
from ...transcription import Transcript
from ..types import TranscriptCandidate


def env_required(env_name: Optional[str], engine_name: str) -> str:
    if not env_name:
        raise RuntimeError(f"{engine_name} requires an environment variable name in config")
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"{engine_name} requested but missing credentials in env var {env_name}")
    return value


def average_word_confidence(transcript: Transcript) -> Optional[float]:
    probs: List[float] = []
    for segment in transcript.segments:
        for word in segment.words or []:
            probs.append(float(word.probability or 0.0))
    if probs:
        return sum(probs) / len(probs)
    scores = [float(segment.quality_score or 0.0) for segment in transcript.segments if segment.text.strip()]
    if scores:
        return sum(scores) / len(scores)
    return None


def transcript_confidence(transcript: Transcript) -> Optional[float]:
    scores = [float(segment.quality_score or 0.0) for segment in transcript.segments if segment.text.strip()]
    if scores:
        return sum(scores) / len(scores)
    return average_word_confidence(transcript)


def timestamp_confidence(transcript: Transcript) -> float:
    segments = [segment for segment in transcript.segments if segment.text.strip()]
    if not segments:
        return 0.0
    with_words = sum(1 for segment in segments if segment.words)
    return round(with_words / len(segments), 4)


def _romanized_indic_present(text: str) -> bool:
    tokens = {token.lower() for token in str(text or "").split()}
    markers = ROMAN_HINDI_MARKERS | ROMAN_PUNJABI_MARKERS | ROMAN_MARWADI_MARKERS
    return bool(tokens.intersection(markers))


_ANY_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_ANY_NUMERAL = re.compile(r"[0-9\u0966-\u096F\u0A66-\u0A6F]")


def infer_normalisation_notes(
    transcript: Transcript,
    *,
    extra_notes: Optional[List[str]] = None,
) -> List[str]:
    text = str(transcript.text or "")
    notes: List[str] = list(extra_notes or [])
    if _ANY_PUNCTUATION.search(text):
        notes.append("punctuation_normalisation_applied_for_comparison")
    if _ANY_NUMERAL.search(text):
        notes.append("numeral_normalisation_applied_for_comparison")
    if _romanized_indic_present(text):
        notes.append("romanized_indic_tokens_present")
    if any("-" in (word.text or "") for segment in transcript.segments for word in (segment.words or [])):
        notes.append("provider_token_split_merge_variation_detected")
    seen = set()
    ordered: List[str] = []
    for note in notes:
        if note and note not in seen:
            seen.add(note)
            ordered.append(note)
    return ordered


def derive_code_switch_signals(
    transcript: Transcript,
    *,
    fasttext_lid=None,
    roman_indic_classifier=None,
) -> Dict[str, Any]:
    lang_report = detect_language(
        full_text=transcript.text,
        fasttext_lid=fasttext_lid,
        roman_indic_classifier=roman_indic_classifier,
        transcript_segments=transcript.segments,
        total_duration_s=transcript.duration,
    )
    report = build_code_switch_report(lang_report.to_dict().get("language_segments", []))
    labels = []
    for segment in report.get("segments", []):
        lang = segment.get("lang")
        if lang and lang not in labels:
            labels.append(lang)
    return {
        "detected": bool(lang_report.code_switching or report.get("switch_count")),
        "dominant_languages": labels or [LANG_NAMES.get(lang_report.primary_language, lang_report.primary_language)],
        "switch_count": int(report.get("switch_count") or 0),
        "switch_patterns": list(report.get("patterns") or []),
        "romanized_indic_present": _romanized_indic_present(transcript.text),
        "switching_score": round(float(lang_report.switching_score), 4),
    }


def infer_detected_languages(code_switch_signals: Dict[str, Any], transcript: Transcript) -> List[str]:
    languages = list(code_switch_signals.get("dominant_languages") or [])
    if not languages and transcript.language:
        languages.append(LANG_NAMES.get(transcript.language, transcript.language))
    return languages


def build_candidate(
    *,
    engine: str,
    transcript: Transcript,
    provider: Optional[str] = None,
    paid_api: bool = False,
    language_hint: Optional[str] = None,
    timing_source: str,
    warnings: Optional[List[str]] = None,
    adapter_metadata: Optional[Dict[str, Any]] = None,
    fasttext_lid=None,
    roman_indic_classifier=None,
    confidence: Optional[float] = None,
    avg_word_probability: Optional[float] = None,
    timestamp_score: Optional[float] = None,
    normalisation_notes: Optional[List[str]] = None,
) -> TranscriptCandidate:
    code_switch_signals = derive_code_switch_signals(
        transcript,
        fasttext_lid=fasttext_lid,
        roman_indic_classifier=roman_indic_classifier,
    )
    return TranscriptCandidate(
        engine=engine,
        provider=provider or engine,
        paid_api=paid_api,
        transcript=transcript,
        confidence=confidence if confidence is not None else transcript_confidence(transcript),
        avg_word_confidence=(
            avg_word_probability
            if avg_word_probability is not None
            else average_word_confidence(transcript)
        ),
        language_hint=language_hint or transcript.language,
        detected_languages=infer_detected_languages(code_switch_signals, transcript),
        code_switch_signals=code_switch_signals,
        timing_source=timing_source,
        timestamp_confidence=(
            timestamp_score
            if timestamp_score is not None
            else timestamp_confidence(transcript)
        ),
        warnings=list(warnings or []),
        adapter_metadata=dict(adapter_metadata or {}),
        normalisation_notes=infer_normalisation_notes(
            transcript,
            extra_notes=normalisation_notes,
        ),
    )
