"""Difficulty-aware routing helpers for premium hybrid ASR."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ..premium.types import RoutingDecision
from ..transcription import Transcript


SUPPORTED_PREMIUM_LANGUAGES = ("Hindi", "English", "Hinglish", "Marwadi", "Punjabi")


def _value_from_field(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("value")
    return value


def normalize_recording_condition(audio_meta: Optional[Dict[str, Any]]) -> str:
    audio_meta = audio_meta or {}
    value = str(
        _value_from_field(audio_meta.get("recording_condition"))
        or _value_from_field(audio_meta.get("environment"))
        or ""
    ).strip().lower()
    if value in {"indoor", "quiet_indoor", "reverberant_indoor", "studio"}:
        return "indoor"
    if value in {"outdoor", "street", "traffic", "field"}:
        return "outdoor"
    return "unknown"


def extract_noise_level(audio_meta: Optional[Dict[str, Any]]) -> str:
    value = str(_value_from_field((audio_meta or {}).get("noise_level")) or "").strip().lower()
    if value in {"low", "moderate", "high", "very_high"}:
        return value
    snr = float((audio_meta or {}).get("snr_db_estimate") or 0.0)
    if snr >= 20.0:
        return "low"
    if snr >= 12.0:
        return "moderate"
    if snr > 0.0:
        return "high"
    return "unknown"


def transcript_quality_signals(transcript: Optional[Transcript]) -> Dict[str, float]:
    if transcript is None or not transcript.segments:
        return {
            "segment_count": 0.0,
            "mean_quality_score": 0.0,
            "low_quality_ratio": 1.0,
            "word_timestamp_ratio": 0.0,
        }
    non_empty = [seg for seg in transcript.segments if str(seg.text or "").strip()]
    if not non_empty:
        return {
            "segment_count": 0.0,
            "mean_quality_score": 0.0,
            "low_quality_ratio": 1.0,
            "word_timestamp_ratio": 0.0,
        }
    scores = [float(seg.quality_score or 0.0) for seg in non_empty]
    words_total = sum(max(len(seg.text.split()), 1) for seg in non_empty)
    words_timed = sum(len(seg.words or []) for seg in non_empty)
    low_quality = sum(1 for score in scores if score < 0.45)
    return {
        "segment_count": float(len(non_empty)),
        "mean_quality_score": sum(scores) / len(scores),
        "low_quality_ratio": low_quality / len(scores),
        "word_timestamp_ratio": words_timed / max(words_total, 1),
    }


def estimate_code_switch_density(
    language_report: Optional[Dict[str, Any]] = None,
    code_switch: Optional[Dict[str, Any]] = None,
) -> float:
    language_report = language_report or {}
    code_switch = code_switch or {}
    switching_score = float(language_report.get("switching_score") or 0.0)
    switching_frequency = float(language_report.get("switching_frequency") or 0.0)
    switch_count = float(code_switch.get("switch_count") or 0.0)
    multilingual_flag = bool(
        language_report.get("multilingual_flag")
        or language_report.get("code_switching")
        or code_switch.get("detected")
    )
    density = switching_score
    density = max(density, min(1.0, switching_frequency / 10.0))
    density = max(density, min(1.0, switch_count / 8.0))
    if multilingual_flag:
        density = max(density, 0.45)
    return min(1.0, density)


def build_routing_context(
    *,
    pipeline_mode: str,
    allow_paid_apis: bool,
    audio_meta: Optional[Dict[str, Any]] = None,
    transcript: Optional[Transcript] = None,
    language_report: Optional[Dict[str, Any]] = None,
    code_switch: Optional[Dict[str, Any]] = None,
    overlap_duration_s: float = 0.0,
    review_priority: Optional[str] = None,
) -> Dict[str, Any]:
    quality = transcript_quality_signals(transcript)
    recording_condition = normalize_recording_condition(audio_meta)
    noise_level = extract_noise_level(audio_meta)
    return {
        "pipeline_mode": pipeline_mode,
        "allow_paid_apis": bool(allow_paid_apis),
        "recording_condition": recording_condition,
        "noise_level": noise_level,
        "overlap_duration_s": float(overlap_duration_s),
        "review_priority": review_priority or "normal",
        "code_switch_density": estimate_code_switch_density(language_report, code_switch),
        **quality,
    }


def difficulty_reasons(context: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if context.get("recording_condition") == "outdoor":
        reasons.append("outdoor_audio")
    if context.get("noise_level") in {"high", "very_high"}:
        reasons.append("high_noise")
    if float(context.get("mean_quality_score") or 0.0) < 0.58:
        reasons.append("low_local_confidence")
    if float(context.get("low_quality_ratio") or 0.0) > 0.30:
        reasons.append("many_low_quality_segments")
    if float(context.get("word_timestamp_ratio") or 0.0) < 0.65:
        reasons.append("unstable_timestamps")
    if float(context.get("code_switch_density") or 0.0) >= 0.45:
        reasons.append("dense_code_switching")
    if float(context.get("overlap_duration_s") or 0.0) > 0.75:
        reasons.append("heavy_overlap")
    if str(context.get("review_priority") or "").lower() in {"high", "urgent"}:
        reasons.append("high_review_priority")
    return reasons


def difficulty_score(context: Dict[str, Any]) -> float:
    score = 0.0
    if context.get("recording_condition") == "outdoor":
        score += 0.20
    noise_level = context.get("noise_level")
    if noise_level == "moderate":
        score += 0.08
    elif noise_level == "high":
        score += 0.18
    elif noise_level == "very_high":
        score += 0.24
    score += max(0.0, 0.60 - float(context.get("mean_quality_score") or 0.0)) * 0.8
    score += min(0.25, float(context.get("low_quality_ratio") or 0.0) * 0.35)
    score += min(0.20, max(0.0, 0.75 - float(context.get("word_timestamp_ratio") or 0.0)) * 0.4)
    score += min(0.20, float(context.get("code_switch_density") or 0.0) * 0.3)
    score += min(0.12, float(context.get("overlap_duration_s") or 0.0) / 10.0)
    if str(context.get("review_priority") or "").lower() in {"high", "urgent"}:
        score += 0.10
    return round(min(1.0, score), 4)


def should_escalate_to_paid_asr(context: Dict[str, Any]) -> bool:
    if str(context.get("pipeline_mode") or "offline_standard") != "premium_accuracy":
        return False
    if not bool(context.get("allow_paid_apis")):
        return False
    return difficulty_score(context) >= 0.35


def build_routing_decision(
    context: Dict[str, Any],
    *,
    attempted_engines: Optional[Iterable[str]] = None,
    skipped_engines: Optional[Iterable[str]] = None,
    engines_used: Optional[Iterable[str]] = None,
) -> RoutingDecision:
    return RoutingDecision(
        pipeline_mode=str(context.get("pipeline_mode") or "offline_standard"),
        paid_api_allowed=bool(context.get("allow_paid_apis")),
        local_first=True,
        difficulty_score=difficulty_score(context),
        should_escalate=should_escalate_to_paid_asr(context),
        escalated_to_paid=any(str(engine) != "whisper_local" for engine in (engines_used or [])),
        reasons=difficulty_reasons(context),
        attempted_engines=list(attempted_engines or []),
        skipped_engines=list(skipped_engines or []),
        engines_used=list(engines_used or []),
    )
