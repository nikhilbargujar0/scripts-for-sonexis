"""Premium quality, code-switch, and provenance helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import PipelineConfig
from ..steps.language import build_code_switch_report
from ..utils.premium_routing import extract_noise_level, normalize_recording_condition
from .types import AlignmentResult, ConsensusResult, PremiumProcessingReport, RoutingDecision, TranscriptCandidate


def build_quality_targets(cfg: PipelineConfig) -> Dict[str, Any]:
    word_target = float(
        getattr(cfg, "word_accuracy_target", None)
        or getattr(cfg, "transcript_accuracy_target", 0.98)
        or 0.98
    )
    timestamp_target = float(getattr(cfg, "timestamp_accuracy_target", 0.98) or 0.98)
    code_switch_target = float(
        getattr(cfg, "code_switch_accuracy_target", None)
        or getattr(cfg, "transcript_accuracy_target", 0.98)
        or 0.98
    )
    human_review_required = bool(
        getattr(cfg, "require_human_review", None)
        if getattr(cfg, "require_human_review", None) is not None
        else getattr(cfg, "metadata_review_required", True)
    )
    return {
        "word_accuracy_target": round(word_target, 4),
        "timestamp_accuracy_target": round(timestamp_target, 4),
        "code_switch_accuracy_target": round(code_switch_target, 4),
        "human_review_required": human_review_required,
    }


def build_quality_metrics() -> Dict[str, Any]:
    return {
        "estimated_word_accuracy": None,
        "estimated_timestamp_accuracy": None,
        "estimated_code_switch_accuracy": None,
        "benchmark_evaluated": False,
        "human_review_completed": False,
    }


def build_code_switch_metadata(
    *,
    language_report: Optional[Dict[str, Any]] = None,
    candidate: Optional[TranscriptCandidate] = None,
) -> Dict[str, Any]:
    language_report = language_report or {}
    code_switch = dict((candidate.code_switch_signals if candidate else {}) or {})
    report = build_code_switch_report(
        language_report.get("language_segments")
        or language_report.get("segments")
        or []
    )
    if report.get("segments"):
        code_switch["switch_count"] = int(report.get("switch_count") or 0)
        code_switch["switch_patterns"] = list(report.get("patterns") or [])
        code_switch["dominant_languages"] = [
            segment.get("lang")
            for segment in report.get("segments", [])
            if segment.get("lang")
        ]
    dominant = []
    for language in code_switch.get("dominant_languages") or []:
        if language and language not in dominant:
            dominant.append(language)
    return {
        "detected": bool(code_switch.get("detected") or code_switch.get("switch_count")),
        "dominant_languages": dominant,
        "switch_count": int(code_switch.get("switch_count") or 0),
        "switch_patterns": list(code_switch.get("switch_patterns") or code_switch.get("patterns") or []),
        "review_required": bool(code_switch.get("detected") or code_switch.get("switch_count")),
    }


def build_premium_processing(
    *,
    pipeline_mode: str,
    routing: RoutingDecision,
    consensus: ConsensusResult,
    alignment: AlignmentResult,
    human_review_required: bool,
) -> Dict[str, Any]:
    report = PremiumProcessingReport(
        pipeline_mode=pipeline_mode,
        paid_api_used=any(engine != "whisper_local" for engine in routing.engines_used),
        engines_used=routing.engines_used,
        consensus_applied=len(consensus.engines_compared) > 1 or consensus.transcript_strategy == "merged_consensus",
        timestamp_refinement_applied=alignment.refinement_applied,
        human_review_required=human_review_required,
    )
    return report.to_dict()


def build_tts_suitability(
    *,
    record_or_meta: Dict[str, Any],
    review_status: str,
    overlap_duration_s: float,
    speaker_count: int,
    alignment: Optional[AlignmentResult] = None,
) -> Dict[str, Any]:
    audio = ((record_or_meta or {}).get("metadata") or {}).get("audio") if "metadata" in (record_or_meta or {}) else (record_or_meta or {})
    noise_level = extract_noise_level(audio)
    recording_condition = normalize_recording_condition(audio)
    reasons = []
    eligible = True
    if speaker_count != 1:
        reasons.append("multi_speaker_segment")
        eligible = False
    if float(overlap_duration_s or 0.0) > 0.0:
        reasons.append("overlap_detected")
        eligible = False
    if noise_level != "low":
        reasons.append("noise_not_low")
        eligible = False
    if recording_condition == "outdoor":
        reasons.append("outdoor_condition")
        eligible = False
    if review_status not in {"approved", "corrected"}:
        reasons.append("review_not_final")
        eligible = False
    if alignment is None:
        reasons.append("timing_not_verified")
        eligible = False
    else:
        if bool(alignment.synthetic_word_timestamps):
            reasons.append("synthetic_timestamps")
            eligible = False
        if str(alignment.timing_quality).lower() not in {"high"}:
            reasons.append("timing_not_stable")
            eligible = False
    return {
        "eligible": eligible,
        "reasons": reasons,
        "confidence": 0.92 if eligible else 0.2,
    }
