"""Validation report helpers for production dataset runs.

Validation stays deterministic and conservative: it records objective checks
and explicit fallbacks, then leaves borderline interpretation to downstream
curators instead of pretending precision we do not have.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .diarisation import SpeakerTurn
from .quality_checker import QualityReport
from .transcription import Transcript


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    confidence: float = 1.0,
    details: Optional[Dict] = None,
) -> Dict:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "confidence": round(float(confidence), 3),
        "details": details or {},
    }


def _transcript_quality(transcript: Optional[Transcript], quality_score_threshold: float = 0.35) -> Dict:
    if transcript is None:
        return {
            "segment_count": 0,
            "word_count": 0,
            "mean_quality_score": 0.0,
            "low_quality_segment_ratio": 1.0,
            "empty_transcript": True,
        }

    scored = [s for s in transcript.segments if s.text.strip()]
    scores = [float(s.quality_score) for s in scored]
    word_count = sum(len(s.words) if s.words else len(s.text.split()) for s in scored)
    return {
        "segment_count": len(scored),
        "word_count": int(word_count),
        "mean_quality_score": round(float(np.mean(scores)), 4) if scores else 0.0,
        "low_quality_segment_ratio": round(
            float(sum(1 for s in scores if s < quality_score_threshold) / len(scores)), 4
        ) if scores else 1.0,
        "empty_transcript": len(scored) == 0,
    }


def build_validation_report(
    *,
    quality_report: Optional[QualityReport] = None,
    transcript: Optional[Transcript] = None,
    requested_diarisation_backend: Optional[str] = None,
    effective_diarisation_backend: Optional[str] = None,
    speech_segments: Optional[List[tuple[float, float]]] = None,
    turns: Optional[List[SpeakerTurn]] = None,
    input_alignment: Optional[Dict] = None,
    mono_mix: Optional[Dict] = None,
    interaction_meta: Optional[Dict] = None,
    expected_overlap_duration_s: float = 0.0,
    session_duration_s: float = 0.0,
    alignment_required: bool = False,
    quality_score_threshold: float = 0.35,
) -> Dict:
    """Build a JSON-safe validation block for one session."""
    issues: List[Dict] = []
    checks: Dict = {}

    if quality_report is not None:
        q = quality_report.to_dict()
        checks["audio_quality"] = q
        for msg in quality_report.errors:
            issues.append(_issue("error", "audio_quality_error", msg))
        for msg in quality_report.warnings:
            issues.append(_issue("warning", "audio_quality_warning", msg))

    tq = _transcript_quality(transcript, quality_score_threshold)
    checks["transcript_quality"] = tq
    if tq["empty_transcript"]:
        issues.append(_issue(
            "error", "empty_transcript",
            "ASR produced no non-empty transcript segments.",
        ))
    elif tq["low_quality_segment_ratio"] > quality_score_threshold:
        pct = int(quality_score_threshold * 100)
        issues.append(_issue(
            "warning", "low_transcript_quality",
            f"More than {pct}% of transcript segments have quality_score < {quality_score_threshold}.",
            confidence=0.8,
            details={"low_quality_segment_ratio": tq["low_quality_segment_ratio"]},
        ))

    checks["speech_detection"] = {
        "speech_segment_count": len(speech_segments or []),
        "turn_count": len(turns or []),
    }
    if not speech_segments:
        issues.append(_issue(
            "warning", "no_speech_segments",
            "VAD returned no speech segments.",
        ))
    if not turns:
        issues.append(_issue(
            "warning", "no_speaker_turns",
            "Diarisation returned no speaker turns.",
        ))
    elif len(speech_segments or []) < 2:
        issues.append(_issue(
            "warning", "too_few_speech_segments",
            "Very few speech segments detected; timing metrics may be unstable.",
            confidence=0.7,
            details={"speech_segment_count": len(speech_segments or [])},
        ))

    checks["clipping_detected"] = bool(
        (quality_report and (quality_report.clipping_ratio or 0.0) >= 0.01)
    )
    if checks["clipping_detected"]:
        issues.append(_issue(
            "warning", "excessive_clipping",
            "Input audio contains clipped samples.",
            details={"clipping_ratio": round(float(quality_report.clipping_ratio or 0.0), 4)},
        ))

    if requested_diarisation_backend or effective_diarisation_backend:
        checks["diarisation"] = {
            "requested_backend": requested_diarisation_backend,
            "effective_backend": effective_diarisation_backend,
        }
        if (
            requested_diarisation_backend
            and effective_diarisation_backend
            and requested_diarisation_backend != effective_diarisation_backend
        ):
            issues.append(_issue(
                "warning", "diarisation_fallback",
                "Requested diarisation backend was not used for this session.",
                details=checks["diarisation"],
            ))

    if input_alignment:
        checks["input_alignment"] = input_alignment
        checks["alignment_applied"] = bool(input_alignment.get("applied"))
        checks["alignment_confidence"] = float(input_alignment.get("confidence", 0.0) or 0.0)
        if alignment_required and not input_alignment.get("applied"):
            issues.append(_issue(
                "warning", "alignment_not_applied",
                "Alignment report exists but waveform shift was not applied.",
                confidence=float(input_alignment.get("confidence", 0.0) or 0.0),
                details={"offset_ms": input_alignment.get("offset_ms")},
            ))
        if input_alignment.get("passed") is False:
            issues.append(_issue(
                "warning", "low_alignment_confidence",
                "Alignment confidence below threshold; interaction metrics may be unreliable.",
                confidence=float(input_alignment.get("confidence", 0.0) or 0.0),
                details={
                    "offset_ms": input_alignment.get("offset_ms"),
                    "method": input_alignment.get("method"),
                },
            ))
        mismatch = float(input_alignment.get("duration_mismatch_s") or 0.0)
        threshold = float(input_alignment.get("max_duration_mismatch_s") or 60.0)
        if mismatch > threshold:
            issues.append(_issue(
                "warning", "speaker_duration_mismatch",
                "Separate speaker files differ substantially in duration.",
                details={"duration_mismatch_s": round(mismatch, 3), "threshold_s": threshold},
            ))

    if mono_mix:
        checks["mono_mix"] = mono_mix
        checks["mono_mix_clipping_detected"] = bool((mono_mix.get("peak_before") or 0.0) > 0.98)
        if checks["mono_mix_clipping_detected"]:
            issues.append(_issue(
                "warning", "mono_mix_clipping",
                "Mono mix needed clipping prevention.",
                details=mono_mix,
            ))

    observed_overlap = float((interaction_meta or {}).get("overlap_duration_s") or (interaction_meta or {}).get("overlap_duration") or 0.0)
    checks["expected_overlap_duration_s"] = round(float(expected_overlap_duration_s), 3)
    checks["observed_overlap_duration_s"] = round(float(observed_overlap), 3)
    if expected_overlap_duration_s > 0.25 and observed_overlap < 0.05:
        issues.append(_issue(
            "warning", "missing_expected_overlap",
            "Speaker VAD suggests overlap, but interaction layer reported almost none.",
            details={
                "expected_overlap_duration_s": round(float(expected_overlap_duration_s), 3),
                "observed_overlap_duration_s": round(float(observed_overlap), 3),
            },
        ))

    checks["suspicious_low_turn_count"] = bool(session_duration_s >= 60.0 and len(turns or []) < 3)
    if checks["suspicious_low_turn_count"]:
        issues.append(_issue(
            "warning", "suspiciously_low_turn_count",
            "Long session has unusually few turns.",
            details={"session_duration_s": round(float(session_duration_s), 3), "turn_count": len(turns or [])},
        ))

    passed = not any(i["severity"] == "error" for i in issues)
    return {
        "passed": passed,
        "issue_count": len(issues),
        "issues": issues,
        "checks": checks,
    }
