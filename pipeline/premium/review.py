"""Human review metadata helpers for premium records."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .types import AlignmentResult, ConsensusResult, RoutingDecision


def _priority_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "normal"


def build_human_review(
    *,
    pipeline_mode: str,
    require_human_review: bool,
    consensus: Optional[ConsensusResult] = None,
    alignment: Optional[AlignmentResult] = None,
    routing: Optional[RoutingDecision] = None,
    audio_condition: str = "unknown",
    code_switch: Optional[Dict[str, Any]] = None,
    accuracy_gate: Optional[Dict[str, Any]] = None,
    targets: Optional[Dict[str, Any]] = None,
    speaker_attribution_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    reasons: List[str] = []
    score = 0.0
    targets = targets or {}
    if audio_condition == "outdoor":
        reasons.append("outdoor_audio")
        score += 0.20
    if consensus is not None and float(consensus.consensus_score) < 0.95:
        reasons.append("low_consensus_score")
        score += 0.80 if float(consensus.consensus_score) < 0.90 else 0.45
    timestamp_target = float(targets.get("timestamp_accuracy_target") or 0.98)
    if alignment is not None and float(alignment.timestamp_confidence) < timestamp_target:
        reasons.append("low_timestamp_confidence")
        score += 0.20
    if alignment is not None and bool(alignment.synthetic_word_timestamps):
        reasons.append("synthetic_word_timestamps")
        score += 0.25
    if alignment is not None and str(alignment.timing_quality).lower() == "low":
        reasons.append("low_timing_quality")
        score += 0.15
    if routing is not None and len(routing.engines_used) > 1:
        reasons.append("multi_engine_disagreement")
        score += 0.15
    code_switch_target = float(targets.get("code_switch_accuracy_target") or 0.99)
    code_switch_confidence = float((code_switch or {}).get("confidence") or 0.0)
    if code_switch and code_switch.get("detected"):
        reasons.append("code_switch_detected")
        score += 0.30
        if code_switch_confidence < code_switch_target:
            reasons.append("low_code_switch_confidence")
            score += 0.20
    speaker_target = float(targets.get("speaker_accuracy_target") or 0.99)
    if speaker_attribution_confidence is not None and float(speaker_attribution_confidence) < speaker_target:
        reasons.append("low_speaker_attribution_confidence")
        score += 0.20
    if accuracy_gate:
        if float(accuracy_gate.get("estimated_word_accuracy") or 0.0) < float(accuracy_gate.get("target_word_accuracy") or 0.99):
            reasons.append("estimated_word_accuracy_below_target")
            score += 0.80 if float(accuracy_gate.get("estimated_word_accuracy") or 0.0) < 0.97 else 0.35
        if bool(accuracy_gate.get("human_review_required")):
            reasons.extend(str(r) for r in accuracy_gate.get("reasons") or [])
    if consensus is not None:
        for flag in consensus.disagreement_flags:
            if flag == "strong_code_switch_disagreement":
                reasons.append("code_switch_region_disagreement")
                score += 0.18
            elif flag == "strong_timestamp_disagreement":
                reasons.append("timestamp_region_disagreement")
                score += 0.16
            elif flag == "strong_transcript_disagreement":
                reasons.append("transcript_disagreement")
                score += 0.12

    reasons = list(dict.fromkeys(reasons))
    review_required = bool(
        require_human_review
        or pipeline_mode == "premium_accuracy"
        or (accuracy_gate and accuracy_gate.get("human_review_required"))
    )
    return {
        "required": review_required,
        "status": "pending",
        "review_stage": "transcript_review",
        "priority": {
            "level": _priority_label(score),
            "score": round(score, 4),
            "reasons": reasons,
        },
        "reviewer_id": None,
        "notes": None,
    }
