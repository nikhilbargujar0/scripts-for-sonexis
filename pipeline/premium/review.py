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
) -> Dict[str, Any]:
    reasons: List[str] = []
    score = 0.0
    if audio_condition == "outdoor":
        reasons.append("outdoor_audio")
        score += 0.20
    if consensus is not None and float(consensus.consensus_score) < 0.75:
        reasons.append("low_consensus_score")
        score += 0.25
    if alignment is not None and float(alignment.timestamp_confidence) < 0.75:
        reasons.append("low_timestamp_confidence")
        score += 0.20
    if routing is not None and len(routing.engines_used) > 1:
        reasons.append("multi_engine_disagreement")
        score += 0.15
    if code_switch and code_switch.get("detected"):
        reasons.append("code_switch_detected")
        score += 0.20

    review_required = bool(require_human_review or pipeline_mode == "premium_accuracy")
    return {
        "required": review_required,
        "status": "pending",
        "review_stage": "transcript_review",
        "reviewer_id": None,
        "notes": None,
        "priority": _priority_label(score),
        "priority_score": round(score, 4),
        "reasons": reasons,
    }
