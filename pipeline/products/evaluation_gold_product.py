"""Evaluation and gold-review export."""
from __future__ import annotations

from typing import Dict


def build_evaluation_gold_product(record: Dict) -> Dict:
    return {
        "session_name": record.get("session_name"),
        "selected_transcript": record.get("transcript", {}),
        "transcript_candidates": record.get("transcript_candidates", []),
        "consensus": record.get("consensus", {}),
        "routing_decision": record.get("routing_decision", {}),
        "timestamp_refinement": record.get("timestamp_refinement", {}),
        "human_review": record.get("human_review", {}),
        "quality_targets": record.get("quality_targets", {}),
        "quality_metrics": record.get("quality_metrics", {}),
        "validation": record.get("validation", {}),
        "premium_processing": record.get("premium_processing", {}),
    }
