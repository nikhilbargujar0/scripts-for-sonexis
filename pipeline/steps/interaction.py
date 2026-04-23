"""Conversation interaction metrics."""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..diarisation import SpeakerTurn
from ..interaction_metadata import OverlapSegment, extract_interaction_metadata


def compute_interaction(turns: List[SpeakerTurn], interruption_threshold_s: float = 0.5) -> Tuple[Dict, Dict[str, float], List[OverlapSegment]]:
    meta, overlap_ratios, overlaps = extract_interaction_metadata(turns, interruption_threshold_s)
    response = meta.get("response_latency") or {}
    public = {
        "turn_count": int(meta.get("total_turns") or 0),
        "interruptions": int(meta.get("interruption_count") or 0),
        "overlap_duration": float(meta.get("overlap_duration_s") or 0.0),
        "avg_response_latency": response.get("mean_s"),
        "dominance": meta.get("dominance") or {},
        "turn_switching_frequency": float(meta.get("turn_switch_frequency") or 0.0),
        "overlaps": [o.to_dict() for o in overlaps],
    }
    enriched = dict(meta)
    enriched.update(public)
    return enriched, overlap_ratios, overlaps
