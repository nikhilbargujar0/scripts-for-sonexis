"""Diarisation product export."""
from __future__ import annotations

from typing import Dict, List


def _overlap_markers(turns: List[Dict]) -> List[Dict]:
    markers: List[Dict] = []
    sorted_turns = sorted(turns, key=lambda item: (item.get("start", 0.0), item.get("end", 0.0)))
    for first, second in zip(sorted_turns, sorted_turns[1:]):
        overlap = min(float(first.get("end", 0.0)), float(second.get("end", 0.0))) - max(
            float(first.get("start", 0.0)),
            float(second.get("start", 0.0)),
        )
        if overlap > 0:
            markers.append(
                {
                    "start": round(max(float(first.get("start", 0.0)), float(second.get("start", 0.0))), 3),
                    "end": round(min(float(first.get("end", 0.0)), float(second.get("end", 0.0))), 3),
                    "speakers": [first.get("speaker"), second.get("speaker")],
                }
            )
    return markers


def build_diarisation_product(record: Dict) -> Dict:
    turns = list(record.get("speaker_segmentation", []) or [])
    return {
        "session_name": record.get("session_name"),
        "speaker_segmentation": turns,
        "overlap_markers": _overlap_markers(turns),
        "timestamp_method": record.get("timestamp_method"),
        "timestamp_confidence": record.get("timestamp_confidence"),
        "input_alignment": record.get("input_alignment", {}),
        "premium_processing": record.get("premium_processing", {}),
    }
