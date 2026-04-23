"""Conservative TTS export subset builder."""
from __future__ import annotations

from typing import Dict, List


def _eligible_segments(record: Dict) -> List[Dict]:
    suitability = record.get("tts_suitability", {})
    review = record.get("human_review", {})
    if not suitability.get("eligible"):
        return []
    if review.get("status") not in {"approved", "corrected"}:
        return []
    return list(record.get("conversation_transcript", []) or [])


def build_tts_export_product(record: Dict) -> Dict:
    return {
        "session_name": record.get("session_name"),
        "tts_suitability": record.get("tts_suitability", {}),
        "human_review": record.get("human_review", {}),
        "eligible_segments": _eligible_segments(record),
    }
