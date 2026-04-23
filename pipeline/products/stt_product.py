"""STT product export."""
from __future__ import annotations

from typing import Dict


def build_stt_product(record: Dict) -> Dict:
    metadata = record.get("metadata", {})
    return {
        "session_name": record.get("session_name"),
        "input_mode": record.get("input_mode"),
        "transcript": record.get("transcript"),
        "conversation_transcript": record.get("conversation_transcript", []),
        "speaker_segmentation": record.get("speaker_segmentation", []),
        "language_segments": (metadata.get("language") or {}).get("segments", []),
        "code_switch": record.get("code_switch", {}),
        "human_review": record.get("human_review", {}),
        "timestamp_method": record.get("timestamp_method"),
        "timestamp_confidence": record.get("timestamp_confidence"),
    }
