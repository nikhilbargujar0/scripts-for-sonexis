"""HuggingFace-style local dataset export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .common import segment_rows, session_id


def _row(row: Dict) -> Dict:
    return {
        "audio": {"path": row.get("audio_path")},
        "session_id": row.get("recording_id"),
        "utterance_id": row.get("utt_id"),
        "speaker": row.get("speaker_id"),
        "language": row.get("language"),
        "transcription": row.get("text"),
        "transcription_normalized": row.get("text_normalized"),
        "start": row.get("start"),
        "end": row.get("end"),
        "duration": row.get("duration"),
        "words": row.get("words") or [],
        "snr_db": row.get("snr_db"),
        "snr_band": row.get("snr_band"),
        "quality_tier": row.get("quality_tier"),
        "confidence_score": row.get("confidence"),
        "confidence_band": row.get("confidence_band"),
        "overlap": row.get("overlap"),
        "code_switch_density": row.get("cs_density"),
        "missing_fields": row.get("missing_fields") or [],
    }


def _dataset_card(record: Dict, rows: List[Dict]) -> str:
    langs = sorted({str(row.get("language") or "und") for row in rows})
    tiers = {}
    for row in rows:
        tier = str(row.get("quality_tier") or "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1
    return (
        "---\n"
        "pretty_name: Sonexis Indic Conversations\n"
        "task_categories:\n"
        "  - automatic-speech-recognition\n"
        "  - speaker-diarization\n"
        f"language: {json.dumps(langs, ensure_ascii=False)}\n"
        "tags:\n"
        "  - code-switching\n"
        "  - indic\n"
        "---\n\n"
        f"# {record.get('session_id') or record.get('session_name')}\n\n"
        f"Rows: {len(rows)}\n\n"
        f"Quality tiers: `{json.dumps(tiers, sort_keys=True)}`\n"
    )


def export_hf_dataset(record: Dict, output_root: str) -> Dict[str, str]:
    sid = session_id(record)
    rows = [_row(row) for row in segment_rows(record)]
    base = Path(output_root) / "exports" / "hf_dataset" / sid
    base.mkdir(parents=True, exist_ok=True)
    jsonl = base / "data.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    features = {
        "audio": "Audio(sampling_rate=16000, decode=False)",
        "session_id": "string",
        "utterance_id": "string",
        "speaker": "string",
        "language": "string",
        "transcription": "string",
        "words": "sequence",
        "quality_tier": "string",
    }
    features_path = base / "features.json"
    features_path.write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
    card = base / "README.md"
    card.write_text(_dataset_card(record, rows), encoding="utf-8")
    return {
        "hf_dataset_jsonl": str(jsonl.resolve()),
        "hf_dataset_features": str(features_path.resolve()),
        "hf_dataset_card": str(card.resolve()),
    }
