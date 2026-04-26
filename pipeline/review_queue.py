"""Review queue manifest helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def build_review_queue(record: Dict) -> List[Dict]:
    session_id = str(record.get("session_id") or record.get("session_name") or "")
    audio_path = str((record.get("file") or {}).get("path") or "")
    rows: List[Dict] = []
    for segment in record.get("transcript", {}).get("segments", []):
        if not segment.get("flagged_for_review"):
            continue
        rows.append({
            "session_id": session_id,
            "segment_id": segment.get("segment_id"),
            "audio_filepath": segment.get("audio_filepath") or audio_path,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "quality_tier": segment.get("quality_tier"),
            "confidence": segment.get("confidence"),
            "review_reasons": list(segment.get("review_reasons") or []),
        })
    return rows


def write_review_queue(record: Dict, output_root: str) -> str | None:
    rows = build_review_queue(record)
    if not rows:
        return None
    path = Path(output_root) / "manifests" / "review_queue.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path.resolve())
