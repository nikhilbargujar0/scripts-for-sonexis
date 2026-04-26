"""Shared exporter helpers."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List


def session_id(record: Dict) -> str:
    return _ascii_id(str(record.get("session_id") or record.get("session_name") or "session"))


def audio_path(record: Dict) -> str:
    file_info = record.get("file") or {}
    return str(file_info.get("path") or "")


def _ascii_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\\-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def segment_rows(record: Dict) -> List[Dict]:
    sid = session_id(record)
    rows: List[Dict] = []
    for idx, segment in enumerate(record.get("transcript", {}).get("segments", []) or []):
        speaker = _ascii_id(str(segment.get("speaker_id") or "SPEAKER_00"))
        utt_id = _ascii_id(f"{speaker}-{sid}-{idx:04d}")
        row = dict(segment)
        row["utt_id"] = utt_id
        row["recording_id"] = sid
        row["speaker_ascii"] = speaker
        row["audio_path"] = str(segment.get("audio_filepath") or audio_path(record))
        rows.append(row)
    return rows


def c_sort(lines: Iterable[str]) -> List[str]:
    return sorted(lines, key=lambda line: line.encode("utf-8"))
