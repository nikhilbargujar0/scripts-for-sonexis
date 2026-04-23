"""batch_writer.py

Multi-format persistence for pipeline records.

Supports:
- ``json``    (default) - one ``<file>.json`` per input audio. Full nested
  record including word timestamps.
- ``jsonl``   - a single ``dataset.jsonl`` with one record per line. Full
  nested structure preserved.
- ``parquet`` - a flattened, columnar ``dataset.parquet`` suitable for
  dumping into DuckDB / pandas / Spark. Nested fields that don't fit a
  tabular schema (full word timestamps) are serialised as JSON strings.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


SUPPORTED_FORMATS = ("json", "jsonl", "parquet")


def _nested_value(obj: Dict, key: str):
    """Extract .value from a {"value": ..., "confidence": ...} field, or return raw."""
    v = obj.get(key)
    if isinstance(v, dict):
        return v.get("value")
    return v


def _nested_conf(obj: Dict, key: str) -> Optional[float]:
    """Extract .confidence from a {"value": ..., "confidence": ...} field."""
    v = obj.get(key)
    if isinstance(v, dict):
        return v.get("confidence")
    return None


def _flatten_for_parquet(record: Dict) -> Dict:
    """Project a nested pipeline record down to a flat, parquet-friendly row."""
    file = record.get("file", {})
    meta = record.get("metadata", {})
    audio = meta.get("audio", {})
    lang = meta.get("language", {})
    conv = meta.get("conversation", {})
    quality = (conv or {}).get("quality", {}) or {}
    transcript = record.get("transcript", {})
    monologue = record.get("monologue_sample") or {}
    validation = record.get("validation", {})

    return {
        "schema_version": record.get("schema_version"),
        "generated_at": record.get("generated_at"),
        "file_name": file.get("name"),
        "file_path": file.get("path"),
        "file_size_bytes": file.get("size_bytes"),
        "file_sha1": file.get("sha1"),
        "duration_s": audio.get("duration_s"),
        "sample_rate": audio.get("sample_rate"),
        "rms_db": audio.get("rms_db"),
        "snr_db_estimate": audio.get("snr_db_estimate"),
        "rt60_s_estimate": audio.get("rt60_s_estimate"),
        "spectral_centroid_khz": audio.get("spectral_centroid_khz"),
        # environment and device_estimate are now {"value":..., "confidence":...}
        "environment": _nested_value(audio, "environment"),
        "environment_confidence": _nested_conf(audio, "environment"),
        "device_estimate": _nested_value(audio, "device_estimate"),
        "device_estimate_confidence": _nested_conf(audio, "device_estimate"),
        "noise_level": _nested_value(audio, "noise_level"),
        "noise_level_confidence": _nested_conf(audio, "noise_level"),
        "primary_language": lang.get("primary_language"),
        "dominant_language": lang.get("dominant_language"),
        "language_confidence": lang.get("confidence"),
        "code_switching": lang.get("code_switching"),
        "multilingual_flag": lang.get("multilingual_flag"),
        "switching_frequency": lang.get("switching_frequency"),
        "switching_score": lang.get("switching_score"),
        "scripts": json.dumps(lang.get("scripts", []), ensure_ascii=False),
        "language_method": lang.get("method"),
        "language_segments_json": json.dumps(
            lang.get("language_segments", []), ensure_ascii=False
        ),
        "turn_count": conv.get("turn_count"),
        "speaker_count": conv.get("speaker_count"),
        "avg_turn_length_s": conv.get("avg_turn_length_s"),
        "total_speech_time_s": conv.get("total_speech_time_s"),
        "topic_keywords": json.dumps(
            conv.get("topic_keywords", []), ensure_ascii=False
        ),
        "intents": json.dumps(conv.get("intents", []), ensure_ascii=False),
        "mean_quality_score": quality.get("mean_quality_score"),
        "duration_weighted_quality_score":
            quality.get("duration_weighted_quality_score"),
        "low_quality_segment_ratio": quality.get("low_quality_segment_ratio"),
        "high_quality_segment_ratio": quality.get("high_quality_segment_ratio"),
        "segment_count": quality.get("segment_count"),
        "transcript_raw": transcript.get("raw"),
        "transcript_normalised": transcript.get("normalised"),
        "transcript_segments_json": json.dumps(
            transcript.get("segments", []), ensure_ascii=False
        ),
        "speaker_segmentation_json": json.dumps(
            record.get("speaker_segmentation", []), ensure_ascii=False
        ),
        "speakers_json": json.dumps(
            meta.get("speakers", {}), ensure_ascii=False
        ),
        "monologue_speaker": monologue.get("speaker"),
        "monologue_start": monologue.get("start"),
        "monologue_end": monologue.get("end"),
        "monologue_duration_s": monologue.get("duration_s"),
        "monologue_in_range": monologue.get("in_range"),
        "monologue_quality_score": monologue.get("quality_score"),
        "monologue_silence_ratio": monologue.get("silence_ratio"),
        "monologue_interruption_free": monologue.get("interruption_free"),
        "monologue_transcript": monologue.get("transcript"),
        "validation_passed": validation.get("passed"),
        "validation_issue_count": validation.get("issue_count"),
        "validation_issues_json": json.dumps(
            validation.get("issues", []), ensure_ascii=False
        ),
        "source_files_json": json.dumps(
            record.get("source_files", []), ensure_ascii=False
        ),
        "artifacts_json": json.dumps(record.get("artifacts", {}), ensure_ascii=False),
        "processing_time_s": record.get("processing_time_s"),
    }


def write_json(record: Dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(record["file"]["name"])[0]
    path = os.path.join(output_dir, stem + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def append_jsonl(record: Dict, jsonl_path: str) -> str:
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonl_path


def write_parquet(records: List[Dict], parquet_path: str) -> str:
    """Write a list of records to a single parquet file (requires pandas+pyarrow)."""
    try:
        import pandas as pd
    except ImportError as err:  # pragma: no cover
        raise RuntimeError(
            "pandas is required for parquet output. pip install pandas pyarrow"
        ) from err
    try:
        import pyarrow  # noqa: F401
    except ImportError as err:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required for parquet output. pip install pyarrow"
        ) from err

    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    rows = [_flatten_for_parquet(r) for r in records]
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False, compression="snappy")
    return parquet_path


class BatchWriter:
    """Incremental writer that supports all three output formats."""

    def __init__(self, output_dir: str, fmt: str = "json",
                 dataset_name: str = "dataset") -> None:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(
                f"unsupported output format: {fmt!r} (expected {SUPPORTED_FORMATS})"
            )
        self.fmt = fmt
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        os.makedirs(output_dir, exist_ok=True)

        self._jsonl_path: Optional[str] = None
        self._buffer: List[Dict] = []

        if fmt == "jsonl":
            self._jsonl_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
            # Truncate any stale file at start so reruns don't double-write.
            open(self._jsonl_path, "w").close()
        elif fmt == "parquet":
            # Parquet is written in a single shot at close() so we buffer here.
            pass

    def write(self, record: Dict) -> str:
        if self.fmt == "json":
            return write_json(record, self.output_dir)
        if self.fmt == "jsonl":
            assert self._jsonl_path is not None
            return append_jsonl(record, self._jsonl_path)
        # parquet
        self._buffer.append(record)
        return ""  # deferred

    def close(self) -> Optional[str]:
        if self.fmt == "parquet" and self._buffer:
            path = os.path.join(self.output_dir, f"{self.dataset_name}.parquet")
            return write_parquet(self._buffer, path)
        if self.fmt == "jsonl":
            return self._jsonl_path
        return None
