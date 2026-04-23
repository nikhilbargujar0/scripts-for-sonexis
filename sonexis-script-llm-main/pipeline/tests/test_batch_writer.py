"""Tests for batch_writer (json, jsonl, parquet)."""
import json
import os

import pytest

from pipeline.batch_writer import (
    SUPPORTED_FORMATS,
    BatchWriter,
    _flatten_for_parquet,
)


def _make_record(name="a.wav", text="hello"):
    return {
        "schema_version": "3.0.0",
        "generated_at": "2026-02-14T10:00:00+00:00",
        "file": {
            "name": name, "path": f"/tmp/{name}",
            "size_bytes": 100, "sha1": "deadbeef",
        },
        "metadata": {
            "audio": {"duration_s": 1.0, "sample_rate": 16000, "rms_db": -22.0,
                      "snr_db_estimate": 20.0, "rt60_s_estimate": 0.2,
                      "spectral_centroid_khz": 2.0, "environment": "studio",
                      "device_estimate": "studio_mic"},
            "language": {"primary_language": "en", "confidence": 0.9,
                         "code_switching": False, "scripts": ["Latin"],
                         "method": "heuristic", "per_segment": []},
            "speakers": {"SPEAKER_00": {"wpm": 120.0}},
            "conversation": {
                "turn_count": 1, "speaker_count": 1,
                "avg_turn_length_s": 1.0, "total_speech_time_s": 1.0,
                "topic_keywords": ["hello"], "intents": ["informational"],
                "quality": {
                    "mean_quality_score": 0.8,
                    "median_quality_score": 0.8,
                    "duration_weighted_quality_score": 0.8,
                    "low_quality_segment_ratio": 0.0,
                    "high_quality_segment_ratio": 1.0,
                    "segment_count": 1,
                },
            },
        },
        "transcript": {
            "raw": text, "normalised": text, "language": "en",
            "language_probability": 0.9, "duration_s": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": text,
                          "language": "en", "avg_logprob": -0.2,
                          "compression_ratio": 1.5, "no_speech_prob": 0.1,
                          "rms_db": -22.0, "quality_score": 0.8,
                          "words": []}],
        },
        "speaker_segmentation": [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "duration": 1.0},
        ],
        "monologue_sample": None,
        "processing_time_s": 0.5,
    }


def test_json_writer_emits_per_file(tmp_path):
    w = BatchWriter(str(tmp_path), fmt="json")
    w.write(_make_record("a.wav"))
    w.write(_make_record("b.wav"))
    w.close()
    assert (tmp_path / "a.json").exists()
    assert (tmp_path / "b.json").exists()
    with open(tmp_path / "a.json") as f:
        data = json.load(f)
    assert data["file"]["name"] == "a.wav"


def test_jsonl_writer_emits_single_file(tmp_path):
    w = BatchWriter(str(tmp_path), fmt="jsonl", dataset_name="batch")
    w.write(_make_record("a.wav"))
    w.write(_make_record("b.wav"))
    out = w.close()
    assert os.path.basename(out) == "batch.jsonl"
    with open(out) as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["file"]["name"] == "a.wav"
    assert json.loads(lines[1])["file"]["name"] == "b.wav"


def test_parquet_writer_flattens_and_persists(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    w = BatchWriter(str(tmp_path), fmt="parquet", dataset_name="ds")
    w.write(_make_record("a.wav"))
    w.write(_make_record("b.wav", text="world"))
    out = w.close()
    assert out.endswith("ds.parquet")
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert set(df["file_name"]) == {"a.wav", "b.wav"}
    assert df["duration_s"].iloc[0] == 1.0
    # Nested structures should round-trip as JSON strings.
    segs = json.loads(df["transcript_segments_json"].iloc[0])
    assert segs[0]["text"] == "hello"


def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError):
        BatchWriter(str(tmp_path), fmt="xml")


def test_supported_formats_constant():
    assert set(SUPPORTED_FORMATS) == {"json", "jsonl", "parquet"}


def test_flatten_projects_expected_fields():
    rec = _make_record()
    flat = _flatten_for_parquet(rec)
    for key in ("file_name", "duration_s", "primary_language",
                "mean_quality_score", "transcript_raw",
                "speaker_segmentation_json", "processing_time_s"):
        assert key in flat
