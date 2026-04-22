"""Tests for transcription helpers. The real ASR model is NOT loaded here
(that would download weights and exceed test budget); we only exercise
the pure functions and dataclasses."""
import math

import numpy as np
import pytest

from pipeline.transcription import (
    ASRConfig,
    FILLERS,
    Transcript,
    TranscriptSegment,
    Word,
    _segment_rms_db,
    compute_quality_score,
    normalise_transcript,
)


def test_normalise_keeps_fillers():
    raw = "um  haan   toh matlab   yaar  ,   kya   hua  ?"
    normalised = normalise_transcript(raw)
    for marker in ("um", "haan", "toh", "matlab", "yaar"):
        assert marker in normalised
    # Should have collapsed redundant whitespace.
    assert "  " not in normalised


def test_fillers_cover_bilingual_markers():
    for marker in ("um", "uh", "haan", "matlab", "yaani"):
        assert marker in FILLERS


def test_asr_config_defaults_are_cpu_friendly():
    cfg = ASRConfig()
    assert cfg.device == "cpu"
    assert cfg.compute_type == "int8"
    assert cfg.model_size == "small"
    assert cfg.temperature == (0.0, 0.4, 0.8)
    assert cfg.show_progress is True


def test_segment_rms_db_is_finite(two_speaker_wav, sr):
    db = _segment_rms_db(two_speaker_wav, sr, 0.0, 1.0)
    assert math.isfinite(db)
    assert -60.0 < db < 0.0

    # Empty slice collapses to silent floor.
    assert _segment_rms_db(two_speaker_wav, sr, 1.0, 1.0) == -120.0


def test_quality_score_bounds():
    # Ideal segment: high logprob, low compression, no-speech ~0, healthy RMS.
    ideal = compute_quality_score(
        avg_logprob=-0.1, compression_ratio=1.5,
        no_speech_prob=0.05, rms_db=-18.0,
    )
    assert 0.85 <= ideal <= 1.0

    # Pathological segment: all axes bad.
    bad = compute_quality_score(
        avg_logprob=-4.0, compression_ratio=3.5,
        no_speech_prob=0.9, rms_db=-55.0,
    )
    assert 0.0 <= bad <= 0.05


def test_quality_score_monotone_in_logprob():
    low = compute_quality_score(avg_logprob=-2.0, compression_ratio=1.5,
                                no_speech_prob=0.1, rms_db=-20.0)
    high = compute_quality_score(avg_logprob=-0.2, compression_ratio=1.5,
                                 no_speech_prob=0.1, rms_db=-20.0)
    assert high > low


def test_transcript_dataclass_roundtrip():
    seg = TranscriptSegment(
        start=0.0, end=1.0, text="hi", language="en",
        avg_logprob=-0.5, compression_ratio=1.4, no_speech_prob=0.1,
        rms_db=-22.0, quality_score=0.75,
        words=[Word(text="hi", start=0.0, end=0.5, probability=0.9)],
    )
    t = Transcript(language="en", language_probability=0.9, duration=1.0,
                   segments=[seg])
    d = t.to_dict()
    assert d["language"] == "en"
    assert d["segments"][0]["quality_score"] == 0.75
    assert d["segments"][0]["words"][0]["text"] == "hi"
    assert t.text.strip() == "hi"
