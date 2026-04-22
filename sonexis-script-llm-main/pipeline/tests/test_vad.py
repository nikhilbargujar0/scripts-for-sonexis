"""Tests for VAD."""
import numpy as np
import pytest

from pipeline.vad import VADConfig, detect_speech


def test_detect_speech_on_two_speaker_wav(two_speaker_wav, sr):
    segs = detect_speech(two_speaker_wav, sr)
    assert isinstance(segs, list)
    assert len(segs) >= 1
    for s, e in segs:
        assert 0.0 <= s < e <= len(two_speaker_wav) / sr + 0.2


def test_vad_returns_no_segments_on_silence(silent_wav, sr):
    segs = detect_speech(silent_wav, sr)
    assert segs == []


def test_merge_and_pad_cover_speech(two_speaker_wav, sr):
    segs = detect_speech(two_speaker_wav, sr,
                         cfg=VADConfig(min_silence_ms=500, pad_ms=50))
    total = sum(e - s for s, e in segs)
    # Bulk of the 11s clip is speech; we should detect a majority of it.
    assert total > 3.0


def test_invalid_sample_rate_raises(two_speaker_wav):
    with pytest.raises(ValueError):
        detect_speech(two_speaker_wav, sample_rate=22050)


def test_segments_are_monotonically_ordered(two_speaker_wav, sr):
    segs = detect_speech(two_speaker_wav, sr)
    for (_s1, e1), (s2, _e2) in zip(segs, segs[1:]):
        assert e1 <= s2 + 1e-6
