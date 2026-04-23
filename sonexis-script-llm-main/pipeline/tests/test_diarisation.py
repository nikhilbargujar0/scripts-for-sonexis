"""Tests for diarisation (MFCC + KMeans fallback)."""
import numpy as np

from pipeline.diarisation import (
    DiarisationConfig,
    SpeakerTurn,
    count_speakers,
    diarise,
)
from pipeline.vad import detect_speech


def test_diarise_two_speakers(two_speaker_wav, sr):
    speech = detect_speech(two_speaker_wav, sr)
    turns = diarise(two_speaker_wav, sr, speech)
    assert len(turns) >= 2
    # There really are two pitches in the fixture; the clusterer should find them.
    assert count_speakers(turns) >= 2


def test_diarise_returns_non_overlapping_turns(two_speaker_wav, sr):
    speech = detect_speech(two_speaker_wav, sr)
    turns = diarise(two_speaker_wav, sr, speech)
    for a, b in zip(turns, turns[1:]):
        assert a.end <= b.start + 1e-6, f"turns overlap: {a} vs {b}"


def test_diarise_empty_vad_yields_no_turns(two_speaker_wav, sr):
    assert diarise(two_speaker_wav, sr, []) == []


def test_diarise_single_speaker_when_bounds_forced(two_speaker_wav, sr):
    speech = detect_speech(two_speaker_wav, sr)
    turns = diarise(
        two_speaker_wav, sr, speech,
        cfg=DiarisationConfig(min_speakers=1, max_speakers=1),
    )
    assert count_speakers(turns) == 1


def test_turn_durations_are_positive(two_speaker_wav, sr):
    speech = detect_speech(two_speaker_wav, sr)
    turns = diarise(two_speaker_wav, sr, speech)
    for t in turns:
        assert isinstance(t, SpeakerTurn)
        assert t.duration() > 0
