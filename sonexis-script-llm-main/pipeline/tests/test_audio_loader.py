"""Tests for audio_loader."""
import os
import wave

import numpy as np
import pytest
import soundfile as sf

from pipeline.audio_loader import (
    LoadedAudio,
    detect_stereo_files,
    iter_audio_files,
    load_audio,
    load_batch,
    load_stereo_as_pair,
)


def test_load_wav(tmp_path, two_speaker_wav, sr):
    path = tmp_path / "sample.wav"
    sf.write(str(path), two_speaker_wav, sr)
    clip = load_audio(str(path))
    assert isinstance(clip, LoadedAudio)
    assert clip.sample_rate == 16_000
    assert clip.waveform.dtype == np.float32
    assert clip.waveform.ndim == 1
    # Loader resamples to 16 kHz; length should roughly match duration.
    assert abs(clip.duration - len(two_speaker_wav) / sr) < 0.05


def test_missing_file_returns_none(tmp_path):
    assert load_audio(str(tmp_path / "no_such_file.wav")) is None


def test_unsupported_extension_is_skipped(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    assert load_audio(str(p)) is None


def test_iter_audio_files_filters(tmp_path, two_speaker_wav, sr):
    sf.write(str(tmp_path / "a.wav"), two_speaker_wav, sr)
    (tmp_path / "b.txt").write_text("nope")
    (tmp_path / "sub").mkdir()
    sf.write(str(tmp_path / "sub" / "c.wav"), two_speaker_wav, sr)
    found = sorted(os.path.basename(p) for p in iter_audio_files(str(tmp_path)))
    assert found == ["a.wav", "c.wav"]


def test_load_batch(tmp_path, two_speaker_wav, sr):
    sf.write(str(tmp_path / "a.wav"), two_speaker_wav, sr)
    sf.write(str(tmp_path / "b.wav"), two_speaker_wav, sr)
    batch = load_batch(str(tmp_path))
    assert len(batch) == 2
    assert all(isinstance(c, LoadedAudio) for c in batch)


def test_detect_stereo_files_and_split(tmp_path, two_speaker_wav, sr):
    stereo = np.stack([two_speaker_wav, two_speaker_wav * 0.5], axis=1)
    path = tmp_path / "conversation_0001"
    path.mkdir()
    stereo_path = path / "stereo.wav"
    sf.write(str(stereo_path), stereo, sr)

    found = detect_stereo_files(str(tmp_path))
    assert found == [str(stereo_path)]

    pair = load_stereo_as_pair(str(stereo_path), session_name="conversation_0001")
    assert pair is not None
    assert pair.recording_type == "stereo"
    assert set(pair.speakers) == {"Speaker_L", "Speaker_R"}
