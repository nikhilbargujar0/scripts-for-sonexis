"""Tests for preprocessing."""
import numpy as np

from pipeline.preprocessing import PreprocessConfig, preprocess


def test_preprocess_peak_normalises(two_speaker_wav, sr):
    # Scale the input down so peak normalisation has work to do.
    quiet = (two_speaker_wav * 0.05).astype(np.float32)
    out = preprocess(quiet, sr, PreprocessConfig(peak_normalize=True, peak_target=0.95))
    assert abs(np.max(np.abs(out)) - 0.95) < 1e-3


def test_preprocess_silence_is_stable(silent_wav, sr):
    out = preprocess(silent_wav, sr)
    assert out.shape == silent_wav.shape
    assert np.max(np.abs(out)) < 1e-5


def test_preprocess_dtype(two_speaker_wav, sr):
    out = preprocess(two_speaker_wav, sr)
    assert out.dtype == np.float32


def test_preprocess_dc_removal(sr):
    wav = np.full(sr, 0.3, dtype=np.float32)  # constant -> pure DC
    out = preprocess(wav, sr, PreprocessConfig(peak_normalize=False, dc_remove=True))
    assert abs(float(np.mean(out))) < 1e-6
