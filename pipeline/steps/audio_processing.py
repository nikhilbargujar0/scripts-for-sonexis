"""Audio loading and deterministic mixing step."""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from ..audio_loader import (
    TARGET_SR,
    LoadedAudio,
    SpeakerPairAudio,
    detect_and_group_pairs,
    detect_stereo_files,
    iter_audio_files,
    load_audio,
    load_speaker_pair,
    load_stereo_as_pair,
)


def normalise_waveform(wav: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    y = np.asarray(wav, dtype=np.float32).copy()
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        y *= float(target_peak / peak)
    return y


def mix_mono(waveforms: Iterable[np.ndarray]) -> np.ndarray:
    """Mix speaker waveforms without clipping and without moving overlaps."""
    waves: List[np.ndarray] = [normalise_waveform(w) for w in waveforms if np.asarray(w).size]
    if not waves:
        return np.zeros(0, dtype=np.float32)
    n = max(len(w) for w in waves)
    out = np.zeros(n, dtype=np.float64)
    rms_values = [float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) for w in waves]
    target_rms = max(min(float(np.median([r for r in rms_values if r > 1e-6] or [0.1])), 0.35), 0.01)
    for wav, rms in zip(waves, rms_values):
        gain = min(target_rms / rms, 2.0) if rms > 1e-6 else 1.0
        out[:len(wav)] += wav.astype(np.float64) * gain
    out /= max(len(waves), 1)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:
        out *= 0.95 / peak
    return out.astype(np.float32)


__all__ = [
    "TARGET_SR",
    "LoadedAudio",
    "SpeakerPairAudio",
    "detect_and_group_pairs",
    "detect_stereo_files",
    "iter_audio_files",
    "load_audio",
    "load_speaker_pair",
    "load_stereo_as_pair",
    "mix_mono",
    "normalise_waveform",
]
