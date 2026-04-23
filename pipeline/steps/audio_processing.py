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
    mixed, _meta = mix_mono_with_metadata(waveforms)
    return mixed


def mix_mono_with_metadata(waveforms: Iterable[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Mix speaker waveforms and return machine-readable clipping metadata."""
    waves: List[np.ndarray] = [normalise_waveform(w) for w in waveforms if np.asarray(w).size]
    if not waves:
        return np.zeros(0, dtype=np.float32), {
            "normalised": True,
            "peak_before": 0.0,
            "peak_after": 0.0,
            "clipping_prevented": False,
        }
    n = max(len(w) for w in waves)
    out = np.zeros(n, dtype=np.float64)
    rms_values = [float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) for w in waves]
    target_rms = max(min(float(np.median([r for r in rms_values if r > 1e-6] or [0.1])), 0.35), 0.01)
    for wav, rms in zip(waves, rms_values):
        gain = min(target_rms / rms, 2.0) if rms > 1e-6 else 1.0
        out[:len(wav)] += wav.astype(np.float64) * gain
    out /= max(len(waves), 1)
    peak_before = float(np.max(np.abs(out))) if out.size else 0.0
    if peak_before > 0.98:
        out *= 0.95 / peak_before
    peak_after = float(np.max(np.abs(out))) if out.size else 0.0
    return out.astype(np.float32), {
        "normalised": True,
        "peak_before": round(peak_before, 4),
        "peak_after": round(peak_after, 4),
        "clipping_prevented": bool(peak_before > 0.98 and peak_after < peak_before),
    }


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
    "mix_mono_with_metadata",
    "normalise_waveform",
]
