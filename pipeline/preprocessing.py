"""preprocessing.py

Light-touch conditioning of raw waveforms before downstream modules.

Principles:
- Do NOT aggressively denoise. Real conversational data is noisy and our
  dataset must preserve those acoustic cues.
- Only apply peak normalisation to bring dynamic range into a sane region
  for VAD / ASR thresholds.
- Optional mild spectral-gate noise reduction is exposed but disabled by
  default. Callers can opt in for particularly harsh recordings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    peak_normalize: bool = True
    peak_target: float = 0.95  # leave some headroom
    dc_remove: bool = True
    denoise: bool = False  # off by default; real speech > clean speech
    denoise_prop: float = 0.5  # mild when enabled


def _remove_dc(wav: np.ndarray) -> np.ndarray:
    if wav.size == 0:
        return wav
    return wav - float(np.mean(wav))


def _peak_normalize(wav: np.ndarray, target: float) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak < 1e-6:
        return wav  # silence or near-silence, leave alone
    return (wav / peak) * target


def _mild_denoise(wav: np.ndarray, sr: int, prop: float) -> np.ndarray:
    try:
        import noisereduce as nr
    except ImportError:
        log.warning("noisereduce not installed; skipping denoise")
        return wav
    try:
        return nr.reduce_noise(
            y=wav,
            sr=sr,
            stationary=True,
            prop_decrease=float(prop),
        ).astype(np.float32)
    except Exception as err:  # pragma: no cover
        log.warning("denoise failed: %s", err)
        return wav


def preprocess(
    wav: np.ndarray,
    sample_rate: int,
    cfg: PreprocessConfig | None = None,
) -> np.ndarray:
    """Return a preprocessed copy of ``wav`` (float32)."""
    cfg = cfg or PreprocessConfig()
    out = np.asarray(wav, dtype=np.float32)
    if cfg.dc_remove:
        out = _remove_dc(out)
    if cfg.denoise:
        out = _mild_denoise(out, sample_rate, cfg.denoise_prop)
    if cfg.peak_normalize:
        out = _peak_normalize(out, cfg.peak_target)
    return out.astype(np.float32)
