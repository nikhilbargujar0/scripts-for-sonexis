"""pipeline/steps/preprocessing.py

Re-exports from pipeline.preprocessing plus ``preprocess_speaker_pair``.

``preprocess_speaker_pair`` normalises each speaker independently before
mixing so that a dominant / loud speaker does not suppress the other one.
It:
  1. DC-removes each channel.
  2. Peak-normalises each channel to a uniform target level.
  3. Mixes with equal weight (no blind overlay).
  4. Applies a final peak-normalise + hard-clip guard on the mix.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from pipeline.preprocessing import (  # noqa: F401
    PreprocessConfig,
    _mild_denoise,
    _peak_normalize,
    _remove_dc,
    preprocess,
)

log = logging.getLogger(__name__)

__all__ = [
    "PreprocessConfig",
    "preprocess",
    "preprocess_speaker_pair",
]


def preprocess_speaker_pair(
    speaker_wavs: Dict[str, np.ndarray],
    sample_rate: int,
    cfg: Optional[PreprocessConfig] = None,
    peak_target: float = 0.85,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Preprocess two speaker waveforms and return a safe mixed signal.

    Each speaker is independently DC-removed and peak-normalised to
    *peak_target* before mixing.  This prevents a loud speaker from
    clipping the mix and ensures both voices are at comparable levels.
    The final mix is hard-clipped at ±1.0 as a safety guard.

    Parameters
    ----------
    speaker_wavs :
        ``{label: waveform_float32}``.  Typically two entries.
    sample_rate :
        Shared sample rate in Hz (used by optional denoiser).
    cfg :
        Optional :class:`PreprocessConfig`.  Defaults to DC-remove +
        peak-normalise only (``denoise=False``).
    peak_target :
        Target peak amplitude for each individual speaker (< 1.0 so
        that the mix stays within [-1, 1] when both speakers overlap).

    Returns
    -------
    (processed_wavs, mixed_wav) :
        ``processed_wavs`` — dict with the same keys as *speaker_wavs*
        but each value normalised independently.
        ``mixed_wav``      — equal-weight mix, peak-normalised to 0.95,
        clipped to [-1, 1].
    """
    cfg = cfg or PreprocessConfig()

    processed: Dict[str, np.ndarray] = {}
    for label, wav in speaker_wavs.items():
        out = np.asarray(wav, dtype=np.float32)
        # DC removal keeps the mix centred around zero.
        if cfg.dc_remove:
            out = _remove_dc(out)
        # Optional per-speaker denoise (off by default).
        if cfg.denoise:
            out = _mild_denoise(out, sample_rate, cfg.denoise_prop)
        # Normalise to *peak_target* — independent of other speakers.
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1e-6:
            out = (out / peak) * peak_target
        processed[label] = out.astype(np.float32)

    # Equal-weight mix
    if not processed:
        return processed, np.zeros(0, dtype=np.float32)

    n = max(len(w) for w in processed.values())
    mix = np.zeros(n, dtype=np.float64)
    for wav in processed.values():
        mix[:len(wav)] += wav.astype(np.float64)
    mix /= max(1, len(processed))

    # Final gain normalise + hard clip guard
    peak_mix = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak_mix > 1e-6:
        mix = mix / peak_mix * 0.95
    mix = np.clip(mix, -1.0, 1.0)

    return processed, mix.astype(np.float32)
