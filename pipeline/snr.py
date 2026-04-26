"""Per-segment WADA-style SNR helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class SegmentSNR:
    snr_db: float | None
    snr_band: str
    snr_reason: str | None = None
    snr_source: str = "wada_snr"
    clipping_ratio: float = 0.0
    # SNR-03: explicit quality tag when clipping is detected
    snr_quality: str | None = None

    def to_dict(self) -> dict:
        return {
            "snr_db": round(float(self.snr_db), 2) if self.snr_db is not None else None,
            "snr_band": self.snr_band,
            "snr_reason": self.snr_reason,
            "snr_source": self.snr_source,
            "clipping_ratio": round(float(self.clipping_ratio), 4),
            "snr_quality": self.snr_quality,
        }


def clipping_ratio(wav: np.ndarray) -> float:
    if wav.size == 0:
        return 0.0
    return float(np.mean(np.abs(wav.astype(np.float32)) >= 0.99))


def _rms_dbfs(wav: np.ndarray) -> float:
    if wav.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms <= 1e-12:
        return -120.0
    return 20.0 * math.log10(rms)


def wada_snr(wav: np.ndarray) -> float | None:
    """Estimate SNR without a clean reference.

    This lightweight implementation uses robust speech/noise percentiles as a
    WADA-compatible production proxy until the lookup-table estimator is tuned.
    """
    data = np.asarray(wav, dtype=np.float32)
    if data.size == 0 or _rms_dbfs(data) < -60.0 or float(np.std(data)) < 1e-6:
        return None
    mag = np.abs(data.astype(np.float64))
    noise = float(np.percentile(mag, 20))
    speech = float(np.percentile(mag, 90))
    if noise <= 1e-9 or speech <= noise:
        return None
    return float(np.clip(20.0 * math.log10(speech / noise), -10.0, 60.0))


def classify_snr(snr_db: float | None, clip_ratio: float = 0.0) -> str:
    if clip_ratio > 0.005:
        return "low"
    if snr_db is None:
        return "low"
    if snr_db >= 20.0:
        return "high"
    if snr_db >= 10.0:
        return "medium"
    return "low"


def segment_snr(
    wav: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    vad_mask: Optional[np.ndarray] = None,
) -> SegmentSNR:
    """Compute per-segment SNR.

    SNR-04: when *vad_mask* (bool array, one entry per sample aligned to *wav*)
    is supplied, SNR is estimated on VAD-positive (speech) frames only so that
    silence between words does not bias the noise-floor estimate downward.
    """
    if end - start < 0.5:
        return SegmentSNR(None, "low", "short_segment", "inherited")
    s = max(0, int(start * sample_rate))
    e = min(len(wav), int(end * sample_rate))
    chunk = np.asarray(wav[s:e], dtype=np.float32)

    # SNR-04: restrict to VAD-positive frames when mask is available.
    if vad_mask is not None and len(vad_mask) > 0:
        mask_chunk = np.asarray(vad_mask[s:e], dtype=bool)
        if mask_chunk.any():
            chunk = chunk[mask_chunk]

    clip = clipping_ratio(chunk)
    if _rms_dbfs(chunk) < -60.0:
        return SegmentSNR(None, "low", "below_noise_floor", "wada_snr", clip)
    snr = wada_snr(chunk)
    band = classify_snr(snr, clip)
    # SNR-03: emit snr_quality="clipped" when clipping forced band to low.
    snr_quality = "clipped" if clip > 0.005 else None
    reason = "clipped" if clip > 0.005 else (None if snr is not None else "unreliable_estimate")
    return SegmentSNR(snr, band, reason, "wada_snr", clip, snr_quality)


def annotate_segments_with_snr(
    segments: Iterable,
    wav: np.ndarray,
    sample_rate: int,
    vad_mask: Optional[np.ndarray] = None,
) -> List[SegmentSNR]:
    """Annotate each segment with SNR metadata.

    SNR-04: pass *vad_mask* (bool array aligned to *wav*) to restrict SNR
    estimation to speech frames only.
    """
    results: List[SegmentSNR] = []
    last_parent: SegmentSNR | None = None
    for seg in segments:
        result = segment_snr(wav, sample_rate, float(seg.start), float(seg.end), vad_mask=vad_mask)
        if result.snr_source == "inherited" and last_parent is not None:
            result = SegmentSNR(
                last_parent.snr_db,
                last_parent.snr_band,
                "short_segment_inherited",
                "inherited",
                last_parent.clipping_ratio,
                last_parent.snr_quality,
            )
        elif result.snr_db is not None:
            last_parent = result
        setattr(seg, "snr_db", result.snr_db)
        setattr(seg, "snr_band", result.snr_band)
        setattr(seg, "snr_reason", result.snr_reason)
        setattr(seg, "snr_source", result.snr_source)
        setattr(seg, "clipping_ratio", result.clipping_ratio)
        setattr(seg, "snr_quality", result.snr_quality)
        results.append(result)
    return results
