"""Speaker-file time alignment by bounded cross-correlation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


class AlignmentError(RuntimeError):
    """Raised when speaker-file alignment is not reliable enough."""


@dataclass(frozen=True)
class AlignmentResult:
    offset_ms: int
    confidence: float
    lag_samples: int
    passed: bool

    def to_dict(self) -> Dict:
        return {
            "offset_ms": int(self.offset_ms),
            "confidence": round(float(self.confidence), 4),
            "lag_samples": int(self.lag_samples),
            "passed": bool(self.passed),
        }


def _preprocess(wav: np.ndarray, sample_rate: int, analysis_rate: int) -> Tuple[np.ndarray, int]:
    y = np.asarray(wav, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=-1)
    if y.size == 0:
        return y, 1
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))
    if peak > 1e-9:
        y = y / peak
    hop = max(1, int(round(sample_rate / analysis_rate)))
    return y[::hop], hop


def estimate_offset(
    speaker_a: np.ndarray,
    speaker_b: np.ndarray,
    sample_rate: int,
    *,
    max_lag_s: float = 1.5,
    min_confidence: float = 0.35,
    analysis_rate: int = 4000,
) -> Dict:
    """Estimate lag of speaker_b relative to speaker_a.

    Positive offset_ms means speaker_b appears delayed and should be shifted
    earlier by that amount for shared-timeline alignment.
    """
    a, hop = _preprocess(speaker_a, sample_rate, analysis_rate)
    b, _ = _preprocess(speaker_b, sample_rate, analysis_rate)
    if a.size < 16 or b.size < 16:
        result = AlignmentResult(0, 0.0, 0, False)
        return result.to_dict()

    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    max_lag = max(1, min(int(round(max_lag_s * sample_rate / hop)), n - 1))
    corr = np.correlate(b, a, mode="full")
    center = len(a) - 1
    window = corr[center - max_lag:center + max_lag + 1]
    if window.size == 0:
        result = AlignmentResult(0, 0.0, 0, False)
        return result.to_dict()

    peak_idx = int(np.argmax(np.abs(window)))
    lag_down = peak_idx - max_lag
    energy = float(np.linalg.norm(a) * np.linalg.norm(b))
    peak = float(abs(window[peak_idx]))
    base_conf = peak / max(energy, 1e-9)
    sorted_abs = np.sort(np.abs(window))
    second = float(sorted_abs[-2]) if sorted_abs.size > 1 else 0.0
    sharpness = (peak - second) / max(peak, 1e-9)
    confidence = float(np.clip(0.75 * base_conf + 0.25 * sharpness, 0.0, 1.0))
    lag_samples = int(lag_down * hop)
    offset_ms = int(round(lag_samples / sample_rate * 1000.0))
    result = AlignmentResult(offset_ms, confidence, lag_samples, confidence >= min_confidence)
    return result.to_dict()


def shift_waveform(wav: np.ndarray, lag_samples: int) -> np.ndarray:
    """Shift waveform onto shared timeline; positive lag moves audio earlier."""
    y = np.asarray(wav, dtype=np.float32)
    if lag_samples == 0 or y.size == 0:
        return y.copy()
    if lag_samples > 0:
        return np.pad(y[lag_samples:], (0, min(lag_samples, y.size)), mode="constant")
    pad = min(abs(lag_samples), y.size)
    return np.pad(y, (pad, 0), mode="constant")[:y.size]


def align_pair(
    speaker_a: np.ndarray,
    speaker_b: np.ndarray,
    sample_rate: int,
    *,
    min_confidence: float = 0.35,
    fail_unreliable: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    report = estimate_offset(speaker_a, speaker_b, sample_rate, min_confidence=min_confidence)
    if fail_unreliable and not report["passed"]:
        raise AlignmentError(
            f"speaker alignment unreliable: confidence={report['confidence']}, offset_ms={report['offset_ms']}"
        )
    return np.asarray(speaker_a, dtype=np.float32).copy(), shift_waveform(speaker_b, int(report["lag_samples"])), report
