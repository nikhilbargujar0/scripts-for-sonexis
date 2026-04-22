"""pipeline/steps/alignment.py

Cross-correlation based speaker-pair clock alignment.

When two speakers are recorded on separate devices (e.g. separate phone legs,
different laptops) that were started at slightly different times, even a small
constant offset skews interaction metrics: overlaps, response latency, and
turn-switch events are all shifted by the same bias.

Algorithm
---------
1. Normalise both waveforms (DC-remove, unit-energy).
2. Compute normalised cross-correlation over a central window of *window_s*
   seconds.  Using the centre avoids start/end silence artefacts.
3. Search only within ±*max_offset_s* samples to rule out pathological peaks.
4. Find the lag that maximises |correlation|.
5. Shift one waveform so that both share a common time axis.
6. Return aligned waveforms + alignment metadata.
7. Raise AlignmentError if confidence is below *min_confidence*.

Confidence
----------
The confidence is the absolute peak of the normalised cross-correlation
(range [0, 1]).  For telephony pairs sharing a common far-end mix the peak
should be > 0.3.  Pairs recorded in different acoustic environments may score
lower; the default threshold is intentionally conservative at 0.10.

Usage
-----
::

    from pipeline.steps.alignment import align_speaker_pair, AlignmentError

    try:
        result = align_speaker_pair(wav1, wav2, sample_rate=16_000)
        aligned_wav1 = result.wav1
        aligned_wav2 = result.wav2
        print(result.to_dict())
    except AlignmentError as e:
        # Confidence too low — fall back to shared_zero_start assumption
        log.warning("alignment skipped: %s", e)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── tuneable defaults ──────────────────────────────────────────────────────

_DEFAULT_MAX_OFFSET_S: float = 5.0   # max clock drift to search (seconds)
_DEFAULT_MIN_CONFIDENCE: float = 0.10  # raise AlignmentError below this
_DEFAULT_WINDOW_S: float = 10.0      # central analysis window (seconds)


# ── public exceptions / result dataclass ──────────────────────────────────

class AlignmentError(RuntimeError):
    """Raised when alignment confidence is too low to trust the result."""


@dataclass
class AlignmentResult:
    """Output of :func:`align_speaker_pair`.

    Attributes
    ----------
    wav1 :
        Aligned speaker-1 waveform (reference; may be zero-padded to match
        length of ``wav2``).
    wav2 :
        Aligned speaker-2 waveform (shifted by *offset_samples* and
        zero-padded to the same length as ``wav1``).
    sample_rate :
        Shared sample rate in Hz.
    offset_samples :
        Signed offset of wav2 relative to wav1.
        **Negative** → wav2 lags wav1 (wav2's content appears later in time).
        **Positive** → wav2 leads wav1 (wav2's content appears earlier in time).
        Matches the ``scipy.signal.correlation_lags`` convention.
    offset_s :
        Offset in seconds (``offset_samples / sample_rate``).
    confidence :
        Normalised cross-correlation peak in [0, 1].  Higher is better.
    method :
        Always ``"cross_correlation"`` for this implementation.
    """
    wav1: np.ndarray
    wav2: np.ndarray
    sample_rate: int
    offset_samples: int
    offset_s: float
    confidence: float
    method: str = "cross_correlation"

    # Sign convention (matches scipy.signal.correlation_lags):
    #   negative offset → wav2 lags wav1  (wav2 started later)
    #   positive offset → wav2 leads wav1 (wav2 started earlier)

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "offset_samples": int(self.offset_samples),
            "offset_s": round(float(self.offset_s), 4),
            "confidence": round(float(self.confidence), 4),
        }


# ── internal helpers ───────────────────────────────────────────────────────

def _normalised_xcorr(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """FFT-based normalised cross-correlation.

    Both arrays are DC-removed and energy-normalised before correlation so
    that the peak value lies in [-1, 1] regardless of absolute amplitude.
    Returns ``(corr, lags)`` where ``lags[i]`` is the delay of *b* relative
    to *a* that corresponds to ``corr[i]``.

    Convention: positive lag → b lags a (b started later).
                negative lag → b leads a (b started earlier).
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    # DC removal
    a -= a.mean()
    b -= b.mean()

    # Energy normalisation
    energy_a = float(np.sqrt(np.sum(a ** 2)))
    energy_b = float(np.sqrt(np.sum(b ** 2)))
    if energy_a < 1e-10 or energy_b < 1e-10:
        n_out = len(a) + len(b) - 1
        return np.zeros(n_out), np.arange(-(len(b) - 1), len(a))
    a /= energy_a
    b /= energy_b

    # Full cross-correlation via FFT (equivalent to np.correlate mode='full')
    n_fft = len(a) + len(b) - 1
    fa = np.fft.rfft(a, n=n_fft)
    fb = np.fft.rfft(b, n=n_fft)
    corr_circ = np.fft.irfft(fa * np.conj(fb), n=n_fft)

    # Convert from circular to linear ("full") convention:
    # corr_circ[0]           → lag 0
    # corr_circ[1..len(a)-1] → lag +1..+(len(a)-1)  (a leads b)
    # corr_circ[n_fft-len(b)+1..n_fft-1] → lag -(len(b)-1)..-1  (b leads a)
    corr = np.concatenate([corr_circ[n_fft - len(b) + 1:], corr_circ[:len(a)]])
    lags = np.arange(-(len(b) - 1), len(a))
    return corr, lags


def _apply_offset(
    wav1: np.ndarray,
    wav2: np.ndarray,
    offset_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply *offset_samples* shift to wav2 relative to wav1.

    Positive offset → wav2 originally started *later* → prepend zeros to
    wav2 so that both time axes share the same origin.
    Negative offset → wav2 originally started *earlier* → prepend zeros to
    wav1.
    Both output arrays are zero-padded to the same length.
    """
    dtype = np.float32

    if offset_samples > 0:
        pad = np.zeros(offset_samples, dtype=dtype)
        wav2_shifted = np.concatenate([pad, wav2.astype(dtype)])
        n = max(len(wav1), len(wav2_shifted))
        w1 = np.zeros(n, dtype=dtype)
        w2 = np.zeros(n, dtype=dtype)
        w1[:len(wav1)] = wav1.astype(dtype)
        w2[:len(wav2_shifted)] = wav2_shifted
    elif offset_samples < 0:
        shift = -offset_samples
        pad = np.zeros(shift, dtype=dtype)
        wav1_shifted = np.concatenate([pad, wav1.astype(dtype)])
        n = max(len(wav1_shifted), len(wav2))
        w1 = np.zeros(n, dtype=dtype)
        w2 = np.zeros(n, dtype=dtype)
        w1[:len(wav1_shifted)] = wav1_shifted
        w2[:len(wav2)] = wav2.astype(dtype)
    else:
        n = max(len(wav1), len(wav2))
        w1 = np.zeros(n, dtype=dtype)
        w2 = np.zeros(n, dtype=dtype)
        w1[:len(wav1)] = wav1.astype(dtype)
        w2[:len(wav2)] = wav2.astype(dtype)

    return w1, w2


# ── public API ─────────────────────────────────────────────────────────────

def align_speaker_pair(
    wav1: np.ndarray,
    wav2: np.ndarray,
    sample_rate: int,
    max_offset_s: float = _DEFAULT_MAX_OFFSET_S,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    window_s: float = _DEFAULT_WINDOW_S,
) -> AlignmentResult:
    """Align two mono speaker waveforms via cross-correlation.

    Parameters
    ----------
    wav1, wav2 :
        Float32 mono waveforms.  May have different lengths.
    sample_rate :
        Shared target sample rate (Hz).  Both arrays must already be
        resampled to this rate before calling.
    max_offset_s :
        Maximum plausible clock drift to search (seconds).  Offsets larger
        than this almost certainly indicate different recording sessions.
    min_confidence :
        Minimum normalised correlation peak accepted as a valid alignment.
        Below this threshold :class:`AlignmentError` is raised.
    window_s :
        Duration (seconds) of the central analysis window.  Using the centre
        avoids start/end silence periods that would produce spurious peaks.

    Returns
    -------
    AlignmentResult
        Contains aligned waveforms (same length, float32) and metadata.

    Raises
    ------
    AlignmentError
        If either waveform is empty, shorter than one second, or if the
        peak cross-correlation is below *min_confidence*.
    """
    if wav1.size == 0 or wav2.size == 0:
        raise AlignmentError("One or both waveforms are empty.")

    max_offset_samples = int(max_offset_s * sample_rate)
    window_samples = int(window_s * sample_rate)
    n = min(len(wav1), len(wav2))

    if n < sample_rate:  # less than 1 second
        raise AlignmentError(
            f"Signals too short for alignment: {n} samples "
            f"({n / sample_rate:.2f}s).  Need at least 1.0s."
        )

    # ── Extract central analysis window ──────────────────────────────
    mid = n // 2
    half_window = window_samples // 2
    start = max(0, mid - half_window)
    end = min(n, start + window_samples)
    # Guard: if window is larger than n, use the whole signal.
    if end - start < max(1, min(n, window_samples)):
        start, end = 0, n

    chunk1 = wav1[start:end].astype(np.float32)
    chunk2 = wav2[start:end].astype(np.float32)

    # ── Cross-correlation ─────────────────────────────────────────────
    corr, lags = _normalised_xcorr(chunk1, chunk2)

    # Restrict search to ±max_offset_samples
    valid_mask = np.abs(lags) <= max_offset_samples
    valid_corr = corr[valid_mask]
    valid_lags = lags[valid_mask]

    if valid_corr.size == 0:
        raise AlignmentError(
            f"No valid lag range within ±{max_offset_s}s of cross-correlation result."
        )

    best_idx = int(np.argmax(np.abs(valid_corr)))
    offset_samples = int(valid_lags[best_idx])
    confidence = float(np.clip(np.abs(valid_corr[best_idx]), 0.0, 1.0))

    log.info(
        "alignment: offset=%+.3fs (%+d samples)  confidence=%.3f",
        offset_samples / sample_rate,
        offset_samples,
        confidence,
    )

    if confidence < min_confidence:
        raise AlignmentError(
            f"Alignment confidence {confidence:.3f} is below the threshold "
            f"{min_confidence:.3f}.  The speakers may not share a common "
            "acoustic source, or the recordings are too different to align."
        )

    # ── Apply offset to produce equal-length aligned arrays ───────────
    wav1_aligned, wav2_aligned = _apply_offset(wav1, wav2, offset_samples)

    return AlignmentResult(
        wav1=wav1_aligned,
        wav2=wav2_aligned,
        sample_rate=sample_rate,
        offset_samples=offset_samples,
        offset_s=round(offset_samples / sample_rate, 4),
        confidence=confidence,
    )
