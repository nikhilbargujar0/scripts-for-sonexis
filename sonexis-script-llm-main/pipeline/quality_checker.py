"""quality_checker.py

Pre-processing quality validation for audio sessions.

Checks:
  - Clipping ratio        (fraction of samples at ±0.99 peak)
  - Silence ratio         (fraction of 50-ms frames below RMS threshold)
  - Duration mismatch     (abs difference between two speaker files)
  - Minimum duration      (audio too short for meaningful ASR)

Returns a QualityReport with .passed / .warnings / .errors.
Integrates with PipelineConfig thresholds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .audio_loader import LoadedAudio, SpeakerPairAudio


# ── thresholds ─────────────────────────────────────────────────────────────

_CLIP_THRESHOLD: float = 0.99       # |sample| >= this → clipped
_SILENCE_RMS_THRESHOLD: float = 0.005  # RMS below this → silent frame
_FRAME_SIZE_S: float = 0.050        # 50 ms frames for silence analysis
_CLIP_RATIO_WARN: float = 0.01      # > 1 % clipped → warning
_CLIP_RATIO_ERROR: float = 0.05     # > 5 % clipped → error
_SILENCE_RATIO_WARN: float = 0.50   # > 50 % silence → warning
_SILENCE_RATIO_ERROR: float = 0.80  # > 80 % silence → error


# ── report dataclass ────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # numeric diagnostics (None if not applicable)
    clipping_ratio: Optional[float] = None      # fraction of clipped samples
    silence_ratio: Optional[float] = None       # fraction of silent frames
    duration_mismatch_s: Optional[float] = None # |spk_a_dur - spk_b_dur|
    duration_s: Optional[float] = None          # total audio duration

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "clipping_ratio": round(self.clipping_ratio, 4) if self.clipping_ratio is not None else None,
            "silence_ratio": round(self.silence_ratio, 4) if self.silence_ratio is not None else None,
            "duration_mismatch_s": round(self.duration_mismatch_s, 3) if self.duration_mismatch_s is not None else None,
            "duration_s": round(self.duration_s, 3) if self.duration_s is not None else None,
        }


# ── internal helpers ────────────────────────────────────────────────────────

def _clipping_ratio(wav: np.ndarray) -> float:
    """Fraction of samples where |amplitude| >= _CLIP_THRESHOLD."""
    if wav.size == 0:
        return 0.0
    clipped = np.sum(np.abs(wav) >= _CLIP_THRESHOLD)
    return float(clipped) / wav.size


def _silence_ratio(wav: np.ndarray, sr: int) -> float:
    """Fraction of 50-ms frames whose RMS is below the silence threshold."""
    if wav.size == 0:
        return 1.0
    frame_len = max(1, int(sr * _FRAME_SIZE_S))
    n_frames = math.ceil(wav.size / frame_len)
    silent = 0
    for i in range(n_frames):
        frame = wav[i * frame_len : (i + 1) * frame_len]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if rms < _SILENCE_RMS_THRESHOLD:
            silent += 1
    return silent / n_frames


def _check_wav(
    wav: np.ndarray,
    sr: int,
    label: str,
    min_duration_s: float,
) -> tuple[List[str], List[str], float, float]:
    """Return (warnings, errors, clip_ratio, sil_ratio) for a single waveform."""
    warnings: List[str] = []
    errors: List[str] = []

    dur = wav.size / max(sr, 1)
    if dur < min_duration_s:
        errors.append(f"{label}: duration {dur:.2f}s below minimum {min_duration_s:.1f}s")

    # Clipping
    cr = _clipping_ratio(wav)
    if cr >= _CLIP_RATIO_ERROR:
        errors.append(f"{label}: heavy clipping ({cr*100:.1f}% samples)")
    elif cr >= _CLIP_RATIO_WARN:
        warnings.append(f"{label}: mild clipping ({cr*100:.1f}% samples)")

    # Silence
    sr_ratio = _silence_ratio(wav, sr)
    if sr_ratio >= _SILENCE_RATIO_ERROR:
        errors.append(f"{label}: mostly silent ({sr_ratio*100:.0f}% frames)")
    elif sr_ratio >= _SILENCE_RATIO_WARN:
        warnings.append(f"{label}: high silence ratio ({sr_ratio*100:.0f}% frames)")

    return warnings, errors, cr, sr_ratio


# ── public API ──────────────────────────────────────────────────────────────

def check_mono(
    audio: LoadedAudio,
    min_duration_s: float = 1.0,
) -> QualityReport:
    """Quality check for a single mono audio file."""
    wav = audio.waveform.astype(np.float32)
    # Normalise int16 if needed
    if wav.max() > 1.0:
        wav = wav / 32768.0

    warnings, errors, cr, sr_ratio = _check_wav(
        wav, audio.sample_rate, audio.path, min_duration_s
    )
    dur = wav.size / max(audio.sample_rate, 1)

    return QualityReport(
        passed=len(errors) == 0,
        warnings=warnings,
        errors=errors,
        clipping_ratio=cr,
        silence_ratio=sr_ratio,
        duration_s=dur,
    )


def check_speaker_pair(
    pair: SpeakerPairAudio,
    min_duration_s: float = 1.0,
    max_duration_mismatch_s: float = 60.0,
) -> QualityReport:
    """Quality check for a dual-speaker session.

    Checks each speaker independently and also validates that the two
    speaker files are not wildly different in duration.
    """
    all_warnings: List[str] = []
    all_errors: List[str] = []
    clip_ratios: List[float] = []
    sil_ratios: List[float] = []
    durations: List[float] = []

    for spk_id, audio in pair.speakers.items():
        label = pair.speaker_map.get(spk_id, spk_id)
        wav = audio.waveform.astype(np.float32)
        if wav.max() > 1.0:
            wav = wav / 32768.0

        w, e, cr, sr_ratio = _check_wav(wav, audio.sample_rate, label, min_duration_s)
        all_warnings.extend(w)
        all_errors.extend(e)
        clip_ratios.append(cr)
        sil_ratios.append(sr_ratio)
        durations.append(wav.size / max(audio.sample_rate, 1))

    # Duration mismatch between speakers
    mismatch: Optional[float] = None
    if len(durations) == 2:
        mismatch = abs(durations[0] - durations[1])
        if mismatch > max_duration_mismatch_s:
            all_warnings.append(
                f"Duration mismatch {mismatch:.1f}s between speakers "
                f"(threshold {max_duration_mismatch_s:.0f}s)"
            )

    total_dur = pair.mixed.waveform.size / max(pair.mixed.sample_rate, 1) if pair.mixed else None

    return QualityReport(
        passed=len(all_errors) == 0,
        warnings=all_warnings,
        errors=all_errors,
        clipping_ratio=float(np.mean(clip_ratios)) if clip_ratios else None,
        silence_ratio=float(np.mean(sil_ratios)) if sil_ratios else None,
        duration_mismatch_s=mismatch,
        duration_s=total_dur,
    )
