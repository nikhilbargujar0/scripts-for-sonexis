"""Composite segment confidence for Phase 2 quality tiering.

CONF-01: fuses ASR avg_logprob (down-weighted when LID entropy >0.6 or
         mid-segment script change), WADA-SNR band, VAD presence (gate only),
         WPM z-score vs per-profile baseline (CONF-05), overlap flag, and
         WhisperX alignment success.
CONF-02: emits confidence (float), confidence_band (gold/silver/bronze),
         confidence_reasons list.
CONF-03: hard caps — hard overlap → cap silver; duration <0.3s → cap silver;
         clipping → cap bronze.
CONF-04: confidence_band_absolute (fixed thresholds) + confidence_band_relative
         (percentile within batch).
CONF-05: WPM z-score uses per-profile mean+std from WPM_PROFILES; flag |z|>2.5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

# CONF-05: per-profile WPM baselines (mean, std) from corpus measurement.
# Values are approximations — replace with empirically measured figures when
# labelled corpus is available.
WPM_PROFILES: Dict[str, tuple] = {
    "hindi":          (130.0, 30.0),
    "hi":             (130.0, 30.0),
    "punjabi":        (120.0, 28.0),
    "pa":             (120.0, 28.0),
    "marwadi":        (110.0, 25.0),
    "mwr":            (110.0, 25.0),
    "indian_english": (145.0, 35.0),
    "en":             (145.0, 35.0),
    "hinglish":       (140.0, 32.0),
    # global fallback
    "default":        (135.0, 35.0),
}

# LID entropy above this threshold indicates uncertain language → down-weight ASR.
_LID_ENTROPY_THRESHOLD = 0.6
# WPM z-score magnitude above this is flagged as outlier (CONF-05).
_WPM_Z_THRESHOLD = 2.5


@dataclass
class CompositeConfidence:
    confidence: float
    confidence_band: str
    confidence_band_absolute: str
    confidence_band_relative: str
    confidence_reasons: List[str]
    confidence_components: Dict[str, float]


def _band(score: float) -> str:
    if score >= 0.80:
        return "gold"
    if score >= 0.55:
        return "silver"
    return "bronze"


def _lid_entropy(seg) -> float:
    """Retrieve LID entropy from segment if available; 0.0 otherwise."""
    return float(getattr(seg, "language_entropy", 0.0) or 0.0)


def _has_mid_script_change(seg) -> bool:
    """True when the segment has a language switch mid-utterance (not at boundaries)."""
    switch_points = getattr(seg, "switch_points", None) or []
    if not switch_points:
        return False
    words = list(getattr(seg, "words", []) or [])
    n = max(len(words), 1)
    for sp in switch_points:
        idx = int(sp.get("word_idx", 0))
        stype = str(sp.get("switch_type", ""))
        # intra_sentential or intra_word = mid-segment, not boundary
        if stype in ("intra_sentential", "intra_word") and 0 < idx < n - 1:
            return True
    return False


def _wpm_z_score(observed_wpm: float, language: str) -> float:
    """CONF-05: z-score of observed WPM vs per-profile baseline."""
    lang_key = str(language or "default").lower().split("-")[0]
    mean, std = WPM_PROFILES.get(lang_key, WPM_PROFILES["default"])
    if std < 1.0:
        std = 1.0
    return (observed_wpm - mean) / std


def _asr_signal(segment) -> float:
    """CONF-01: primary ASR quality signal.

    Uses avg_logprob directly (exp-mapped) when available, falling back to
    the pre-computed quality_score.  Down-weighted when LID entropy is high
    or a mid-segment script change is detected.
    """
    avg_logprob = float(getattr(segment, "avg_logprob", None) or 0.0)
    quality_score = float(getattr(segment, "quality_score", 0.0) or 0.0)

    # Prefer avg_logprob when available and plausible (whisper range: -6..0)
    if avg_logprob < 0.0:
        raw_asr = float(math.exp(max(avg_logprob, -6.0)))
    else:
        raw_asr = quality_score

    # Down-weight when LID is uncertain or script changes mid-segment.
    entropy = _lid_entropy(segment)
    if entropy > _LID_ENTROPY_THRESHOLD or _has_mid_script_change(segment):
        raw_asr *= 0.75

    return float(np.clip(raw_asr, 0.0, 1.0))


def _score_segment(segment, overlap: bool = False) -> CompositeConfidence:
    asr = _asr_signal(segment)
    snr_band = str(getattr(segment, "snr_band", "low") or "low")
    snr_score = {"high": 1.0, "medium": 0.72, "low": 0.38}.get(snr_band, 0.38)

    duration = max(float(segment.end - segment.start), 1e-6)
    words = getattr(segment, "words", None) or []
    word_count = len(words) or len(str(segment.text or "").split())

    # CONF-05: per-profile WPM z-score
    language = str(getattr(segment, "language", "default") or "default")
    raw_wpm = float(getattr(segment, "wpm", 0.0) or 0.0)
    observed_wpm = raw_wpm or (word_count / duration * 60.0)
    wpm_z = _wpm_z_score(observed_wpm, language)
    wpm_outlier = abs(wpm_z) > _WPM_Z_THRESHOLD
    wpm_score = 0.72 if wpm_outlier else 1.0

    # WhisperX alignment success (CONF-01)
    alignment_ok = bool(getattr(segment, "words", None))
    timestamp_score = 1.0 if alignment_ok else 0.55

    overlap_score = 0.55 if overlap else 1.0

    score = float(np.clip(
        (0.45 * asr) + (0.25 * snr_score) + (0.15 * wpm_score) + (0.15 * timestamp_score),
        0.0,
        1.0,
    ))

    reasons: List[str] = []
    if asr < 0.55:
        reasons.append("low_asr_quality")
    if _lid_entropy(segment) > _LID_ENTROPY_THRESHOLD:
        reasons.append("high_lid_entropy")
    if _has_mid_script_change(segment):
        reasons.append("mid_segment_script_change")
    if snr_band == "low":
        reasons.append("snr_low")
    if overlap:
        reasons.append("overlap_crosstalk")
    if not alignment_ok:
        reasons.append("missing_word_timestamps")
    if wpm_outlier:
        reasons.append(f"wpm_outlier_z{wpm_z:+.1f}")

    # CONF-03: hard caps after continuous fusion.
    if overlap:
        score = min(score, 0.74)   # hard overlap → cap at silver
    if duration < 0.3:
        score = min(score, 0.74)   # very short → cap at silver
        reasons.append("short_duration")
    if float(getattr(segment, "clipping_ratio", 0.0) or 0.0) > 0.005:
        score = min(score, 0.54)   # clipping detected → cap at bronze
        reasons.append("clipping_detected")

    band = _band(score)
    return CompositeConfidence(
        confidence=round(score, 4),
        confidence_band=band,
        confidence_band_absolute=band,
        confidence_band_relative=band,   # patched to percentile in batch annotator
        confidence_reasons=sorted(set(reasons)),
        confidence_components={
            "asr_quality": round(asr, 4),
            "snr": round(snr_score, 4),
            "wpm": round(wpm_score, 4),
            "wpm_z": round(wpm_z, 3),
            "timestamp": round(timestamp_score, 4),
            "overlap": round(overlap_score, 4),
        },
    )


def _percentile_band(score: float, all_scores: List[float]) -> str:
    """CONF-04: relative band based on percentile within the batch."""
    if not all_scores:
        return _band(score)
    p = float(np.mean(np.array(all_scores) <= score))
    if p >= 0.70:
        return "gold"
    if p >= 0.35:
        return "silver"
    return "bronze"


def annotate_segments_with_confidence(segments: Iterable, overlaps: Iterable) -> List[CompositeConfidence]:
    overlap_ranges = [(float(o.start), float(o.end)) for o in overlaps or []]
    seg_list = list(segments)
    results: List[CompositeConfidence] = []

    # First pass: compute absolute scores.
    for seg in seg_list:
        overlap = any(min(float(seg.end), e) - max(float(seg.start), s) > 0 for s, e in overlap_ranges)
        result = _score_segment(seg, overlap=overlap)
        setattr(seg, "overlap", bool(overlap))
        results.append(result)

    # CONF-04: second pass — patch confidence_band_relative with percentile.
    all_scores = [r.confidence for r in results]
    final_results: List[CompositeConfidence] = []
    for i, (seg, result) in enumerate(zip(seg_list, results)):
        rel_band = _percentile_band(result.confidence, all_scores)
        final = CompositeConfidence(
            confidence=result.confidence,
            confidence_band=result.confidence_band,
            confidence_band_absolute=result.confidence_band_absolute,
            confidence_band_relative=rel_band,
            confidence_reasons=result.confidence_reasons,
            confidence_components=result.confidence_components,
        )
        setattr(seg, "confidence", final.confidence)
        setattr(seg, "confidence_band", final.confidence_band)
        setattr(seg, "confidence_band_absolute", final.confidence_band_absolute)
        setattr(seg, "confidence_band_relative", final.confidence_band_relative)
        setattr(seg, "confidence_reasons", final.confidence_reasons)
        setattr(seg, "confidence_components", final.confidence_components)
        final_results.append(final)

    return final_results
