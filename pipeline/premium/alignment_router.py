"""Timestamp refinement router for premium hybrid transcripts."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import PipelineConfig
from ..transcription import Transcript, TranscriptSegment, Word
from .types import AlignmentResult, TranscriptCandidate


def _premium_alignment_cfg(cfg: PipelineConfig) -> Dict[str, Any]:
    premium = dict(getattr(cfg, "premium", {}) or {})
    return dict(premium.get("alignment") or {})


def _word_timestamps_available(transcript: Transcript) -> bool:
    return any(segment.words for segment in transcript.segments)


def _segment_timestamps_available(transcript: Transcript) -> bool:
    return any(segment.end > segment.start for segment in transcript.segments)


def _fallback_word_alignment(transcript: Transcript) -> Transcript:
    rebuilt: List[TranscriptSegment] = []
    for segment in transcript.segments:
        if segment.words or not segment.text.strip():
            rebuilt.append(segment)
            continue
        tokens = [token for token in segment.text.split() if token]
        if not tokens or segment.end <= segment.start:
            rebuilt.append(segment)
            continue
        duration = (segment.end - segment.start) / len(tokens)
        words = [
            Word(
                text=token,
                start=segment.start + idx * duration,
                end=segment.start + (idx + 1) * duration,
                probability=float(segment.quality_score or 0.0),
            )
            for idx, token in enumerate(tokens)
        ]
        rebuilt.append(replace(segment, words=words))
    return Transcript(
        language=transcript.language,
        language_probability=transcript.language_probability,
        duration=transcript.duration,
        segments=rebuilt,
    )


def _refine_with_whisperx(
    transcript: Transcript,
    wav: np.ndarray,
    sample_rate: int,
    cfg: PipelineConfig,
) -> Optional[Transcript]:
    try:
        import whisperx  # type: ignore
    except ImportError:
        return None

    segments_payload = [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in transcript.segments
        if seg.text.strip()
    ]
    if not segments_payload:
        return None

    try:  # pragma: no cover
        device = getattr(cfg, "device", "cpu")
        if device == "auto":
            device = "cpu"
        align_model, metadata = whisperx.load_align_model(
            language_code=str(transcript.language or "en").split("-")[0],
            device=device,
        )
        aligned = whisperx.align(
            segments_payload,
            align_model,
            metadata,
            wav,
            device,
            return_char_alignments=False,
        )
    except Exception:
        return None

    rebuilt: List[TranscriptSegment] = []
    for base_seg, aligned_seg in zip(transcript.segments, aligned.get("segments", [])):
        words = [
            Word(
                text=str(item.get("word") or item.get("text") or "").strip(),
                start=float(item.get("start") or aligned_seg.get("start") or base_seg.start),
                end=float(item.get("end") or aligned_seg.get("end") or base_seg.end),
                probability=float(item.get("score") or base_seg.quality_score or 0.0),
            )
            for item in aligned_seg.get("words", [])
            if str(item.get("word") or item.get("text") or "").strip()
        ]
        rebuilt.append(
            replace(
                base_seg,
                start=float(aligned_seg.get("start") or base_seg.start),
                end=float(aligned_seg.get("end") or base_seg.end),
                words=words or base_seg.words,
            )
        )
    if len(transcript.segments) > len(rebuilt):
        rebuilt.extend(transcript.segments[len(rebuilt):])
    return Transcript(
        language=transcript.language,
        language_probability=transcript.language_probability,
        duration=transcript.duration,
        segments=rebuilt,
    )


def refine_timestamps(
    candidate: TranscriptCandidate,
    *,
    wav: np.ndarray,
    sample_rate: int,
    cfg: PipelineConfig,
) -> AlignmentResult:
    alignment_cfg = _premium_alignment_cfg(cfg)
    vendor_enabled = bool(alignment_cfg.get("vendor_word_timestamps_enabled", True))
    whisperx_enabled = bool(alignment_cfg.get("whisperx_enabled", False))

    if (
        vendor_enabled
        and candidate.timing_source == "vendor_word_timestamps"
        and float(candidate.timestamp_confidence or 0.0) >= 0.75
        and _word_timestamps_available(candidate.transcript)
    ):
        return AlignmentResult(
            transcript=candidate.transcript,
            timestamp_method="vendor_word_timestamps",
            timestamp_confidence=float(candidate.timestamp_confidence or 0.0),
            word_timestamps_available=True,
            segment_timestamps_available=_segment_timestamps_available(candidate.transcript),
            refinement_applied=False,
            notes=["trusted_vendor_timestamps_selected"],
        )

    if whisperx_enabled:
        refined = _refine_with_whisperx(candidate.transcript, wav, sample_rate, cfg)
        if refined is not None:
            return AlignmentResult(
                transcript=refined,
                timestamp_method="whisperx_refinement",
                timestamp_confidence=max(float(candidate.timestamp_confidence or 0.0), 0.85),
                word_timestamps_available=_word_timestamps_available(refined),
                segment_timestamps_available=_segment_timestamps_available(refined),
                refinement_applied=True,
                notes=["whisperx_alignment_applied"],
            )

    fallback = _fallback_word_alignment(candidate.transcript)
    return AlignmentResult(
        transcript=fallback,
        timestamp_method="local_alignment_fallback",
        timestamp_confidence=max(0.45, float(candidate.timestamp_confidence or 0.0)),
        word_timestamps_available=_word_timestamps_available(fallback),
        segment_timestamps_available=_segment_timestamps_available(fallback),
        refinement_applied=_word_timestamps_available(fallback) and not _word_timestamps_available(candidate.transcript),
        notes=["local_fallback_alignment_used"],
    )
