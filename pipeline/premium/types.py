"""Shared premium hybrid pipeline dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..transcription import Transcript


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def transcript_to_payload(
    transcript: Transcript,
    *,
    include_segments: bool = True,
    include_words: bool = True,
) -> Dict[str, Any]:
    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    if include_segments:
        for segment in transcript.segments:
            payload = segment.to_dict()
            if not include_words:
                payload["words"] = []
            segments.append(payload)
    if include_words:
        words = [
            word.to_dict()
            for segment in transcript.segments
            for word in (segment.words or [])
        ]
    return {
        "raw": transcript.text,
        "language": transcript.language,
        "language_probability": round(float(transcript.language_probability), 4),
        "duration_s": round(float(transcript.duration), 3),
        "segments": segments,
        "words": words,
    }


@dataclass
class TranscriptCandidate:
    engine: str
    transcript: Transcript
    provider: Optional[str] = None
    paid_api: bool = False
    confidence: Optional[float] = None
    avg_word_confidence: Optional[float] = None
    language_hint: Optional[str] = None
    detected_languages: List[str] = field(default_factory=list)
    code_switch_signals: Dict[str, Any] = field(default_factory=dict)
    timing_source: str = "local_word_timestamps"
    timestamp_confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    adapter_metadata: Dict[str, Any] = field(default_factory=dict)
    normalisation_notes: List[str] = field(default_factory=list)

    def to_dict(
        self,
        *,
        include_segments: bool = True,
        include_words: bool = True,
    ) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "provider": self.provider or self.engine,
            "paid_api": bool(self.paid_api),
            "transcript": transcript_to_payload(
                self.transcript,
                include_segments=include_segments,
                include_words=include_words,
            ),
            "confidence": _round_or_none(self.confidence),
            "avg_word_confidence": _round_or_none(self.avg_word_confidence),
            "language_hint": self.language_hint,
            "detected_languages": list(self.detected_languages),
            "code_switch_signals": dict(self.code_switch_signals),
            "timing_source": self.timing_source,
            "timestamp_confidence": _round_or_none(self.timestamp_confidence),
            "warnings": list(self.warnings),
            "adapter_metadata": dict(self.adapter_metadata),
            "normalisation_notes": list(self.normalisation_notes),
        }


@dataclass
class RoutingDecision:
    pipeline_mode: str
    paid_api_allowed: bool
    local_first: bool = True
    difficulty_score: float = 0.0
    should_escalate: bool = False
    escalated_to_paid: bool = False
    reasons: List[str] = field(default_factory=list)
    attempted_engines: List[str] = field(default_factory=list)
    skipped_engines: List[str] = field(default_factory=list)
    engines_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_mode": self.pipeline_mode,
            "paid_api_allowed": bool(self.paid_api_allowed),
            "local_first": bool(self.local_first),
            "difficulty_score": round(float(self.difficulty_score), 4),
            "should_escalate": bool(self.should_escalate),
            "escalated_to_paid": bool(self.escalated_to_paid),
            "reasons": list(self.reasons),
            "attempted_engines": list(self.attempted_engines),
            "skipped_engines": list(self.skipped_engines),
            "engines_attempted": list(self.attempted_engines),
            "engines_skipped": list(self.skipped_engines),
            "engines_used": list(self.engines_used),
        }


@dataclass
class ConsensusResult:
    transcript: Transcript
    selected_engine: str
    consensus_score: float
    engines_compared: List[str]
    transcript_strategy: str = "best_single_engine"
    review_recommended: bool = False
    rationale: List[str] = field(default_factory=list)
    candidate_rationales: Dict[str, List[str]] = field(default_factory=dict)
    disagreement_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_engine": self.selected_engine,
            "consensus_score": round(float(self.consensus_score), 4),
            "engines_compared": list(self.engines_compared),
            "transcript_strategy": self.transcript_strategy,
            "review_recommended": bool(self.review_recommended),
            "rationale": list(self.rationale),
            "candidate_rationales": {key: list(value) for key, value in self.candidate_rationales.items()},
            "disagreement_flags": list(self.disagreement_flags),
        }


@dataclass
class AlignmentResult:
    transcript: Transcript
    timestamp_method: str
    timestamp_confidence: float
    word_timestamps_available: bool
    segment_timestamps_available: bool
    refinement_applied: bool = False
    synthetic_word_timestamps: bool = False
    timing_quality: str = "medium"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_method": self.timestamp_method,
            "timestamp_confidence": round(float(self.timestamp_confidence), 4),
            "word_timestamps_available": bool(self.word_timestamps_available),
            "segment_timestamps_available": bool(self.segment_timestamps_available),
            "refinement_applied": bool(self.refinement_applied),
            "synthetic_word_timestamps": bool(self.synthetic_word_timestamps),
            "timing_quality": self.timing_quality,
            "notes": list(self.notes),
        }


@dataclass
class PremiumProcessingReport:
    pipeline_mode: str
    paid_api_used: bool
    engines_used: List[str]
    consensus_applied: bool
    timestamp_refinement_applied: bool
    human_review_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_mode": self.pipeline_mode,
            "paid_api_used": bool(self.paid_api_used),
            "engines_used": list(self.engines_used),
            "consensus_applied": bool(self.consensus_applied),
            "timestamp_refinement_applied": bool(self.timestamp_refinement_applied),
            "human_review_required": bool(self.human_review_required),
        }
