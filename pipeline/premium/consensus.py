"""Transcript consensus selection for premium ASR candidates."""
from __future__ import annotations

from dataclasses import replace
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

from ..transcription import Transcript, TranscriptSegment
from .types import ConsensusResult, TranscriptCandidate


def _tokens(text: str) -> List[str]:
    return [token for token in str(text or "").lower().split() if token]


def token_agreement(a: TranscriptCandidate, b: TranscriptCandidate) -> float:
    a_text = " ".join(_tokens(a.transcript.text))
    b_text = " ".join(_tokens(b.transcript.text))
    if not a_text and not b_text:
        return 1.0
    return SequenceMatcher(None, a_text, b_text).ratio()


def _code_switch_score(candidate: TranscriptCandidate) -> float:
    signals = candidate.code_switch_signals or {}
    detected = 0.15 if signals.get("detected") else 0.0
    switching_score = float(signals.get("switching_score") or 0.0)
    languages = min(0.15, len(signals.get("dominant_languages") or []) * 0.05)
    return min(1.0, detected + switching_score * 0.5 + languages)


def _audio_condition_weight(audio_condition: str) -> float:
    if audio_condition == "outdoor":
        return 1.1
    if audio_condition == "unknown":
        return 1.03
    return 1.0


def score_candidate(
    candidate: TranscriptCandidate,
    candidates: Iterable[TranscriptCandidate],
    *,
    audio_condition: str = "unknown",
) -> float:
    peer_scores = [
        token_agreement(candidate, other)
        for other in candidates
        if other.engine != candidate.engine
    ]
    agreement = sum(peer_scores) / len(peer_scores) if peer_scores else 1.0
    confidence = float(candidate.confidence or 0.0)
    avg_word = float(candidate.avg_word_confidence or 0.0)
    timestamp = float(candidate.timestamp_confidence or 0.0)
    code_switch = _code_switch_score(candidate)
    score = (
        confidence * 0.35
        + avg_word * 0.20
        + timestamp * 0.20
        + agreement * 0.15
        + code_switch * 0.10
    )
    return round(min(1.0, score * _audio_condition_weight(audio_condition)), 4)


def _merged_transcript(primary: Transcript, secondary: Transcript) -> Transcript:
    if not primary.segments:
        return secondary
    if not secondary.segments:
        return primary
    merged_segments: List[TranscriptSegment] = []
    count = min(len(primary.segments), len(secondary.segments))
    for idx in range(count):
        first = primary.segments[idx]
        second = secondary.segments[idx]
        merged_segments.append(
            replace(
                second,
                text=first.text or second.text,
                language=first.language or second.language,
                quality_score=max(float(first.quality_score or 0.0), float(second.quality_score or 0.0)),
            )
        )
    if len(primary.segments) > count:
        merged_segments.extend(primary.segments[count:])
    elif len(secondary.segments) > count:
        merged_segments.extend(secondary.segments[count:])
    return Transcript(
        language=primary.language or secondary.language,
        language_probability=max(primary.language_probability, secondary.language_probability),
        duration=max(primary.duration, secondary.duration),
        segments=merged_segments,
    )


def choose_consensus(
    candidates: List[TranscriptCandidate],
    *,
    audio_condition: str = "unknown",
) -> Tuple[TranscriptCandidate, ConsensusResult]:
    if not candidates:
        raise ValueError("Consensus requires at least one transcript candidate")
    if len(candidates) == 1:
        only = candidates[0]
        return only, ConsensusResult(
            transcript=only.transcript,
            selected_engine=only.engine,
            consensus_score=round(float(only.confidence or 0.0), 4),
            engines_compared=[only.engine],
            transcript_strategy="best_single_engine",
            review_recommended=bool((only.confidence or 0.0) < 0.65),
            rationale=["single_candidate_available"],
        )

    scored: List[Tuple[float, TranscriptCandidate]] = sorted(
        (
            score_candidate(candidate, candidates, audio_condition=audio_condition),
            candidate,
        )
        for candidate in candidates
    )
    scored.sort(key=lambda item: item[0], reverse=True)
    top_score, top_candidate = scored[0]
    second_score, second_candidate = scored[1]
    top_agreement = token_agreement(top_candidate, second_candidate)

    top_conf = float(top_candidate.confidence or 0.0)
    second_conf = float(second_candidate.confidence or 0.0)
    top_ts = float(top_candidate.timestamp_confidence or 0.0)
    second_ts = float(second_candidate.timestamp_confidence or 0.0)
    if (
        top_agreement >= 0.82
        and abs(top_conf - second_conf) <= 0.08
        and abs(top_ts - second_ts) >= 0.10
    ):
        text_primary = top_candidate if top_conf >= second_conf else second_candidate
        timing_primary = top_candidate if top_ts >= second_ts else second_candidate
        merged_candidate = TranscriptCandidate(
            engine="merged_consensus",
            provider="consensus",
            paid_api=top_candidate.paid_api or second_candidate.paid_api,
            transcript=_merged_transcript(text_primary.transcript, timing_primary.transcript),
            confidence=max(top_conf, second_conf),
            avg_word_confidence=max(
                float(top_candidate.avg_word_confidence or 0.0),
                float(second_candidate.avg_word_confidence or 0.0),
            ),
            language_hint=text_primary.language_hint or timing_primary.language_hint,
            detected_languages=list({
                *top_candidate.detected_languages,
                *second_candidate.detected_languages,
            }),
            code_switch_signals=text_primary.code_switch_signals or timing_primary.code_switch_signals,
            timing_source=timing_primary.timing_source,
            timestamp_confidence=max(top_ts, second_ts),
            warnings=list({*top_candidate.warnings, *second_candidate.warnings}),
            adapter_metadata={
                "merged_from": [top_candidate.engine, second_candidate.engine],
                "agreement": round(top_agreement, 4),
            },
        )
        return merged_candidate, ConsensusResult(
            transcript=merged_candidate.transcript,
            selected_engine=merged_candidate.engine,
            consensus_score=max(top_score, second_score),
            engines_compared=[candidate.engine for candidate in candidates],
            transcript_strategy="merged_consensus",
            review_recommended=max(top_score, second_score) < 0.72,
            rationale=[
                "high_text_agreement",
                "secondary_candidate_has_stronger_timestamps",
            ],
        )

    return top_candidate, ConsensusResult(
        transcript=top_candidate.transcript,
        selected_engine=top_candidate.engine,
        consensus_score=top_score,
        engines_compared=[candidate.engine for candidate in candidates],
        transcript_strategy="best_single_engine",
        review_recommended=top_score < 0.72 or top_agreement < 0.70,
        rationale=[
            f"top_score={top_score:.4f}",
            f"runner_up_score={second_score:.4f}",
            f"top_pair_agreement={top_agreement:.4f}",
        ],
    )
