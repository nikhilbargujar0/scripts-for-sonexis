"""Multilingual-aware transcript consensus selection for premium ASR candidates."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Sequence, Tuple

from ..transcription import Transcript, TranscriptSegment
from .types import ConsensusResult, TranscriptCandidate

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_INDIC_DIGITS = str.maketrans("०१२३४५६७८९੦੧੨੩੪੫੬੭੮੯", "01234567890123456789")


def _normalize_text(text: str) -> str:
    collapsed = _WHITESPACE.sub(" ", str(text or "").strip()).lower()
    collapsed = collapsed.translate(_INDIC_DIGITS)
    collapsed = _PUNCT.sub(" ", collapsed)
    return _WHITESPACE.sub(" ", collapsed).strip()


def _tokens(text: str) -> List[str]:
    return [token for token in _normalize_text(text).split(" ") if token]


def _token_sequence_similarity(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    if not a_tokens and not b_tokens:
        return 1.0
    return SequenceMatcher(None, list(a_tokens), list(b_tokens)).ratio()


def _token_bag_overlap(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    if not a_tokens and not b_tokens:
        return 1.0
    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    shared = sum(min(a_counts[token], b_counts[token]) for token in set(a_counts) | set(b_counts))
    total = max(sum(a_counts.values()), sum(b_counts.values()), 1)
    return shared / total


def token_agreement(a: TranscriptCandidate, b: TranscriptCandidate) -> float:
    a_tokens = _tokens(a.transcript.text)
    b_tokens = _tokens(b.transcript.text)
    sequence = _token_sequence_similarity(a_tokens, b_tokens)
    bag = _token_bag_overlap(a_tokens, b_tokens)
    return round((sequence * 0.65) + (bag * 0.35), 4)


def _segment_count_penalty(a: Transcript, b: Transcript) -> float:
    a_count = len([segment for segment in a.segments if segment.text.strip()])
    b_count = len([segment for segment in b.segments if segment.text.strip()])
    if max(a_count, b_count, 1) == 0:
        return 1.0
    return 1.0 - (abs(a_count - b_count) / max(a_count, b_count, 1))


def _timing_agreement(a: TranscriptCandidate, b: TranscriptCandidate) -> float:
    a_segments = [segment for segment in a.transcript.segments if segment.text.strip()]
    b_segments = [segment for segment in b.transcript.segments if segment.text.strip()]
    if not a_segments or not b_segments:
        return 0.0
    count = min(len(a_segments), len(b_segments))
    if count == 0:
        return 0.0
    deltas: List[float] = []
    for idx in range(count):
        first = a_segments[idx]
        second = b_segments[idx]
        deltas.append(abs(float(first.start) - float(second.start)))
        deltas.append(abs(float(first.end) - float(second.end)))
    mean_delta = sum(deltas) / max(len(deltas), 1)
    structure = _segment_count_penalty(a.transcript, b.transcript)
    agreement = max(0.0, 1.0 - min(1.0, mean_delta / 1.2))
    return round((agreement * 0.7) + (structure * 0.3), 4)


def _switch_structure(candidate: TranscriptCandidate) -> Tuple[List[str], int, List[str]]:
    signals = candidate.code_switch_signals or {}
    dominant = [str(value) for value in signals.get("dominant_languages") or [] if value]
    patterns = [str(value) for value in signals.get("switch_patterns") or [] if value]
    switch_count = int(signals.get("switch_count") or 0)
    return dominant, switch_count, patterns


def _switch_structure_agreement(a: TranscriptCandidate, b: TranscriptCandidate) -> float:
    a_dom, a_count, a_patterns = _switch_structure(a)
    b_dom, b_count, b_patterns = _switch_structure(b)
    shared_langs = len(set(a_dom) & set(b_dom))
    total_langs = max(len(set(a_dom) | set(b_dom)), 1)
    lang_score = shared_langs / total_langs
    count_score = 1.0 - min(1.0, abs(a_count - b_count) / max(max(a_count, b_count), 1))
    shared_patterns = len(set(a_patterns) & set(b_patterns))
    total_patterns = max(len(set(a_patterns) | set(b_patterns)), 1)
    pattern_score = shared_patterns / total_patterns if (a_patterns or b_patterns) else 1.0
    return round((lang_score * 0.4) + (count_score * 0.3) + (pattern_score * 0.3), 4)


def _code_switch_preservation_score(candidate: TranscriptCandidate, peers: Iterable[TranscriptCandidate]) -> float:
    signals = candidate.code_switch_signals or {}
    detected = bool(signals.get("detected"))
    switching_score = float(signals.get("switching_score") or 0.0)
    dominant_languages = list(signals.get("dominant_languages") or [])
    peer_detections = [bool((peer.code_switch_signals or {}).get("detected")) for peer in peers if peer.engine != candidate.engine]
    peer_multilingual = any(len((peer.code_switch_signals or {}).get("dominant_languages") or []) >= 2 for peer in peers if peer.engine != candidate.engine)
    score = min(1.0, switching_score + min(0.3, len(dominant_languages) * 0.1))
    if detected:
        score = max(score, 0.5)
    if peer_detections and not detected:
        score *= 0.55
    if peer_multilingual and len(dominant_languages) < 2:
        score *= 0.65
    return round(score, 4)


def _candidate_rationale(
    candidate: TranscriptCandidate,
    *,
    agreement: float,
    timing: float,
    switch_structure: float,
    code_switch_score: float,
) -> List[str]:
    rationale = [
        f"token_agreement={agreement:.4f}",
        f"timing_agreement={timing:.4f}",
        f"code_switch_preservation={code_switch_score:.4f}",
        f"switch_structure_agreement={switch_structure:.4f}",
    ]
    if candidate.normalisation_notes:
        rationale.append("normalisation=" + ",".join(candidate.normalisation_notes))
    if float(candidate.timestamp_confidence or 0.0) < 0.55:
        rationale.append("weak_timestamp_confidence")
    if len((candidate.code_switch_signals or {}).get("dominant_languages") or []) < 2 and bool((candidate.code_switch_signals or {}).get("detected")):
        rationale.append("collapsed_multilingual_boundaries")
    return rationale


def score_candidate(
    candidate: TranscriptCandidate,
    candidates: Iterable[TranscriptCandidate],
    *,
    audio_condition: str = "unknown",
) -> Tuple[float, Dict[str, float], List[str]]:
    peers = [other for other in candidates if other.engine != candidate.engine]
    if peers:
        token_scores = [token_agreement(candidate, other) for other in peers]
        timing_scores = [_timing_agreement(candidate, other) for other in peers]
        switch_scores = [_switch_structure_agreement(candidate, other) for other in peers]
        agreement = sum(token_scores) / len(token_scores)
        timing = sum(timing_scores) / len(timing_scores)
        switch_structure = sum(switch_scores) / len(switch_scores)
    else:
        agreement = 1.0
        timing = 1.0
        switch_structure = 1.0
    confidence = float(candidate.confidence or 0.0)
    avg_word = float(candidate.avg_word_confidence or 0.0)
    timestamp = float(candidate.timestamp_confidence or 0.0)
    code_switch_score = _code_switch_preservation_score(candidate, peers)
    score = (
        confidence * 0.24
        + avg_word * 0.14
        + timestamp * 0.16
        + agreement * 0.20
        + timing * 0.10
        + switch_structure * 0.08
        + code_switch_score * 0.08
    )
    if audio_condition == "outdoor":
        score *= 1.03
    elif audio_condition == "unknown":
        score *= 1.01
    rationale = _candidate_rationale(
        candidate,
        agreement=agreement,
        timing=timing,
        switch_structure=switch_structure,
        code_switch_score=code_switch_score,
    )
    metrics = {
        "agreement": round(agreement, 4),
        "timing": round(timing, 4),
        "switch_structure": round(switch_structure, 4),
        "code_switch_score": round(code_switch_score, 4),
    }
    return round(min(1.0, score), 4), metrics, rationale


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


def _disagreement_flags(top: TranscriptCandidate, runner_up: TranscriptCandidate) -> List[str]:
    flags: List[str] = []
    if token_agreement(top, runner_up) < 0.72:
        flags.append("strong_transcript_disagreement")
    if _timing_agreement(top, runner_up) < 0.70:
        flags.append("strong_timestamp_disagreement")
    if _switch_structure_agreement(top, runner_up) < 0.70:
        flags.append("strong_code_switch_disagreement")
    return flags


def _merge_allowed(top: TranscriptCandidate, runner_up: TranscriptCandidate) -> bool:
    agreement = token_agreement(top, runner_up)
    timing = _timing_agreement(top, runner_up)
    switch_structure = _switch_structure_agreement(top, runner_up)
    top_conf = float(top.confidence or 0.0)
    runner_conf = float(runner_up.confidence or 0.0)
    segment_penalty = _segment_count_penalty(top.transcript, runner_up.transcript)
    return (
        agreement >= 0.88
        and timing >= 0.82
        and switch_structure >= 0.80
        and segment_penalty >= 0.80
        and abs(top_conf - runner_conf) <= 0.08
        and abs(float(top.timestamp_confidence or 0.0) - float(runner_up.timestamp_confidence or 0.0)) >= 0.10
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
            candidate_rationales={only.engine: ["single_candidate_available"]},
            disagreement_flags=[],
        )

    scored: List[Tuple[float, TranscriptCandidate, Dict[str, float], List[str]]] = []
    for candidate in candidates:
        score, metrics, rationale = score_candidate(candidate, candidates, audio_condition=audio_condition)
        scored.append((score, candidate, metrics, rationale))
    scored.sort(key=lambda item: item[0], reverse=True)

    top_score, top_candidate, top_metrics, top_rationale = scored[0]
    second_score, second_candidate, second_metrics, second_rationale = scored[1]
    candidate_rationales = {
        candidate.engine: rationale
        for _score, candidate, _metrics, rationale in scored
    }
    disagreement_flags = _disagreement_flags(top_candidate, second_candidate)

    if _merge_allowed(top_candidate, second_candidate):
        text_primary = top_candidate if float(top_candidate.confidence or 0.0) >= float(second_candidate.confidence or 0.0) else second_candidate
        timing_primary = top_candidate if float(top_candidate.timestamp_confidence or 0.0) >= float(second_candidate.timestamp_confidence or 0.0) else second_candidate
        merged_candidate = TranscriptCandidate(
            engine="merged_consensus",
            provider="consensus",
            paid_api=top_candidate.paid_api or second_candidate.paid_api,
            transcript=_merged_transcript(text_primary.transcript, timing_primary.transcript),
            confidence=max(float(top_candidate.confidence or 0.0), float(second_candidate.confidence or 0.0)),
            avg_word_confidence=max(
                float(top_candidate.avg_word_confidence or 0.0),
                float(second_candidate.avg_word_confidence or 0.0),
            ),
            language_hint=text_primary.language_hint or timing_primary.language_hint,
            detected_languages=list(dict.fromkeys([*top_candidate.detected_languages, *second_candidate.detected_languages])),
            code_switch_signals=text_primary.code_switch_signals or timing_primary.code_switch_signals,
            timing_source=timing_primary.timing_source,
            timestamp_confidence=max(
                float(top_candidate.timestamp_confidence or 0.0),
                float(second_candidate.timestamp_confidence or 0.0),
            ),
            warnings=list(dict.fromkeys([*top_candidate.warnings, *second_candidate.warnings])),
            adapter_metadata={
                "merged_from": [top_candidate.engine, second_candidate.engine],
                "token_agreement": token_agreement(top_candidate, second_candidate),
                "timing_agreement": _timing_agreement(top_candidate, second_candidate),
                "switch_structure_agreement": _switch_structure_agreement(top_candidate, second_candidate),
            },
            normalisation_notes=list(dict.fromkeys([*top_candidate.normalisation_notes, *second_candidate.normalisation_notes])),
        )
        return merged_candidate, ConsensusResult(
            transcript=merged_candidate.transcript,
            selected_engine=merged_candidate.engine,
            consensus_score=max(top_score, second_score),
            engines_compared=[candidate.engine for candidate in candidates],
            transcript_strategy="merged_consensus",
            review_recommended=max(top_score, second_score) < 0.78 or bool(disagreement_flags),
            rationale=[
                "merge_allowed_by_token_timing_switch_agreement",
                f"top_token_agreement={token_agreement(top_candidate, second_candidate):.4f}",
                f"top_timing_agreement={_timing_agreement(top_candidate, second_candidate):.4f}",
                f"top_switch_structure_agreement={_switch_structure_agreement(top_candidate, second_candidate):.4f}",
            ],
            candidate_rationales=candidate_rationales,
            disagreement_flags=disagreement_flags,
        )

    winner_rationale = list(top_rationale)
    if disagreement_flags:
        winner_rationale.append("disagreement=" + ",".join(disagreement_flags))
    return top_candidate, ConsensusResult(
        transcript=top_candidate.transcript,
        selected_engine=top_candidate.engine,
        consensus_score=top_score,
        engines_compared=[candidate.engine for candidate in candidates],
        transcript_strategy="best_single_engine",
        review_recommended=top_score < 0.78 or bool(disagreement_flags),
        rationale=winner_rationale + [
            f"runner_up_score={second_score:.4f}",
            f"runner_up_timing={second_metrics['timing']:.4f}",
            f"runner_up_switch_structure={second_metrics['switch_structure']:.4f}",
        ],
        candidate_rationales=candidate_rationales,
        disagreement_flags=disagreement_flags,
    )
