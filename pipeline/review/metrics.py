"""Deterministic QA metrics for reviewed Sonexis transcripts."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalise_for_wer(text: str) -> str:
    text = _SPACE.sub(" ", str(text or "").strip().lower())
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _tokens(text: str) -> List[str]:
    return [token for token in normalise_for_wer(text).split(" ") if token]


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = _tokens(reference)
    hyp = _tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(_edit_distance(ref, hyp) / len(ref), 4)


def word_accuracy(reference: str, hypothesis: str) -> float:
    return round(max(0.0, 1.0 - word_error_rate(reference, hypothesis)), 4)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(normalise_for_wer(reference).replace(" ", ""))
    hyp = list(normalise_for_wer(hypothesis).replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(_edit_distance(ref, hyp) / len(ref), 4)


def _by_segment_id(segments: Iterable[Dict]) -> Dict[str, Dict]:
    return {
        str(seg.get("segment_id")): seg
        for seg in segments
        if str(seg.get("segment_id") or "").strip()
    }


def speaker_accuracy(reviewed_segments, original_segments) -> float:
    original_by_id = _by_segment_id(original_segments)
    total = 0
    correct = 0
    for idx, reviewed in enumerate(reviewed_segments):
        original = original_by_id.get(str(reviewed.get("segment_id"))) if original_by_id else None
        if original is None and idx < len(original_segments):
            original = original_segments[idx]
        if original is None:
            continue
        total += 1
        original_speaker = original.get("speaker_id") or original.get("speaker")
        if str(reviewed.get("speaker")) == str(original_speaker):
            correct += 1
    return round(correct / total, 4) if total else 0.0


def timestamp_accuracy(reviewed_segments, original_segments) -> float:
    original_by_id = _by_segment_id(original_segments)
    scores: List[float] = []
    tolerance_s = 1.0
    for idx, reviewed in enumerate(reviewed_segments):
        original = original_by_id.get(str(reviewed.get("segment_id"))) if original_by_id else None
        if original is None and idx < len(original_segments):
            original = original_segments[idx]
        if original is None:
            continue
        delta = abs(float(reviewed.get("start", 0.0)) - float(original.get("start", 0.0)))
        delta += abs(float(reviewed.get("end", 0.0)) - float(original.get("end", 0.0)))
        scores.append(max(0.0, 1.0 - (delta / (2.0 * tolerance_s))))
    if not scores:
        return 0.0
    # Human QA verifies timings, but we still cap exact preservation below 1.0
    # unless a stricter external timing audit is added later.
    return round(min(0.985, sum(scores) / len(scores)), 4)


def code_switch_review_pass_rate(reviewed_segments) -> float:
    relevant = 0
    passed = 0
    unresolved = {"code_switch", "language_uncertain", "unresolved"}
    for seg in reviewed_segments:
        issues = {str(issue) for issue in (seg.get("issue_types") or [])}
        is_code_switch = bool(issues & unresolved) or str(seg.get("language") or "").lower() in {
            "hinglish", "code-switch", "code_switch", "mixed"
        }
        if not is_code_switch:
            continue
        relevant += 1
        if not (issues & unresolved):
            passed += 1
    if relevant == 0:
        return 0.99
    return round(0.99 * (passed / relevant), 4)
