"""monologue_extractor.py  # noqa: E501

Locate the best single-speaker monologue between 10 and 30 seconds long.

Scoring replaces the old naive "pick longest" approach. Each candidate span
is scored on three axes:

  1. duration_score  - prefer 15-25s; penalise below 10s or above 30s
  2. continuity_score - fraction of span time that is speech (low silence ratio)
  3. interruption_score - 1.0 if no other speaker overlaps; partial credit otherwise

Final score = duration_score * continuity_score * interruption_score

The best-scoring in-range candidate wins. If no candidate is in range,
we return the highest-scoring partial match (trimmed to max_duration_s if needed)
and flag in_range=False.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .diarisation import SpeakerTurn
from .transcription import Transcript, Word


@dataclass
class MonologueConfig:
    min_duration_s: float = 10.0
    max_duration_s: float = 30.0
    max_internal_gap_s: float = 1.5
    # Ideal window for scoring (centred in allowed range).
    ideal_min_s: float = 15.0
    ideal_max_s: float = 25.0


@dataclass
class Monologue:
    speaker: str
    start: float
    end: float
    transcript: str
    words: List[dict]
    in_range: bool
    quality_score: float = 0.0      # composite 0-1 score
    silence_ratio: float = 0.0      # fraction of span that is silence
    interruption_free: bool = True  # no other speaker overlaps

    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration_s": round(self.duration(), 3),
            "transcript": self.transcript,
            "words": self.words,
            "in_range": self.in_range,
            "quality_score": round(float(self.quality_score), 4),
            "silence_ratio": round(float(self.silence_ratio), 4),
            "interruption_free": bool(self.interruption_free),
        }


def _collect_words_in_span(transcript: Transcript, start: float, end: float) -> List[Word]:
    out: List[Word] = []
    for seg in transcript.segments:
        if seg.end < start or seg.start > end:
            continue
        for w in seg.words or []:
            mid = (w.start + w.end) / 2.0
            if start - 1e-3 <= mid <= end + 1e-3:
                out.append(w)
    return out


def _duration_score(duration_s: float, cfg: MonologueConfig) -> float:
    """Score 1.0 in ideal window, tapers outside."""
    if duration_s < cfg.min_duration_s:
        return duration_s / cfg.min_duration_s * 0.5
    if duration_s > cfg.max_duration_s:
        # Over-long spans get trimmed later; give partial credit.
        return 0.7
    if cfg.ideal_min_s <= duration_s <= cfg.ideal_max_s:
        return 1.0
    # Linear ramp from min→ideal_min and ideal_max→max.
    if duration_s < cfg.ideal_min_s:
        return 0.7 + 0.3 * (duration_s - cfg.min_duration_s) / max(
            1e-3, cfg.ideal_min_s - cfg.min_duration_s
        )
    return 0.7 + 0.3 * (cfg.max_duration_s - duration_s) / max(
        1e-3, cfg.max_duration_s - cfg.ideal_max_s
    )


def _silence_ratio(words: List[Word], span_duration: float) -> float:
    """Fraction of span time not covered by word timestamps."""
    if not words or span_duration <= 0:
        return 1.0
    speech_time = sum(max(0.0, w.end - w.start) for w in words)
    return max(0.0, 1.0 - speech_time / span_duration)


def _interruption_free(
    span_start: float,
    span_end: float,
    speaker: str,
    all_turns: List[SpeakerTurn],
) -> bool:
    """True if no other speaker has a turn overlapping this span."""
    for t in all_turns:
        if t.speaker == speaker:
            continue
        if t.end <= span_start or t.start >= span_end:
            continue
        return False
    return True


def _score_candidate(
    m: "Monologue",
    cfg: MonologueConfig,
    all_turns: List[SpeakerTurn],
) -> float:
    dur = m.duration()
    d_score = _duration_score(dur, cfg)
    # Silence ratio → continuity score (lower silence = better).
    continuity = max(0.0, 1.0 - m.silence_ratio * 1.5)
    # Interruption score: full credit if no overlap, 0.5 otherwise.
    interruption = 1.0 if m.interruption_free else 0.5
    return float(d_score * continuity * interruption)


def extract_monologue(
    transcript: Transcript,
    turns: List[SpeakerTurn],
    cfg: MonologueConfig | None = None,
) -> Optional[Monologue]:
    cfg = cfg or MonologueConfig()

    if not turns:
        turns = [SpeakerTurn(0.0, transcript.duration, "SPEAKER_00")]

    # Step 1 - split each turn into contiguous sub-spans at silence gaps.
    candidates: List[Monologue] = []
    for turn in turns:
        words = _collect_words_in_span(transcript, turn.start, turn.end)
        if not words:
            continue
        words = sorted(words, key=lambda w: w.start)
        span_start = words[0].start
        prev_end = words[0].end
        buffer: List[Word] = [words[0]]

        def flush(buf: List[Word], s_start: float, s_end: float) -> None:
            if not buf:
                return
            text = " ".join(w.text.strip() for w in buf).strip()
            if not text:
                return
            dur = s_end - s_start
            in_range = cfg.min_duration_s <= dur <= cfg.max_duration_s
            sil_ratio = _silence_ratio(buf, dur)
            no_interrupt = _interruption_free(s_start, s_end, turn.speaker, turns)
            candidates.append(Monologue(
                speaker=turn.speaker,
                start=s_start,
                end=s_end,
                transcript=text,
                words=[w.to_dict() for w in buf],
                in_range=in_range,
                quality_score=0.0,  # filled below
                silence_ratio=sil_ratio,
                interruption_free=no_interrupt,
            ))

        for w in words[1:]:
            gap = w.start - prev_end
            if gap > cfg.max_internal_gap_s:
                flush(buffer, span_start, prev_end)
                span_start = w.start
                buffer = [w]
            else:
                buffer.append(w)
            prev_end = w.end
        flush(buffer, span_start, prev_end)

    if not candidates:
        return None

    # Step 2 - score all candidates.
    for m in candidates:
        m.quality_score = _score_candidate(m, cfg, turns)

    # Step 3 - prefer in-range candidate with highest score.
    in_range = [c for c in candidates if c.in_range]
    if in_range:
        best = max(in_range, key=lambda c: c.quality_score)
        return best

    # Step 4 - no in-range candidate; trim the best long one.
    best = max(candidates, key=lambda c: c.quality_score)
    if best.duration() > cfg.max_duration_s:
        trimmed_words = []
        cutoff = best.start + cfg.max_duration_s
        for wd in best.words:
            if wd["start"] <= cutoff:
                trimmed_words.append(wd)
            else:
                break
        if trimmed_words:
            text = " ".join(w["text"].strip() for w in trimmed_words).strip()
            end = trimmed_words[-1]["end"]
            dur = end - best.start
            return Monologue(
                speaker=best.speaker,
                start=best.start,
                end=end,
                transcript=text,
                words=trimmed_words,
                in_range=cfg.min_duration_s <= dur <= cfg.max_duration_s,
                quality_score=best.quality_score,
                silence_ratio=best.silence_ratio,
                interruption_free=best.interruption_free,
            )
    return best


# ── per-speaker helpers ──────────────────────────────────────────────────────

def extract_monologues_per_speaker(
    transcript: Transcript,
    turns: List[SpeakerTurn],
    cfg: "MonologueConfig | None" = None,
) -> Dict[str, Optional[Monologue]]:
    """Extract best monologue for each speaker independently.

    Filters turns per speaker so the interruption check only considers
    other speakers' turns.  Returns {speaker_id: Monologue | None}.
    """
    cfg = cfg or MonologueConfig()
    speakers = sorted(set(t.speaker for t in turns))
    result: Dict[str, Optional[Monologue]] = {}
    for spk in speakers:
        spk_turns = [t for t in turns if t.speaker == spk]
        result[spk] = extract_monologue(transcript, spk_turns, cfg)
    return result


def extract_audio_clip(
    wav: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
) -> np.ndarray:
    """Slice [start, end] seconds from wav. Returns empty array if out of bounds."""
    if wav.size == 0 or end <= start:
        return np.array([], dtype=wav.dtype)
    start_sample = max(0, int(round(start * sample_rate)))
    end_sample = min(wav.size, int(round(end * sample_rate)))
    return wav[start_sample:end_sample].copy()
