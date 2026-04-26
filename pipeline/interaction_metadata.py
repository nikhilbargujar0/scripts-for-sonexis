"""interaction_metadata.py

Dual-speaker interaction metrics.  This is the layer most pipelines skip.

Real conversational data has:
  - overlapping speech  (two speakers simultaneously active)
  - interruptions       (short overlap at a turn boundary < threshold)
  - response latency    (gap between one speaker's turn ending and the other's starting)
  - turn dominance      (who speaks more, who initiates more)
  - engagement          (heuristic: fast responses + frequent switching + some interruptions)

All computations are deterministic: same turns → same output.
No randomness. No ML calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .diarisation import SpeakerTurn


# ── overlap detection ──────────────────────────────────────────────────────

@dataclass
class OverlapSegment:
    start: float
    end: float
    speaker_a: str
    speaker_b: str
    is_interruption: bool  # True when overlap < interruption_threshold_s

    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        overlap_type = "backchannel" if self.is_interruption else "crosstalk"
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker_a": self.speaker_a,
            "speaker_b": self.speaker_b,
            "overlap": True,
            "overlap_type": overlap_type,
            "overlap_speakers": [self.speaker_a, self.speaker_b],
            "overlap_start": round(self.start, 3),
            "overlap_end": round(self.end, 3),
            "duration_s": round(self.duration(), 3),
            "is_interruption": self.is_interruption,
        }


def _find_overlaps(
    turns: List[SpeakerTurn],
    interruption_threshold_s: float = 0.5,
    min_overlap_s: float = 0.05,
) -> List[OverlapSegment]:
    """Return all intervals where two different speakers are simultaneously active.

    Turns must be sorted by start time (they are after _merge_turns).
    We use a simple sweep: for each turn A, scan forward until we find
    turns that start after A ends.
    """
    if not turns:
        return []
    sorted_turns = sorted(turns, key=lambda t: t.start)
    overlaps: List[OverlapSegment] = []

    for i, a in enumerate(sorted_turns):
        for b in sorted_turns[i + 1:]:
            if b.start >= a.end:
                break  # sorted → no more overlaps possible with a
            if a.speaker == b.speaker:
                continue
            overlap_start = max(a.start, b.start)
            overlap_end = min(a.end, b.end)
            dur = overlap_end - overlap_start
            if dur >= min_overlap_s:
                overlaps.append(OverlapSegment(
                    start=overlap_start,
                    end=overlap_end,
                    speaker_a=a.speaker,
                    speaker_b=b.speaker,
                    is_interruption=dur < interruption_threshold_s,
                ))
    return overlaps


# ── response latency ───────────────────────────────────────────────────────

def _compute_response_latencies(turns: List[SpeakerTurn]) -> List[float]:
    """Gap between end of one turn and start of the next turn by a different speaker.

    Negative gaps (overlaps) are excluded — they are captured separately.
    Returns a list of non-negative floats in seconds.
    """
    latencies: List[float] = []
    for a, b in zip(turns, turns[1:]):
        if a.speaker != b.speaker:
            gap = b.start - a.end
            if gap >= 0.0:
                latencies.append(float(gap))
    return latencies


# ── engagement score ───────────────────────────────────────────────────────

def _engagement_score(
    avg_latency_s: float,
    switch_freq_per_min: float,
    interruption_count: int,
    total_turns: int,
) -> float:
    """Heuristic engagement in [0, 1].

    High engagement = fast response + frequent turn-switching + mild interruptions.

    Latency:    0s→1.0,  3s→0.5,  ≥10s→~0.05  (exponential decay)
    Switch freq: ≥20/min→1.0,  5/min→0.25,  0→0.0
    Interruption ratio: 0-5% turns→0.5 (baseline),
                        5-20%→ramps to 1.0 (animated),
                        >30%→drops (chaotic)
    """
    # latency component
    lat_score = math.exp(-avg_latency_s / 3.0)

    # switch frequency component
    switch_score = float(np.clip(switch_freq_per_min / 20.0, 0.0, 1.0))

    # interruption component
    iratio = interruption_count / max(total_turns, 1)
    if iratio < 0.05:
        int_score = 0.5
    elif iratio < 0.20:
        int_score = 0.5 + 0.5 * (iratio - 0.05) / 0.15
    else:
        int_score = max(0.25, 1.0 - iratio * 2)

    # geometric mean of three components (all components must be non-zero to score high)
    product = lat_score * switch_score * int_score
    return float(np.clip(product ** (1.0 / 3.0), 0.0, 1.0))


# ── per-speaker overlap ratio ──────────────────────────────────────────────

def _overlap_ratio_per_speaker(
    overlaps: List[OverlapSegment],
    speaker_speaking_time: Dict[str, float],
) -> Dict[str, float]:
    """Fraction of each speaker's speaking time spent in overlapping speech."""
    spk_overlap: Dict[str, float] = {}
    for o in overlaps:
        dur = o.duration()
        spk_overlap[o.speaker_a] = spk_overlap.get(o.speaker_a, 0.0) + dur
        spk_overlap[o.speaker_b] = spk_overlap.get(o.speaker_b, 0.0) + dur

    return {
        spk: round(spk_overlap.get(spk, 0.0) / max(t, 1e-6), 4)
        for spk, t in speaker_speaking_time.items()
    }


# ── main extraction function ───────────────────────────────────────────────

def extract_interaction_metadata(
    turns: List[SpeakerTurn],
    interruption_threshold_s: float = 0.5,
) -> Tuple[Dict, Dict[str, float], List[OverlapSegment]]:
    """Compute interaction metrics for a dual-speaker conversation.

    Returns:
        (metadata_dict, overlap_ratio_per_speaker, overlap_segments)

    overlap_ratio_per_speaker is needed by extract_speaker_metadata to add
    overlap_ratio per speaker without a second pass.
    """
    _empty = {
        "interruption_count": 0,
        "overlap_duration_s": 0.0,
        "overlap_segments": 0,
        "response_latency": {"mean_s": None, "median_s": None, "min_s": None},
        "turn_switch_frequency": 0.0,
        "dominance": {},
        "dominant_speaker": None,
        "engagement_score": 0.0,
        "total_turns": 0,
        "conversation_duration_s": 0.0,
    }
    if not turns:
        return _empty, {}, []

    sorted_turns = sorted(turns, key=lambda t: t.start)
    speakers = sorted(set(t.speaker for t in sorted_turns))

    # Speaking time per speaker
    spk_time: Dict[str, float] = {
        s: sum(t.duration() for t in sorted_turns if t.speaker == s)
        for s in speakers
    }
    total_spoken = sum(spk_time.values()) or 1.0

    # Overlaps + interruptions
    overlaps = _find_overlaps(sorted_turns, interruption_threshold_s)
    total_overlap_dur = sum(o.duration() for o in overlaps)
    interruptions = [o for o in overlaps if o.is_interruption]

    # Response latency
    latencies = _compute_response_latencies(sorted_turns)

    # Turn switches
    switches = sum(
        1 for a, b in zip(sorted_turns, sorted_turns[1:])
        if a.speaker != b.speaker
    )
    conv_dur = max(sorted_turns[-1].end - sorted_turns[0].start, 1e-6)
    switch_freq = switches / (conv_dur / 60.0)

    # Dominance
    dominance = {s: round(spk_time[s] / total_spoken, 4) for s in speakers}
    dominant_speaker = max(dominance, key=dominance.get) if dominance else None

    # Engagement
    avg_lat = float(np.mean(latencies)) if latencies else 5.0
    engagement = _engagement_score(avg_lat, switch_freq, len(interruptions), len(sorted_turns))

    # Per-speaker overlap ratio (passed out separately so metadata_extraction can use it)
    overlap_ratios = _overlap_ratio_per_speaker(overlaps, spk_time)

    meta = {
        "interruption_count": len(interruptions),
        "overlap_duration_s": round(total_overlap_dur, 3),
        "overlap_segments": len(overlaps),
        "response_latency": {
            "mean_s": round(float(np.mean(latencies)), 3) if latencies else None,
            "median_s": round(float(np.median(latencies)), 3) if latencies else None,
            "min_s": round(float(np.min(latencies)), 3) if latencies else None,
            "max_s": round(float(np.max(latencies)), 3) if latencies else None,
        },
        "turn_switch_frequency": round(switch_freq, 4),
        "dominance": dominance,
        "dominant_speaker": dominant_speaker,
        "engagement_score": round(engagement, 4),
        "total_turns": len(sorted_turns),
        "conversation_duration_s": round(conv_dur, 3),
        "speaking_time_per_speaker": {s: round(t, 3) for s, t in spk_time.items()},
    }
    return meta, overlap_ratios, overlaps
