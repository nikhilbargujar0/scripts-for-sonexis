"""Monologue extraction step wrapper."""
from __future__ import annotations

from typing import Dict, List, Optional

from ..diarisation import SpeakerTurn
from ..monologue_extractor import Monologue, MonologueConfig, extract_monologue, extract_monologues_per_speaker
from ..transcription import Transcript


def extract_best_monologues(
    transcript: Transcript,
    turns: List[SpeakerTurn],
    cfg: Optional[MonologueConfig] = None,
) -> Dict[str, Optional[Monologue]]:
    """Score candidates by duration, speech density, silence, and overlap."""
    return extract_monologues_per_speaker(transcript, turns, cfg or MonologueConfig())
