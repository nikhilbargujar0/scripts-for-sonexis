"""pipeline/steps/monologue.py — re-exports from pipeline.monologue_extractor."""
from pipeline.monologue_extractor import (  # noqa: F401
    Monologue,
    MonologueConfig,
    extract_audio_clip,
    extract_monologue,
    extract_monologues_per_speaker,
)

__all__ = [
    "Monologue",
    "MonologueConfig",
    "extract_audio_clip",
    "extract_monologue",
    "extract_monologues_per_speaker",
]
