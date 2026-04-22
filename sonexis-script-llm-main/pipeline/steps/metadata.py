"""pipeline/steps/metadata.py — re-exports from pipeline.metadata_extraction."""
from pipeline.metadata_extraction import (  # noqa: F401
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)

__all__ = [
    "extract_audio_metadata",
    "extract_conversation_metadata",
    "extract_speaker_metadata",
]
