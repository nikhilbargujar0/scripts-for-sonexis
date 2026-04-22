"""pipeline/steps/interaction.py — re-exports from pipeline.interaction_metadata."""
from pipeline.interaction_metadata import (  # noqa: F401
    OverlapSegment,
    extract_interaction_metadata,
)

__all__ = [
    "OverlapSegment",
    "extract_interaction_metadata",
]
