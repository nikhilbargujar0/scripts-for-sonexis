"""pipeline/steps/language.py

Re-exports from pipeline.language_detection.

Includes ``switching_pattern`` (hi→en, en→hi transition counts) which was
added to :class:`LanguageReport` as part of this refactor.
"""
from pipeline.language_detection import (  # noqa: F401
    FastTextLID,
    LanguageReport,
    LanguageSegment,
    _compute_switching_pattern,
    detect_language,
    detect_language_per_speaker,
)

__all__ = [
    "FastTextLID",
    "LanguageReport",
    "LanguageSegment",
    "detect_language",
    "detect_language_per_speaker",
    # Exposed for direct use when only the pattern (not full report) is needed.
    "switching_pattern_from_segments",
]


def switching_pattern_from_segments(language_segments) -> dict:
    """Return ``{lang_a→lang_b: count}`` transition dict from a segment list.

    Convenience wrapper around the internal
    :func:`pipeline.language_detection._compute_switching_pattern`.
    """
    return _compute_switching_pattern(language_segments)
