"""Tests for validation report assembly."""
from pipeline.quality_checker import QualityReport
from pipeline.transcription import Transcript, TranscriptSegment
from pipeline.validation import build_validation_report


def test_validation_flags_diarisation_fallback():
    transcript = Transcript(
        language="en",
        language_probability=0.9,
        duration=1.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=1.0,
                text="hello",
                language="en",
                avg_logprob=-0.2,
                compression_ratio=1.4,
                no_speech_prob=0.1,
                rms_db=-20.0,
                quality_score=0.8,
                words=[],
            )
        ],
    )
    report = build_validation_report(
        quality_report=QualityReport(passed=True),
        transcript=transcript,
        requested_diarisation_backend="pyannote",
        effective_diarisation_backend="kmeans_fallback",
        speech_segments=[(0.0, 1.0)],
        turns=[],
    )
    codes = {issue["code"] for issue in report["issues"]}
    assert "diarisation_fallback" in codes
    assert "no_speaker_turns" in codes
