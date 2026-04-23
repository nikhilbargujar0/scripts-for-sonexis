from __future__ import annotations

import unittest

from pipeline.diarisation import SpeakerTurn
from pipeline.transcription import Transcript, TranscriptSegment
from pipeline.validation import build_validation_report


class ValidationReportTests(unittest.TestCase):
    def test_validation_flags_alignment_and_mix_risks(self) -> None:
        transcript = Transcript(
            language="en",
            language_probability=0.9,
            duration=120.0,
            segments=[
                TranscriptSegment(
                    start=0.0,
                    end=1.0,
                    text="hello",
                    language="en",
                    avg_logprob=-0.1,
                    quality_score=0.9,
                )
            ],
        )
        turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00", 1.0)]

        report = build_validation_report(
            transcript=transcript,
            speech_segments=[(0.0, 1.0)],
            turns=turns,
            input_alignment={
                "method": "cross_correlation",
                "offset_ms": 123,
                "confidence": 0.31,
                "passed": False,
                "applied": False,
            },
            mono_mix={
                "normalised": True,
                "peak_before": 1.24,
                "peak_after": 0.92,
                "clipping_prevented": True,
            },
            interaction_meta={"overlap_duration": 0.0},
            expected_overlap_duration_s=0.8,
            session_duration_s=120.0,
            alignment_required=True,
        )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("alignment_not_applied", codes)
        self.assertIn("low_alignment_confidence", codes)
        self.assertIn("mono_mix_clipping", codes)
        self.assertIn("missing_expected_overlap", codes)
        self.assertIn("suspiciously_low_turn_count", codes)
        self.assertFalse(report["checks"]["alignment_applied"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
