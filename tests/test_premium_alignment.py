from __future__ import annotations

import unittest

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.premium.alignment_router import refine_timestamps
from pipeline.premium.types import TranscriptCandidate
from pipeline.transcription import Transcript, TranscriptSegment


class PremiumAlignmentTests(unittest.TestCase):
    def test_synthetic_fallback_is_marked_low_quality(self) -> None:
        transcript = Transcript(
            language="en",
            language_probability=0.8,
            duration=2.0,
            segments=[
                TranscriptSegment(
                    start=0.0,
                    end=2.0,
                    text="hello namaste",
                    language="en",
                    avg_logprob=-0.1,
                    quality_score=0.6,
                    words=[],
                )
            ],
        )
        candidate = TranscriptCandidate(
            engine="whisper_local",
            transcript=transcript,
            timestamp_confidence=0.5,
            timing_source="local_word_timestamps",
        )
        result = refine_timestamps(
            candidate,
            wav=np.zeros(32000, dtype=np.float32),
            sample_rate=16000,
            cfg=PipelineConfig(),
        )

        self.assertTrue(result.synthetic_word_timestamps)
        self.assertEqual(result.timing_quality, "low")
        self.assertLessEqual(result.timestamp_confidence, 0.35)
        self.assertIn("synthetic_word_timestamps_generated_by_even_word_split", result.notes)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
