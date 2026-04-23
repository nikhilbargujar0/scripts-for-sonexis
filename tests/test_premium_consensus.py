from __future__ import annotations

import unittest

from pipeline.premium.consensus import choose_consensus
from pipeline.premium.types import TranscriptCandidate
from pipeline.transcription import Transcript, TranscriptSegment, Word


def _candidate(
    engine: str,
    text: str,
    *,
    confidence: float,
    timestamp_confidence: float,
) -> TranscriptCandidate:
    words = text.split()
    transcript = Transcript(
        language="en",
        language_probability=confidence,
        duration=2.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text=text,
                language="en",
                avg_logprob=-0.1,
                quality_score=confidence,
                words=[
                    Word(token, idx * 0.5, (idx + 1) * 0.5, confidence)
                    for idx, token in enumerate(words)
                ],
            )
        ],
    )
    return TranscriptCandidate(
        engine=engine,
        provider=engine,
        paid_api=engine != "whisper_local",
        transcript=transcript,
        confidence=confidence,
        avg_word_confidence=confidence,
        language_hint="en",
        detected_languages=["English"],
        code_switch_signals={"detected": False, "dominant_languages": ["English"], "switch_count": 0, "switch_patterns": []},
        timing_source="vendor_word_timestamps" if engine != "whisper_local" else "local_word_timestamps",
        timestamp_confidence=timestamp_confidence,
    )


class PremiumConsensusTests(unittest.TestCase):
    def test_best_candidate_is_selected(self) -> None:
        winner, result = choose_consensus(
            [
                _candidate("whisper_local", "hello there", confidence=0.52, timestamp_confidence=0.45),
                _candidate("deepgram", "hello there", confidence=0.92, timestamp_confidence=0.96),
                _candidate("google_stt_v2", "yellow there", confidence=0.61, timestamp_confidence=0.9),
            ]
        )

        self.assertEqual(winner.engine, "deepgram")
        self.assertEqual(result.transcript_strategy, "best_single_engine")
        self.assertIn("deepgram", result.candidate_rationales)

    def test_merged_consensus_path_prefers_better_timestamps(self) -> None:
        winner, result = choose_consensus(
            [
                _candidate("deepgram", "hello namaste", confidence=0.88, timestamp_confidence=0.72),
                _candidate("google_stt_v2", "hello namaste", confidence=0.85, timestamp_confidence=0.96),
            ]
        )

        self.assertEqual(winner.engine, "merged_consensus")
        self.assertEqual(result.transcript_strategy, "merged_consensus")
        self.assertEqual(winner.timing_source, "vendor_word_timestamps")
        self.assertIn("merge_allowed_by_token_timing_switch_agreement", result.rationale[0])

    def test_code_switch_disagreement_is_preserved_in_rationale(self) -> None:
        first = _candidate("deepgram", "hello namaste", confidence=0.86, timestamp_confidence=0.9)
        first.code_switch_signals = {
            "detected": True,
            "dominant_languages": ["English", "Hindi"],
            "switch_count": 3,
            "switch_patterns": ["en->hi", "hi->en"],
            "switching_score": 0.7,
        }
        second = _candidate("google_stt_v2", "hello namaste", confidence=0.84, timestamp_confidence=0.91)
        second.code_switch_signals = {
            "detected": False,
            "dominant_languages": ["English"],
            "switch_count": 0,
            "switch_patterns": [],
            "switching_score": 0.0,
        }

        winner, result = choose_consensus([first, second])

        self.assertEqual(winner.engine, "deepgram")
        self.assertIn("strong_code_switch_disagreement", result.disagreement_flags)
        self.assertTrue(any("code_switch_preservation" in item for item in result.candidate_rationales["deepgram"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
