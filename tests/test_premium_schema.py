from __future__ import annotations

import unittest

import numpy as np

from pipeline.diarisation import SpeakerTurn
from pipeline.language_detection import LanguageReport
from pipeline.metadata_extraction import extract_audio_metadata, extract_conversation_metadata, extract_speaker_metadata
from pipeline.output_formatter import build_record
from pipeline.steps.validation import validate_record_against_schema
from pipeline.transcription import Transcript, TranscriptSegment, Word


class PremiumSchemaTests(unittest.TestCase):
    def _transcript(self) -> Transcript:
        return Transcript(
            language="en",
            language_probability=0.9,
            duration=2.0,
            segments=[
                TranscriptSegment(
                    start=0.0,
                    end=2.0,
                    text="hello namaste",
                    language="en",
                    avg_logprob=-0.1,
                    quality_score=0.88,
                    words=[Word("hello", 0.0, 0.8, 0.9), Word("namaste", 0.9, 1.8, 0.9)],
                )
            ],
        )

    def test_offline_record_validates_with_new_defaults(self) -> None:
        transcript = self._transcript()
        turns = [SpeakerTurn(0.0, 2.0, "SPEAKER_00", 1.0)]
        audio_meta = extract_audio_metadata(np.zeros(32000, dtype=np.float32), 16000, [(0.0, 2.0)])
        speaker_meta = extract_speaker_metadata(
            transcript=transcript,
            turns=turns,
            language="en",
            scripts=["Latin"],
            filler_lexicon=set(),
            total_audio_duration_s=2.0,
        )
        conversation_meta = extract_conversation_metadata(transcript, turns)
        record = build_record(
            audio_path="/tmp/offline.wav",
            transcript=transcript,
            turns=turns,
            language=LanguageReport(primary_language="en", confidence=0.9),
            audio_meta=audio_meta,
            speaker_meta=speaker_meta,
            conversation_meta=conversation_meta,
            monologue=None,
            processing={"offline_mode": True, "random_seed": 0},
        )
        validate_record_against_schema(record)

    def test_premium_record_validates(self) -> None:
        transcript = self._transcript()
        turns = [SpeakerTurn(0.0, 2.0, "SPEAKER_00", 1.0)]
        audio_meta = extract_audio_metadata(np.zeros(32000, dtype=np.float32), 16000, [(0.0, 2.0)])
        speaker_meta = extract_speaker_metadata(
            transcript=transcript,
            turns=turns,
            language="en",
            scripts=["Latin"],
            filler_lexicon=set(),
            total_audio_duration_s=2.0,
        )
        conversation_meta = extract_conversation_metadata(transcript, turns)
        record = build_record(
            audio_path="/tmp/premium.wav",
            transcript=transcript,
            turns=turns,
            language=LanguageReport(primary_language="en", confidence=0.9),
            audio_meta=audio_meta,
            speaker_meta=speaker_meta,
            conversation_meta=conversation_meta,
            monologue=None,
            transcript_candidates=[
                {
                    "engine": "whisper_local",
                    "provider": "local",
                    "paid_api": False,
                    "transcript": {"raw": "hello namaste"},
                    "timing_source": "local_word_timestamps",
                }
            ],
            routing_decision={"pipeline_mode": "premium_accuracy"},
            timestamp_method="vendor_word_timestamps",
            timestamp_confidence=0.93,
            timestamp_refinement={"timestamp_method": "vendor_word_timestamps"},
            tts_suitability={"eligible": False, "reasons": ["review_not_final"], "confidence": 0.3},
            dataset_products=["stt", "evaluation_gold"],
            premium_processing={
                "pipeline_mode": "premium_accuracy",
                "paid_api_used": True,
                "engines_used": ["whisper_local", "deepgram"],
                "consensus_applied": True,
                "timestamp_refinement_applied": True,
                "human_review_required": True,
            },
        )
        validate_record_against_schema(record)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
