from __future__ import annotations

import unittest

import numpy as np

from pipeline.diarisation import SpeakerTurn
from pipeline.language_detection import LanguageReport
from pipeline.metadata_extraction import (
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)
from pipeline.output_formatter import build_record
from pipeline.processors.downstream import apply_user_metadata
from pipeline.steps.validation import validate_record_against_schema
from pipeline.transcription import Transcript, TranscriptSegment, Word


class DatasetMetadataExtensionTests(unittest.TestCase):
    def _transcript(self) -> Transcript:
        return Transcript(
            language="en",
            language_probability=0.91,
            duration=3.0,
            segments=[
                TranscriptSegment(
                    start=0.0,
                    end=1.2,
                    text="hello I need help with payment",
                    language="en",
                    avg_logprob=-0.1,
                    quality_score=0.88,
                    words=[
                        Word("hello", 0.0, 0.2),
                        Word("I", 0.25, 0.3),
                        Word("need", 0.31, 0.45),
                        Word("help", 0.46, 0.7),
                        Word("with", 0.71, 0.82),
                        Word("payment", 0.83, 1.1),
                    ],
                ),
                TranscriptSegment(
                    start=1.3,
                    end=2.1,
                    text="sure I can check that",
                    language="en",
                    avg_logprob=-0.15,
                    quality_score=0.81,
                    words=[
                        Word("sure", 1.3, 1.45),
                        Word("I", 1.5, 1.55),
                        Word("can", 1.56, 1.68),
                        Word("check", 1.69, 1.87),
                        Word("that", 1.88, 2.05),
                    ],
                ),
            ],
        )

    def test_audio_metadata_includes_extended_fields(self) -> None:
        sr = 16000
        t = np.linspace(0.0, 1.0, sr, endpoint=False)
        wav = 0.2 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        audio = extract_audio_metadata(
            wav,
            sr,
            [(0.0, 1.0)],
            source_info={
                "sample_rate_hz": 48000,
                "processed_sample_rate_hz": 16000,
                "bit_depth": 24,
                "channels": 1,
                "codec": "pcm_24",
                "container_format": "wav",
            },
        )
        self.assertEqual(audio["sample_rate_hz"], 48000)
        self.assertEqual(audio["processed_sample_rate_hz"], 16000)
        self.assertIn("effective_bandwidth_hz", audio)
        self.assertEqual(audio["effective_bandwidth_hz"]["source"], "measured")
        self.assertIn("lufs", audio)
        self.assertIn("method", audio["lufs"])

    def test_speaker_metadata_uses_wrapped_accent_region_dialect(self) -> None:
        transcript = self._transcript()
        turns = [
            SpeakerTurn(0.0, 1.2, "SPEAKER_00", 1.0),
            SpeakerTurn(1.3, 2.1, "SPEAKER_01", 1.0),
        ]
        speaker_meta = extract_speaker_metadata(
            transcript=transcript,
            turns=turns,
            language="en",
            scripts=["Latin"],
            filler_lexicon=set(),
            speaker_labels={"SPEAKER_00": "Agent", "SPEAKER_01": "Caller"},
            total_audio_duration_s=3.0,
        )
        speaker_meta = apply_user_metadata(
            speaker_meta,
            {
                "*": {
                    "region": {"value": "Rajasthan", "confidence": 1.0, "source": "user_provided"},
                    "dialect": {"value": "Marwadi", "confidence": 1.0, "source": "user_provided"},
                }
            },
            speaker_map={"SPEAKER_00": "Agent", "SPEAKER_01": "Caller"},
        )
        self.assertEqual(speaker_meta["SPEAKER_00"]["speaker_id"], "SPEAKER_00")
        self.assertEqual(speaker_meta["SPEAKER_00"]["accent"]["source"], "inferred")
        self.assertEqual(speaker_meta["SPEAKER_00"]["region"]["source"], "user_provided")
        self.assertEqual(speaker_meta["SPEAKER_00"]["dialect"]["value"], "Marwadi")

    def test_record_gets_purpose_quality_and_schema_validates(self) -> None:
        transcript = self._transcript()
        turns = [
            SpeakerTurn(0.0, 1.2, "SPEAKER_00", 1.0),
            SpeakerTurn(1.0, 2.1, "SPEAKER_01", 1.0),
        ]
        audio_meta = extract_audio_metadata(
            np.zeros(32000, dtype=np.float32),
            16000,
            [(0.0, 1.2), (1.0, 2.1)],
            source_info={
                "sample_rate_hz": 48000,
                "processed_sample_rate_hz": 16000,
                "bit_depth": 16,
                "channels": 2,
                "codec": "pcm_16",
                "container_format": "wav",
            },
        )
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
            audio_path="/tmp/example.wav",
            transcript=transcript,
            turns=turns,
            language=LanguageReport(primary_language="en", confidence=0.9),
            audio_meta=audio_meta,
            speaker_meta=speaker_meta,
            conversation_meta=conversation_meta,
            monologue=None,
            interaction_meta={"overlap_duration": 0.2, "turn_count": 2},
            monologues={},
            processing={"offline_mode": True, "random_seed": 0},
        )

        self.assertIn("stt", record["dataset_purpose"]["primary"])
        self.assertIn("diarisation", record["dataset_purpose"]["primary"])
        self.assertIn("tts", record["dataset_purpose"]["not_recommended_for"])
        self.assertEqual(record["quality_targets"]["word_accuracy_target"], 0.98)
        self.assertIsNone(record["quality_metrics"]["estimated_word_accuracy"])
        self.assertIn("human_review", record)
        self.assertIn("code_switch", record)
        self.assertIn("premium_processing", record)
        validate_record_against_schema(record)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
