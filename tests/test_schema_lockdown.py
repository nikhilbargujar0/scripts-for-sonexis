from __future__ import annotations

import copy
import unittest

import numpy as np
from jsonschema.exceptions import ValidationError

from pipeline.diarisation import SpeakerTurn
from pipeline.language_detection import LanguageReport, LanguageSegment
from pipeline.metadata_extraction import (
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)
from pipeline.output_formatter import DATASET_SCHEMA_VERSION, SCHEMA_VERSION, build_record
from pipeline.schema_validator import SchemaVersionError, validate_record
from pipeline.transcription import Transcript, TranscriptSegment, Word, _text_normalized


def _make_transcript() -> Transcript:
    return Transcript(
        language="hi",
        language_probability=0.92,
        duration=3.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=3.0,
                text="नमस्ते, कैसे हो?",
                language="hi",
                avg_logprob=-0.15,
                quality_score=0.88,
                words=[
                    Word("नमस्ते", 0.0, 1.0, 0.9),
                    Word("कैसे", 1.1, 2.0, 0.88),
                    Word("हो", 2.1, 3.0, 0.85),
                ],
            )
        ],
    )


def _make_record(**kwargs) -> dict:
    transcript = _make_transcript()
    turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00", 1.0)]
    wav = np.zeros(48000, dtype=np.float32)
    audio_meta = extract_audio_metadata(wav, 16000, [(0.0, 3.0)])
    speaker_meta = extract_speaker_metadata(
        transcript=transcript,
        turns=turns,
        language="hi",
        scripts=["Devanagari"],
        filler_lexicon=set(),
        total_audio_duration_s=3.0,
    )
    conversation_meta = extract_conversation_metadata(transcript, turns)
    language = LanguageReport(
        primary_language="hi",
        confidence=0.92,
        dominant_language="hi",
        language_segments=[LanguageSegment(0.0, 3.0, "hi", 0.92)],
    )
    return build_record(
        audio_path="/tmp/test_session.wav",
        transcript=transcript,
        turns=turns,
        language=language,
        audio_meta=audio_meta,
        speaker_meta=speaker_meta,
        conversation_meta=conversation_meta,
        monologue=None,
        session_name="test_session",
        model_versions={"whisper": "small", "diarisation": "kmeans", "vad": "auto"},
        config_hash="a" * 64,
        total_speech_duration_sec=3.0,
        **kwargs,
    )


class TestSegmentMandatoryFields(unittest.TestCase):
    def test_segment_mandatory_fields(self):
        segment = _make_record()["transcript"]["segments"][0]
        for field in (
            "segment_id",
            "audio_filepath",
            "start",
            "end",
            "duration",
            "speaker_id",
            "language",
            "text",
            "text_normalized",
            "sample_rate",
            "missing_fields",
        ):
            self.assertIn(field, segment)
        self.assertEqual(segment["segment_id"], "test_session_0000")
        self.assertEqual(segment["language"], "hi-Deva")
        self.assertEqual(segment["speaker_id"], "SPEAKER_00")


class TestSessionMandatoryFields(unittest.TestCase):
    def test_session_mandatory_fields(self):
        record = _make_record()
        for field in (
            "session_id",
            "pipeline_version",
            "model_versions",
            "config_hash",
            "schema_version",
            "num_speakers",
            "total_duration_sec",
            "total_speech_duration_sec",
            "dominant_language",
            "language_distribution",
        ):
            self.assertIn(field, record)
        self.assertEqual(record["session_id"], "test_session")
        self.assertEqual(record["dominant_language"], "hi-Deva")
        self.assertEqual(record["language_distribution"], {"hi-Deva": 1})


class TestTextNormalized(unittest.TestCase):
    def test_text_normalized(self):
        self.assertEqual(_text_normalized("Hello, World!"), "hello world")
        self.assertEqual(_text_normalized("नमस्ते, कैसे हो?"), "नमस्ते कैसे हो")
        self.assertEqual(_text_normalized("[Music] hello"), "music hello")
        segment = _make_record()["transcript"]["segments"][0]
        self.assertEqual(segment["text_normalized"], "नमस्ते कैसे हो")


class TestMissingFieldsList(unittest.TestCase):
    def test_missing_fields_list(self):
        segment = _make_record()["transcript"]["segments"][0]
        self.assertEqual(segment["missing_fields"], ["punctuation", "snr_db", "confidence"])
        self.assertTrue(all(isinstance(item, str) for item in segment["missing_fields"]))


class TestSchemaVersion(unittest.TestCase):
    def test_schema_version_and_mismatch(self):
        record = _make_record()
        self.assertEqual(DATASET_SCHEMA_VERSION, "1.0")
        self.assertEqual(SCHEMA_VERSION, "3.0.0")
        self.assertEqual(record["schema_version"], "1.0")
        self.assertEqual(record["pipeline_record_version"], "3.0.0")

        bad = copy.deepcopy(record)
        bad["schema_version"] = "99.0"
        with self.assertRaises(SchemaVersionError) as ctx:
            validate_record(bad)
        self.assertIn("99.0", str(ctx.exception))
        self.assertIn("1.0", str(ctx.exception))


class TestSchemaValidationGate(unittest.TestCase):
    def test_valid_record_passes_gate(self):
        validate_record(_make_record())

    def test_missing_required_field_raises(self):
        bad = copy.deepcopy(_make_record())
        del bad["session_id"]
        with self.assertRaises(ValidationError):
            validate_record(bad)

    def test_invalid_segment_field_type_raises(self):
        bad = copy.deepcopy(_make_record())
        bad["transcript"]["segments"][0]["sample_rate"] = "sixteen_k"
        with self.assertRaises(ValidationError):
            validate_record(bad)

    def test_empty_segment_id_raises(self):
        bad = copy.deepcopy(_make_record())
        bad["transcript"]["segments"][0]["segment_id"] = ""
        with self.assertRaises(ValidationError):
            validate_record(bad)


if __name__ == "__main__":
    unittest.main()
