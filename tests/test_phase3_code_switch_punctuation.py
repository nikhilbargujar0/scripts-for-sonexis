from __future__ import annotations

import unittest

import numpy as np

from pipeline.code_switch import enrich_code_switch_segments, script_tag
from pipeline.diarisation import SpeakerTurn
from pipeline.language_detection import LanguageReport, LanguageSegment
from pipeline.metadata_extraction import (
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)
from pipeline.output_formatter import build_record
from pipeline.punctuation import apply_punctuation_metadata
from pipeline.schema_validator import validate_record
from pipeline.transcription import Transcript, TranscriptSegment, Word


class Phase3PunctuationCodeSwitchTests(unittest.TestCase):
    def test_punctuation_attaches_to_word_not_new_token(self):
        segment = TranscriptSegment(
            start=0.0,
            end=1.0,
            text="hello world",
            language="en",
            avg_logprob=-0.1,
            quality_score=0.9,
            words=[Word("hello", 0.0, 0.4, 0.9), Word("world", 0.5, 0.9, 0.9)],
        )
        apply_punctuation_metadata([segment], enabled=True)
        self.assertEqual(len(segment.words), 2)
        self.assertEqual(segment.words[-1].trailing_punct, ".")
        self.assertTrue(segment.punctuation_applied)

    def test_hindi_punctuation_is_honestly_marked_missing(self):
        segment = TranscriptSegment(
            start=0.0,
            end=1.0,
            text="नमस्ते कैसे हो",
            language="hi",
            avg_logprob=-0.1,
            words=[Word("नमस्ते", 0.0, 0.4, 0.9), Word("हो", 0.5, 0.9, 0.9)],
        )
        apply_punctuation_metadata([segment], enabled=True)
        data = segment.to_dict(segment_id="s_0000", audio_filepath="/tmp/a.wav", speaker_id="SPEAKER_00")
        self.assertFalse(data["punctuation_applied"])
        self.assertEqual(data["punct_skipped"], "hindi_model_unavailable")
        self.assertIn("punctuation", data["missing_fields"])

    def test_code_switch_enrichment_sets_word_scripts_and_switch_points(self):
        segment = TranscriptSegment(
            start=0.0,
            end=2.0,
            text="hello namaste hai",
            language="en",
            avg_logprob=-0.1,
            words=[
                Word("hello", 0.0, 0.5, 0.9),
                Word("namaste", 0.6, 1.0, 0.9),
                Word("hai", 1.1, 1.6, 0.9),
            ],
        )
        enrich_code_switch_segments([segment])
        self.assertEqual(script_tag("hello"), "Latn")
        self.assertEqual(segment.words[0].script, "Latn")
        self.assertEqual(segment.words[2].language, "hi-Latn")
        self.assertGreaterEqual(segment.cs_density, 1.0)
        self.assertTrue(segment.switch_points)
        self.assertIn(segment.switch_points[0]["switch_type"], {"tag", "intra_sentential", "intra_word"})

    def test_record_schema_validates_with_phase3_fields(self):
        transcript = Transcript(
            language="en",
            language_probability=0.9,
            duration=2.0,
            segments=[
                TranscriptSegment(
                    start=0.0,
                    end=2.0,
                    text="hello hai",
                    language="en",
                    avg_logprob=-0.1,
                    quality_score=0.9,
                    words=[Word("hello", 0.0, 0.8, 0.9), Word("hai", 0.9, 1.8, 0.9)],
                )
            ],
        )
        apply_punctuation_metadata(transcript.segments, enabled=True)
        enrich_code_switch_segments(transcript.segments)
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
        record = build_record(
            audio_path="/tmp/phase3.wav",
            transcript=transcript,
            turns=turns,
            language=LanguageReport(
                primary_language="en",
                confidence=0.9,
                dominant_language="en",
                language_segments=[
                    LanguageSegment(0.0, 1.0, "en", 0.9),
                    LanguageSegment(1.0, 2.0, "hi-Latn", 0.8),
                ],
            ),
            audio_meta=audio_meta,
            speaker_meta=speaker_meta,
            conversation_meta=extract_conversation_metadata(transcript, turns),
            monologue=None,
            session_name="phase3",
            model_versions={"whisper": "small"},
            config_hash="b" * 64,
            total_speech_duration_sec=2.0,
        )
        segment = record["transcript"]["segments"][0]
        self.assertIn("matrix_language", segment)
        self.assertIn("switch_points", segment)
        self.assertIn("cs_density", segment)
        self.assertIn("switch_points", record["code_switch"])
        validate_record(record)


if __name__ == "__main__":
    unittest.main()
