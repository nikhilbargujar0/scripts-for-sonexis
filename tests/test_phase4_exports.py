from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.diarisation import SpeakerTurn
from pipeline.exporters import export_phase4_formats
from pipeline.language_detection import LanguageReport
from pipeline.metadata_extraction import (
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)
from pipeline.output_formatter import build_record
from pipeline.quality_tier import classify_record
from pipeline.transcription import Transcript, TranscriptSegment, Word


def _record() -> dict:
    transcript = Transcript(
        language="en",
        language_probability=0.9,
        duration=2.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="hello world",
                language="en",
                avg_logprob=-0.1,
                quality_score=0.9,
                words=[Word("hello", 0.0, 0.8, 0.9), Word("world", 0.9, 1.8, 0.9)],
            )
        ],
    )
    setattr(transcript.segments[0], "confidence", 0.9)
    setattr(transcript.segments[0], "confidence_band", "gold")
    setattr(transcript.segments[0], "snr_band", "high")
    setattr(transcript.segments[0], "punctuation_applied", True)
    setattr(transcript.segments[0], "matrix_language", "en")
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
        audio_path="/tmp/phase4.wav",
        transcript=transcript,
        turns=turns,
        language=LanguageReport(primary_language="en", confidence=0.9),
        audio_meta=audio_meta,
        speaker_meta=speaker_meta,
        conversation_meta=extract_conversation_metadata(transcript, turns),
        monologue=None,
        session_name="phase4",
        model_versions={"whisper": "small"},
        config_hash="c" * 64,
        total_speech_duration_sec=2.0,
    )
    classify_record(record)
    return record


class Phase4ExportTests(unittest.TestCase):
    def test_phase4_exports_write_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = export_phase4_formats(_record(), tmp)
            required = {
                "csv_manifest",
                "tsv_manifest",
                "hf_dataset_jsonl",
                "hf_dataset_features",
                "hf_dataset_card",
                "kaldi_wav_scp",
                "kaldi_segments",
                "kaldi_text",
                "kaldi_utt2spk",
                "kaldi_spk2utt",
                "kaldi_utt2dur",
                "kaldi_utt2lang",
                "rttm",
                "ctm",
            }
            self.assertTrue(required.issubset(set(artifacts)))
            for key in required:
                self.assertTrue(Path(artifacts[key]).exists(), key)

    def test_kaldi_and_hf_content_is_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = export_phase4_formats(_record(), tmp)
            text = Path(artifacts["kaldi_text"]).read_text(encoding="utf-8")
            self.assertIn("hello world", text)
            utt2lang = Path(artifacts["kaldi_utt2lang"]).read_text(encoding="utf-8")
            self.assertIn("en-IN", utt2lang)
            rttm = Path(artifacts["rttm"]).read_text(encoding="utf-8")
            self.assertTrue(rttm.startswith("SPEAKER phase4"))
            ctm = Path(artifacts["ctm"]).read_text(encoding="utf-8")
            self.assertIn("hello", ctm)
            rows = [
                json.loads(line)
                for line in Path(artifacts["hf_dataset_jsonl"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["audio"]["path"], "/tmp/phase4.wav")
            self.assertEqual(rows[0]["quality_tier"], "gold")


if __name__ == "__main__":
    unittest.main()
