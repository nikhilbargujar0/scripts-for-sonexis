from __future__ import annotations

import tempfile
import unittest

import numpy as np

from pipeline.confidence import annotate_segments_with_confidence
from pipeline.diarisation import SpeakerTurn
from pipeline.metadata_extraction import extract_audio_metadata
from pipeline.interaction_metadata import extract_interaction_metadata
from pipeline.quality_tier import classify_record
from pipeline.review_queue import build_review_queue, write_review_queue
from pipeline.snr import annotate_segments_with_snr, segment_snr
from pipeline.transcription import TranscriptSegment, Word


class Phase2QualitySignalsTests(unittest.TestCase):
    def test_segment_snr_emits_band_and_clipping_reason(self):
        wav = np.ones(16000, dtype=np.float32)
        result = segment_snr(wav, 16000, 0.0, 1.0)
        self.assertEqual(result.snr_band, "low")
        self.assertEqual(result.snr_reason, "clipped")

    def test_snr_and_confidence_are_attached_to_segments(self):
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False)
        wav = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        seg = TranscriptSegment(
            start=0.0,
            end=1.0,
            text="hello world",
            language="en",
            avg_logprob=-0.1,
            quality_score=0.9,
            words=[Word("hello", 0.0, 0.4, 0.9), Word("world", 0.5, 0.9, 0.9)],
        )
        annotate_segments_with_snr([seg], wav, sr)
        annotate_segments_with_confidence([seg], [])
        data = seg.to_dict(segment_id="s_0000", audio_filepath="/tmp/a.wav", speaker_id="SPEAKER_00")
        self.assertIn(data["snr_band"], {"high", "medium", "low"})
        self.assertIn(data["confidence_band"], {"gold", "silver", "bronze"})
        self.assertNotIn("snr_db", data["missing_fields"])
        self.assertNotIn("confidence", data["missing_fields"])

    def test_overlap_metadata_uses_phase2_shape(self):
        turns = [
            SpeakerTurn(0.0, 1.0, "SPEAKER_00", 1.0),
            SpeakerTurn(0.75, 1.4, "SPEAKER_01", 1.0),
        ]
        meta, ratios, overlaps = extract_interaction_metadata(turns, interruption_threshold_s=0.5)
        self.assertGreater(meta["overlap_duration_s"], 0)
        self.assertGreater(ratios["SPEAKER_00"], 0)
        payload = overlaps[0].to_dict()
        self.assertTrue(payload["overlap"])
        self.assertEqual(payload["overlap_type"], "backchannel")
        self.assertEqual(payload["overlap_speakers"], ["SPEAKER_00", "SPEAKER_01"])

    def test_quality_tier_and_review_queue(self):
        record = {
            "session_id": "sess",
            "file": {"path": "/tmp/a.wav"},
            "validation": {"checks": {}},
            "transcript": {
                "segments": [
                    {
                        "segment_id": "sess_0000",
                        "audio_filepath": "/tmp/a.wav",
                        "start": 0.0,
                        "end": 1.0,
                        "confidence_band": "gold",
                        "confidence": 0.9,
                        "confidence_reasons": [],
                        "snr_band": "high",
                        "overlap": False,
                    },
                    {
                        "segment_id": "sess_0001",
                        "audio_filepath": "/tmp/a.wav",
                        "start": 1.0,
                        "end": 2.0,
                        "confidence_band": "silver",
                        "confidence": 0.6,
                        "confidence_reasons": ["snr_low"],
                        "snr_band": "low",
                        "overlap": False,
                    },
                ]
            },
        }
        classify_record(record)
        self.assertEqual(record["transcript"]["segments"][0]["quality_tier"], "gold")
        self.assertEqual(record["transcript"]["segments"][1]["quality_tier"], "bronze")
        rows = build_review_queue(record)
        self.assertEqual(len(rows), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_review_queue(record, tmp)
            self.assertIsNotNone(path)

    def test_audio_metadata_tags_bandwidth_class(self):
        wav = np.zeros(16000, dtype=np.float32)
        meta = extract_audio_metadata(wav, 16000, [(0.0, 1.0)])
        self.assertIn(meta["audio_band"], {"narrowband", "wideband"})


if __name__ == "__main__":
    unittest.main()
