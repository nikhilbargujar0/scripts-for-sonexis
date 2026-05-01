from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.audio_loader import validate_studio_conversation_folder


class StudioSpeakerFolderTests(unittest.TestCase):
    def _write_conversation(self, root: Path, *, metadata: bool = True) -> Path:
        conversation = root / "conversation_0001"
        (conversation / "speaker_1").mkdir(parents=True)
        (conversation / "speaker_2").mkdir(parents=True)
        wav = np.zeros(4800, dtype=np.float32)
        sf.write(conversation / "speaker_1" / "speaker_1.wav", wav, 48_000, subtype="PCM_16")
        sf.write(conversation / "speaker_2" / "speaker_2.wav", wav, 48_000, subtype="PCM_16")
        if metadata:
            (conversation / "metadata.json").write_text(
                json.dumps(
                    {
                        "conversation_id": "conversation_0001",
                        "scenario_id": "scenario_001",
                        "scenario_name": "Casual planning",
                        "topic": "travel",
                        "sub_topic": "weekend itinerary",
                        "conversation_style": "unscripted",
                        "scripted": False,
                        "language_mix": ["en-IN"],
                    }
                ),
                encoding="utf-8",
            )
        return conversation

    def test_validates_canonical_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conversation = self._write_conversation(Path(tmp))
            session, p1, l1, p2, l2, metadata, context = validate_studio_conversation_folder(str(conversation))

        self.assertEqual(session, "conversation_0001")
        self.assertEqual((l1, l2), ("speaker_1", "speaker_2"))
        self.assertTrue(p1.endswith("speaker_1.wav"))
        self.assertTrue(p2.endswith("speaker_2.wav"))
        self.assertEqual(metadata["topic"], "travel")
        self.assertTrue(context["audio_format"]["speaker_1"]["sample_rate_preferred"])

    def test_missing_metadata_fails_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conversation = self._write_conversation(Path(tmp), metadata=False)
            with self.assertRaisesRegex(ValueError, "metadata.json is required"):
                validate_studio_conversation_folder(str(conversation))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
