from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from pipeline.input_normalizer import normalize_messy_input
from scripts import main as cli


def _wav(path: Path, frames: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * frames)


def _speaker_pair(root: Path, folder: str = "1", s1: str = "Speaker 1/Speaker 1.wav", s2: str = "Speaker 2/Speaker 2.wav") -> None:
    _wav(root / folder / s1)
    _wav(root / folder / s2)


class InputNormalizerTests(unittest.TestCase):
    def test_messy_numbered_conversations_normalise_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _speaker_pair(root / "input" / "english")
            report = normalize_messy_input(root / "input", root / "work")
            metadata_path = root / "work" / "english" / "conversation_0001" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(report["languages"]["english"]["valid_conversations"], 1)
            self.assertTrue(metadata_path.exists())
            self.assertTrue((root / "work" / "english" / "conversation_0001" / "speaker_1" / "speaker_1.wav").exists())
            self.assertTrue((root / "work" / "english" / "conversation_0001" / "speaker_2" / "speaker_2.wav").exists())
            self.assertEqual(metadata["language"], "en")
            self.assertEqual(metadata["language_folder"], "english")
            self.assertTrue(metadata["input_normalized"])

    def test_case_insensitive_language_and_speaker_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _speaker_pair(root / "input" / "English", folder="2", s1="sPk_1/audio.wav", s2="SPEAKER-2/audio.wav")
            report = normalize_messy_input(root / "input", root / "work")
            self.assertEqual(report["languages"]["english"]["valid_conversations"], 1)
            self.assertTrue((root / "work" / "english" / "conversation_0001").exists())

    def test_ignores_junk_folders_and_zip_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wav(root / "input" / "hinglish" / "output" / "Speaker 1" / "Speaker 1.wav")
            _wav(root / "input" / "hinglish" / "QA" / "Speaker 2" / "Speaker 2.wav")
            (root / "input" / "hinglish" / "old.zip").parent.mkdir(parents=True, exist_ok=True)
            (root / "input" / "hinglish" / "old.zip").write_bytes(b"zip")
            _speaker_pair(root / "input" / "hinglish")
            report = normalize_messy_input(root / "input", root / "work")
            self.assertEqual(report["languages"]["hinglish"]["valid_conversations"], 1)
            self.assertEqual(report["total_valid_conversations"], 1)

    def test_skips_incomplete_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _wav(root / "input" / "hindi" / "1" / "Speaker 1" / "Speaker 1.wav")
            report = normalize_messy_input(root / "input", root / "work")
            self.assertEqual(report["languages"]["hindi"]["valid_conversations"], 0)
            reasons = [row["reason"] for row in report["languages"]["hindi"]["skipped"]]
            self.assertIn("missing_speaker_2", reasons)

    def test_preserves_source_metadata_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conv = root / "input" / "punjabi" / "1"
            conv.mkdir(parents=True, exist_ok=True)
            (conv / "metadata.json").write_text(
                json.dumps({"scenario": "customer support", "language": "wrong"}),
                encoding="utf-8",
            )
            _speaker_pair(root / "input" / "punjabi")
            normalize_messy_input(root / "input", root / "work")
            metadata = json.loads((root / "work" / "punjabi" / "conversation_0001" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["language"], "pa")
            self.assertEqual(metadata["source_metadata"]["scenario"], "customer support")
            self.assertEqual(metadata["source_metadata"]["language"], "wrong")

    def test_multi_language_root_creates_separate_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _speaker_pair(root / "input" / "english")
            _speaker_pair(root / "input" / "hindi")
            report = normalize_messy_input(root / "input", root / "work")
            self.assertEqual(report["total_valid_conversations"], 2)
            self.assertTrue((root / "work" / "english" / "conversation_0001").exists())
            self.assertTrue((root / "work" / "hindi" / "conversation_0001").exists())

    def test_no_valid_conversations_fails_clearly_through_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input" / "english" / "1").mkdir(parents=True)
            output = root / "output"
            with self.assertRaisesRegex(RuntimeError, "No valid speaker-separated conversations"):
                cli.main([
                    "--input", str(root / "input"),
                    "--output", str(output),
                    "--normalise_messy_input", "true",
                    "--offline_mode", "false",
                ])

            self.assertTrue((output / "_normalized_input" / "normalization_report.json").exists())

    def test_audit_input_only_does_not_run_asr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _speaker_pair(root / "input" / "english")
            output = root / "output"
            with patch.object(cli, "process_conversation") as mocked:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = cli.main([
                        "--input", str(root / "input"),
                        "--output", str(output),
                        "--normalise_messy_input", "true",
                        "--audit_input_only", "true",
                        "--offline_mode", "false",
                    ])

            self.assertEqual(exit_code, 0)
            mocked.assert_not_called()
            self.assertTrue((output / "_normalized_input" / "normalization_report.json").exists())
            self.assertFalse((output / "english" / "annotations").exists())
            self.assertFalse((output / "english" / "review").exists())
            self.assertFalse((output / "english" / "transcripts").exists())
            self.assertFalse((output / "english" / "manifests").exists())

    def test_main_processes_each_normalized_language_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _speaker_pair(root / "input" / "english")
            _speaker_pair(root / "input" / "hindi")
            output = root / "output"

            def fake_process(input_path: str, output_path: str, cfg):
                return {
                    "output_path": output_path,
                    "records": [{"session_name": Path(input_path).name}],
                    "validation_reports": [],
                    "downloads": {},
                }

            with patch.object(cli, "process_conversation", side_effect=fake_process) as mocked:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = cli.main([
                        "--input", str(root / "input"),
                        "--output", str(output),
                        "--normalise_messy_input", "true",
                        "--offline_mode", "false",
                    ])

            self.assertEqual(exit_code, 0)
            called_outputs = {Path(call.args[1]).relative_to(output).as_posix() for call in mocked.call_args_list}
            self.assertEqual(called_outputs, {"english", "hindi"})
            for call in mocked.call_args_list:
                self.assertEqual(call.args[2].input_type, "speaker_folders")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
