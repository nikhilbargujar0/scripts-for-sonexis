from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import main as cli


class CliTests(unittest.TestCase):
    def test_bool_parser_accepts_explicit_false_values(self) -> None:
        self.assertFalse(cli._bool("false"))
        self.assertFalse(cli._bool("0"))
        self.assertFalse(cli._bool("off"))

    def test_bool_parser_rejects_typos(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._bool("treu")

    def test_output_format_reaches_pipeline_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            result = {
                "output_path": str(output),
                "records": [],
                "validation_reports": [],
                "downloads": {},
            }
            with patch.object(cli, "process_conversation", return_value=result) as mocked:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = cli.main(
                        [
                            "--input",
                            str(Path(tmp) / "input.wav"),
                            "--output",
                            str(output),
                            "--offline_mode",
                            "false",
                            "--output_format",
                            "jsonl",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            cfg = mocked.call_args.args[2]
            self.assertFalse(cfg.offline_mode)
            self.assertEqual(cfg.output_format, "jsonl")

    def test_premium_config_must_be_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "premium.json"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

            with self.assertRaises(ValueError):
                cli._load_premium_config(str(path))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
