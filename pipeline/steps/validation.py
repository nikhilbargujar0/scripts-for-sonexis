"""Validation and schema enforcement step."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ..validation import build_validation_report


def write_validation_report(report: Dict, output_dir: str, session_name: str = "session") -> str:
    path = Path(output_dir) / "validation" / f"{session_name}_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def load_dataset_schema(schema_path: str | None = None) -> Dict:
    if schema_path is None:
        schema_path = str(Path(__file__).resolve().parents[2] / "schema" / "dataset_schema.json")
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def validate_record_against_schema(record: Dict, schema_path: str | None = None) -> None:
    try:
        from jsonschema import validate
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jsonschema is required for schema enforcement. Run: pip install -r requirements.txt") from exc
    validate(instance=record, schema=load_dataset_schema(schema_path))


__all__ = ["build_validation_report", "write_validation_report", "validate_record_against_schema"]
