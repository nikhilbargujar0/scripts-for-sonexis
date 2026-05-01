"""Hard validation gate for Sonexis dataset records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

_SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "dataset_record.schema.json"
_validator: Optional[Draft202012Validator] = None


class SchemaVersionError(RuntimeError):
    """Raised when record['schema_version'] does not match DATASET_SCHEMA_VERSION."""


def load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _get_validator() -> Draft202012Validator:
    global _validator
    if _validator is None:
        _validator = Draft202012Validator(load_schema())
    return _validator


def validate_record(record: dict) -> None:
    """Raise on schema-version mismatch or structural schema violation."""
    from .output_formatter import DATASET_SCHEMA_VERSION

    actual_version = record.get("schema_version")
    if actual_version != DATASET_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"schema_version mismatch: expected {DATASET_SCHEMA_VERSION!r}, "
            f"got {actual_version!r}"
        )

    errors = list(_get_validator().iter_errors(record))
    if errors:
        errors.sort(key=lambda e: len(e.path), reverse=True)
        first = errors[0]
        path = " -> ".join(str(p) for p in first.path) if first.path else "(root)"
        raise ValidationError(
            f"Schema validation failed at [{path}]: {first.message} "
            f"({len(errors)} total error(s) in record)"
        )


def invalidate_cache() -> None:
    global _validator
    _validator = None
