"""pipeline/steps/output.py

Re-exports from pipeline.output_formatter plus ``validate_record``.

``validate_record`` checks a finished dataset record against the canonical
JSON Schema in ``schema/dataset_schema.json``.  It raises ``SchemaError``
(a subclass of ``ValueError``) when the record is structurally invalid so
that the pipeline can fail early and produce a clear diagnostic instead of
silently writing a malformed file.

Usage::

    from pipeline.steps.output import build_record, validate_record, SchemaError

    record = build_record(...)
    try:
        validate_record(record)
    except SchemaError as e:
        log.error("schema validation failed: %s", e)
        raise
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from pipeline.output_formatter import build_record  # noqa: F401

log = logging.getLogger(__name__)

__all__ = [
    "build_record",
    "validate_record",
    "SchemaError",
]

# ── schema loading ─────────────────────────────────────────────────────────

_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__),   # pipeline/steps/
    "..", "..",                   # project root
    "schema",
    "dataset_schema.json",
)
_SCHEMA_PATH = os.path.normpath(_SCHEMA_PATH)

_schema_cache: Optional[Dict] = None


def _load_schema() -> Dict:
    global _schema_cache
    if _schema_cache is None:
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as fh:
            _schema_cache = json.load(fh)
    return _schema_cache


# ── public API ─────────────────────────────────────────────────────────────

class SchemaError(ValueError):
    """Raised when a dataset record does not conform to the JSON Schema."""


def validate_record(record: Dict[str, Any]) -> None:
    """Validate *record* against the Sonexis dataset schema.

    Requires the ``jsonschema`` package (``pip install jsonschema``).
    If ``jsonschema`` is not installed, validation is **skipped** with a
    warning rather than hard-failing, so the pipeline remains usable in
    minimal installations.

    Parameters
    ----------
    record :
        The dict returned by :func:`build_record`.

    Raises
    ------
    SchemaError
        If the record violates the schema.
    ImportError
        Not raised — the import failure is handled internally (logged
        as a warning and silently skipped).
    """
    try:
        import jsonschema
    except ImportError:
        log.warning(
            "jsonschema not installed; schema validation skipped.  "
            "Install with: pip install jsonschema"
        )
        return

    schema = _load_schema()
    try:
        jsonschema.validate(instance=record, schema=schema)
    except jsonschema.ValidationError as exc:
        raise SchemaError(
            f"Dataset record failed schema validation: {exc.message}\n"
            f"  Path: {' -> '.join(str(p) for p in exc.absolute_path)}"
        ) from exc
    except jsonschema.SchemaError as exc:
        # The schema itself is malformed — log but don't block production.
        log.error("dataset_schema.json is invalid: %s", exc.message)
