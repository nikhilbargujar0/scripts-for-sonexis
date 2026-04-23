"""Confidence-safe metadata field helpers."""
from __future__ import annotations

from typing import Any, Dict


def _clean_value(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return text or "unknown"


def inferred_field(value: Any, confidence: float, threshold: float = 0.5) -> Dict:
    """Wrap inferred metadata so low-confidence values never look precise."""
    clean = _clean_value(value)
    score = round(float(confidence), 3)
    if clean == "unknown" or score < threshold:
        clean = "uncertain"
    return {
        "value": clean,
        "confidence": score,
        "source": "inferred",
    }


def provided_field(value: Any, source: str = "user_provided") -> Dict:
    """Wrap user-provided metadata. User input is treated as authoritative."""
    return {
        "value": _clean_value(value),
        "confidence": 1.0,
        "source": source,
    }


def measured_field(
    value: Any,
    confidence: float,
    *,
    method: str,
    source: str = "measured",
) -> Dict:
    """Wrap measured or estimated numeric metadata with provenance."""
    return {
        "value": value,
        "confidence": round(float(confidence), 3),
        "source": source,
        "method": method,
    }
