"""Confidence-safe metadata helpers."""
from __future__ import annotations

from typing import Any, Dict


def confident(value: Any, confidence: float, threshold: float = 0.5) -> Dict:
    c = round(float(confidence), 3)
    return {"value": value if c >= threshold else "uncertain", "confidence": c}


def normalise_confidence_fields(metadata: Dict, threshold: float = 0.5) -> Dict:
    """Wrap obvious uncertain scalar metadata without inventing precision."""
    out = dict(metadata or {})
    for key in ("environment", "device_type", "accent", "speaking_style"):
        val = out.get(key)
        conf_key = f"{key}_confidence"
        if val is not None and not isinstance(val, dict):
            out[key] = confident(val, float(out.get(conf_key, 0.5) or 0.5), threshold)
    return out
