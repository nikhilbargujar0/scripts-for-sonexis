"""Segment-level language and code-switch summaries."""
from __future__ import annotations

from typing import Dict, Iterable, List

LANG_NAMES = {
    "hi": "Hindi",
    "en": "English",
    "pa": "Punjabi",
    "raj": "Marwadi",
    "mrw": "Marwadi",
}


def _label(code: str) -> str:
    return LANG_NAMES.get((code or "").lower(), code or "unknown")


def build_code_switch_report(language_segments: Iterable[Dict]) -> Dict:
    segments: List[Dict] = []
    for seg in language_segments or []:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        raw = seg.get("lang") or seg.get("language") or seg.get("label") or "unknown"
        code = str(raw)
        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "lang": _label(code),
            "lang_code": code,
            "confidence": round(float(seg.get("confidence", 0.5) or 0.5), 3),
        })
    segments.sort(key=lambda s: (s["start"], s["end"]))

    switch_count = 0
    patterns: List[str] = []
    for a, b in zip(segments, segments[1:]):
        if a["lang_code"] != b["lang_code"]:
            switch_count += 1
            pattern = f"{a['lang_code']}→{b['lang_code']}"
            if pattern not in patterns:
                patterns.append(pattern)

    duration = max((s["end"] for s in segments), default=0.0) - min((s["start"] for s in segments), default=0.0)
    switching_frequency = switch_count / max(duration / 60.0, 1e-6) if duration > 0 else 0.0
    return {
        "segments": segments,
        "switch_count": switch_count,
        "patterns": patterns,
        "switching_frequency": round(float(switching_frequency), 4),
    }
