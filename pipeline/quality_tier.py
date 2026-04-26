"""Gold/Silver/Bronze quality tiering."""
from __future__ import annotations

from typing import Dict, List


def classify_segment(segment: Dict) -> Dict:
    tier = str(segment.get("confidence_band") or "bronze")
    reasons: List[str] = list(segment.get("confidence_reasons") or [])
    if segment.get("overlap"):
        tier = "silver" if tier == "gold" else "bronze"
        reasons.append("overlap")
    if segment.get("snr_band") == "low":
        tier = "bronze"
        reasons.append("snr_low")
    if "clipping_detected" in reasons:
        tier = "bronze"
    flagged = tier == "bronze" or bool(reasons and any(r in {"overlap", "snr_low", "clipping_detected"} for r in reasons))
    segment["quality_tier"] = tier
    segment["tier_reasons"] = sorted(set(reasons))
    segment["flagged_for_review"] = bool(flagged)
    segment["review_reasons"] = list(segment["tier_reasons"]) if flagged else []
    return segment


def classify_record(record: Dict) -> Dict:
    segments = record.get("transcript", {}).get("segments", [])
    counts = {"gold": 0, "silver": 0, "bronze": 0}
    for segment in segments:
        classify_segment(segment)
        counts[segment["quality_tier"]] = counts.get(segment["quality_tier"], 0) + 1
    record["quality_tiers"] = counts
    record.setdefault("validation", {}).setdefault("checks", {})["quality_tiers"] = counts
    return record
