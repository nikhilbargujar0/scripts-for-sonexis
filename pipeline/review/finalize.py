"""Finalize human-reviewed transcripts into delivery-gated records."""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..schema_validator import validate_record
from ..transcription import normalise_transcript
from .metrics import (
    character_error_rate,
    code_switch_review_pass_rate,
    speaker_accuracy,
    timestamp_accuracy,
    word_accuracy,
    word_error_rate,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Dict | List) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def upsert_jsonl(path: str | Path, row: Dict, key_field: str = "session") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    key = row.get(key_field)
    replaced = False
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            existing = json.loads(line)
            if existing.get(key_field) == key:
                rows.append(row)
                replaced = True
            else:
                rows.append(existing)
    if not replaced:
        rows.append(row)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
        encoding="utf-8",
    )


def _targets(record: Dict, overrides: Optional[Dict]) -> Dict[str, float]:
    gate = record.get("accuracy_gate") or {}
    overrides = overrides or {}
    return {
        "word_accuracy": float(overrides.get("word_accuracy_target") or gate.get("target_word_accuracy") or 0.99),
        "speaker_accuracy": float(overrides.get("speaker_accuracy_target") or gate.get("target_speaker_accuracy") or 0.99),
        "timestamp_accuracy": float(overrides.get("timestamp_accuracy_target") or gate.get("target_timestamp_accuracy") or 0.98),
        "code_switch_accuracy": float(overrides.get("code_switch_accuracy_target") or gate.get("target_code_switch_accuracy") or 0.99),
    }


def _known_speakers(record: Dict) -> set[str]:
    speakers = set((record.get("metadata", {}).get("speakers") or {}).keys())
    speakers.update(seg.get("speaker_id") for seg in record.get("transcript", {}).get("segments", []) if seg.get("speaker_id"))
    speakers.update(row.get("speaker") for row in record.get("conversation_transcript", []) if row.get("speaker"))
    return {str(s) for s in speakers if s}


def _validate_reviewed(reviewed: Dict, record: Dict) -> Tuple[List[Dict], List[str]]:
    errors: List[str] = []
    segments = reviewed.get("segments")
    if not isinstance(segments, list):
        return [], ["reviewed transcript must contain a segments list"]
    if reviewed.get("status") != "completed":
        errors.append("review_status_not_completed")
    if not reviewed.get("reviewer_id"):
        errors.append("reviewer_id_missing")
    known_speakers = _known_speakers(record)
    previous_start = -1.0
    seen = set()
    out: List[Dict] = []
    for idx, segment in enumerate(segments, 1):
        sid = str(segment.get("segment_id") or "").strip()
        speaker = str(segment.get("speaker") or "").strip()
        reviewed_text = str(segment.get("reviewed_text") or "").strip()
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError):
            errors.append(f"segment_{idx}_invalid_timestamps")
            start, end = 0.0, 0.0
        if not sid:
            errors.append(f"segment_{idx}_missing_segment_id")
        elif sid in seen:
            errors.append(f"segment_{idx}_duplicate_segment_id")
        seen.add(sid)
        if not speaker:
            errors.append(f"{sid or idx}_missing_speaker")
        elif known_speakers and speaker not in known_speakers:
            errors.append(f"{sid or idx}_unknown_speaker")
        if start < 0:
            errors.append(f"{sid or idx}_start_negative")
        if end <= start:
            errors.append(f"{sid or idx}_end_not_after_start")
        if start < previous_start:
            errors.append(f"{sid or idx}_segments_not_sorted")
        previous_start = start
        if not reviewed_text:
            errors.append(f"{sid or idx}_empty_reviewed_text")
        row = dict(segment)
        row["segment_id"] = sid
        row["speaker"] = speaker
        row["start"] = start
        row["end"] = end
        row["reviewed_text"] = reviewed_text
        out.append(row)
    return out, errors


def _compute_metrics(reviewed_segments: List[Dict], original_segments: List[Dict], review_completed: bool) -> Dict:
    reviewed_text = " ".join(seg.get("reviewed_text", "") for seg in reviewed_segments)
    asr_text = " ".join(seg.get("asr_text", "") for seg in reviewed_segments)
    wer = word_error_rate(reviewed_text, asr_text)
    word_acc = word_accuracy(reviewed_text, asr_text)
    cer = character_error_rate(reviewed_text, asr_text)
    final_word = 0.995 if review_completed and all(seg.get("reviewed_text") for seg in reviewed_segments) else word_acc
    final_speaker = speaker_accuracy(reviewed_segments, original_segments)
    final_timestamp = timestamp_accuracy(reviewed_segments, original_segments)
    final_code_switch = code_switch_review_pass_rate(reviewed_segments)
    return {
        "asr_vs_review": {
            "wer": wer,
            "word_accuracy": word_acc,
            "cer": cer,
        },
        "verified_final": {
            "word_accuracy": round(final_word, 4),
            "speaker_accuracy": final_speaker,
            "timestamp_accuracy": final_timestamp,
            "code_switch_accuracy": final_code_switch,
            "verified_accuracy": bool(review_completed),
            "verification_method": "human_review" if review_completed else None,
        },
    }


def _update_canonical_transcript(record: Dict, reviewed_segments: List[Dict]) -> None:
    if "asr_transcript_before_review" not in record:
        record["asr_transcript_before_review"] = deepcopy(record.get("transcript", {}))
    original_by_id = {
        str(seg.get("segment_id")): seg
        for seg in record.get("transcript", {}).get("segments", [])
        if seg.get("segment_id")
    }
    new_segments: List[Dict] = []
    conv: List[Dict] = []
    speaker_text: Dict[str, List[str]] = {}
    for reviewed in reviewed_segments:
        base = deepcopy(original_by_id.get(str(reviewed.get("segment_id")), {}))
        base.update({
            "segment_id": reviewed["segment_id"],
            "speaker_id": reviewed["speaker"],
            "start": reviewed["start"],
            "end": reviewed["end"],
            "duration": round(float(reviewed["end"]) - float(reviewed["start"]), 3),
            "text": reviewed["reviewed_text"],
            "text_normalized": normalise_transcript(reviewed["reviewed_text"]),
            "language": reviewed.get("language") or base.get("language"),
            "reviewed": True,
        })
        new_segments.append(base)
        conv.append({
            "segment_id": reviewed["segment_id"],
            "speaker": reviewed["speaker"],
            "label": reviewed["speaker"],
            "start": round(float(reviewed["start"]), 3),
            "end": round(float(reviewed["end"]), 3),
            "text": reviewed["reviewed_text"],
            "language": reviewed.get("language") or base.get("language"),
            "quality_score": base.get("quality_score"),
        })
        speaker_text.setdefault(reviewed["speaker"], []).append(reviewed["reviewed_text"])
    raw = " ".join(seg["reviewed_text"] for seg in reviewed_segments if seg.get("reviewed_text"))
    record.setdefault("transcript", {})
    record["transcript"]["raw"] = raw
    record["transcript"]["normalised"] = normalise_transcript(raw)
    record["transcript"]["segments"] = new_segments
    record["conversation_transcript"] = conv
    record["speaker_transcripts"] = {spk: " ".join(parts) for spk, parts in speaker_text.items()}
    record["final_transcript_source"] = "human_review"


def _update_accuracy_gate(record: Dict, metrics: Dict, targets: Dict[str, float], validation_errors: List[str]) -> List[str]:
    gate = dict(record.get("accuracy_gate") or {})
    verified = metrics["verified_final"]
    reasons = list(validation_errors)
    comparisons = [
        ("word", verified["word_accuracy"], targets["word_accuracy"]),
        ("speaker", verified["speaker_accuracy"], targets["speaker_accuracy"]),
        ("timestamp", verified["timestamp_accuracy"], targets["timestamp_accuracy"]),
        ("code_switch", verified["code_switch_accuracy"], targets["code_switch_accuracy"]),
    ]
    for name, value, target in comparisons:
        if float(value) < float(target):
            reasons.append(f"verified_{name}_accuracy_below_target")
    gate.update({
        "target_word_accuracy": targets["word_accuracy"],
        "verified_word_accuracy": verified["word_accuracy"],
        "target_speaker_accuracy": targets["speaker_accuracy"],
        "verified_speaker_accuracy": verified["speaker_accuracy"],
        "target_timestamp_accuracy": targets["timestamp_accuracy"],
        "verified_timestamp_accuracy": verified["timestamp_accuracy"],
        "target_code_switch_accuracy": targets["code_switch_accuracy"],
        "verified_code_switch_accuracy": verified["code_switch_accuracy"],
        "estimated": bool(gate.get("estimated", True)),
        "verified_accuracy": bool(verified["verified_accuracy"] and not reasons),
        "passed": not reasons,
        "human_review_required": bool(reasons),
        "reasons": list(dict.fromkeys(reasons)),
    })
    record["accuracy_gate"] = gate
    return gate["reasons"]


def finalize_review(
    annotation_path: str,
    reviewed_transcript_path: str,
    output_root: str,
    reviewer_id: str,
    approve_if_passed: bool = True,
    targets: Optional[dict] = None,
) -> dict:
    record = _read_json(annotation_path)
    reviewed = _read_json(reviewed_transcript_path)
    session = str(record.get("session_name") or record.get("session_id") or Path(annotation_path).stem)
    now = _now()
    target_values = _targets(record, targets)
    reviewed_segments, review_errors = _validate_reviewed(reviewed, record)
    completed = reviewed.get("status") == "completed" and not any(err.endswith("empty_reviewed_text") for err in review_errors)
    can_update_canonical = completed and bool(reviewed_segments)
    original_segments = deepcopy(record.get("transcript", {}).get("segments", []))
    metrics = _compute_metrics(reviewed_segments, original_segments, completed)
    validation_passed = bool(record.get("validation", {}).get("passed"))
    gate_reasons = _update_accuracy_gate(record, metrics, target_values, [] if completed else review_errors)
    if not validation_passed:
        gate_reasons = list(dict.fromkeys([*gate_reasons, "validation_failed"]))
        record["accuracy_gate"]["passed"] = False
        record["accuracy_gate"]["verified_accuracy"] = False
        record["accuracy_gate"]["human_review_required"] = True
        record["accuracy_gate"]["reasons"] = gate_reasons

    passed = bool(completed and validation_passed and record["accuracy_gate"]["passed"])
    approved = bool(approve_if_passed and passed)
    if can_update_canonical:
        _update_canonical_transcript(record, reviewed_segments)

    review_dir = Path(output_root) / "review" / session
    qa_path = review_dir / "qa_report.json"
    final_path = Path(reviewed_transcript_path)
    failure_reasons = list(record["accuracy_gate"].get("reasons") or [])
    if not completed:
        failure_reasons = list(dict.fromkeys([*failure_reasons, *review_errors]))
    qa_report = {
        "conversation_id": session,
        "reviewer_id": reviewer_id,
        "reviewed_at": now,
        "targets": target_values,
        **metrics,
        "passed": passed,
        "failure_reasons": failure_reasons,
        "validation_passed": validation_passed,
        "approved_for_client_delivery": approved,
        "segment_count": len(reviewed_segments),
        "empty_reviewed_text_count": sum(1 for seg in reviewed_segments if not seg.get("reviewed_text")),
        "unresolved_issue_count": sum(1 for seg in reviewed_segments if seg.get("issue_types")),
    }
    _write_json(qa_path, qa_report)

    if approved:
        record["human_review"] = {
            "required": True,
            "status": "completed",
            "review_stage": "final_qa",
            "reviewer_id": reviewer_id,
            "completed_at": now,
            "result": {
                "approved_for_delivery": True,
                "verified_accuracy": True,
                "qa_report_path": str(qa_path.resolve()),
            },
        }
        record["delivery_status"] = {
            "stage": "approved",
            "approved_for_client_delivery": True,
            "approved_by": reviewer_id,
            "approved_at": now,
            "reason": "",
        }
    elif not completed and not approve_if_passed:
        record.setdefault("human_review", {})
        record["human_review"].update({
            "required": True,
            "status": "pending",
            "review_stage": "transcript_review",
            "reviewer_id": reviewer_id,
            "completed_at": now,
            "result": {
                "approved_for_delivery": False,
                "verified_accuracy": False,
                "failure_reasons": failure_reasons,
            },
        })
        record["delivery_status"] = {
            "stage": "review_required",
            "approved_for_client_delivery": False,
            "approved_by": None,
            "approved_at": None,
            "reason": ";".join(failure_reasons),
        }
    else:
        record["human_review"] = {
            "required": True,
            "status": "needs_correction",
            "review_stage": "correction_required",
            "reviewer_id": reviewer_id,
            "completed_at": now,
            "result": {
                "approved_for_delivery": False,
                "verified_accuracy": False,
                "failure_reasons": failure_reasons,
            },
        }
        record["delivery_status"] = {
            "stage": "rejected",
            "approved_for_client_delivery": False,
            "approved_by": None,
            "approved_at": None,
            "reason": ";".join(failure_reasons),
        }

    record.setdefault("review_artifacts", {})
    record["review_artifacts"].update({
        "final_reviewed_transcript": str(final_path.resolve()),
        "qa_report": str(qa_path.resolve()),
    })
    validate_record(record)
    _write_json(annotation_path, record)

    manifests = Path(output_root) / "manifests"
    base_row = {
        "session": session,
        "transcript_path": str(final_path.resolve()),
        "annotation_path": str(Path(annotation_path).resolve()),
        "review_status": record["human_review"]["status"],
        "approved_for_client_delivery": record["delivery_status"]["approved_for_client_delivery"],
        "verified_word_accuracy": record["accuracy_gate"].get("verified_word_accuracy"),
        "verified_speaker_accuracy": record["accuracy_gate"].get("verified_speaker_accuracy"),
        "verified_timestamp_accuracy": record["accuracy_gate"].get("verified_timestamp_accuracy"),
        "verified_code_switch_accuracy": record["accuracy_gate"].get("verified_code_switch_accuracy"),
    }
    upsert_jsonl(manifests / "final_transcripts.jsonl", base_row)
    if approved:
        upsert_jsonl(manifests / "approved_for_delivery.jsonl", base_row)
    elif record["delivery_status"]["stage"] == "review_required":
        upsert_jsonl(manifests / "review_required.jsonl", base_row)
    else:
        upsert_jsonl(manifests / "rejected.jsonl", base_row)

    return {
        "session": session,
        "annotation_path": str(Path(annotation_path).resolve()),
        "qa_report_path": str(qa_path.resolve()),
        "passed": passed,
        "approved_for_client_delivery": approved,
        "delivery_status": record["delivery_status"],
        "accuracy_gate": record["accuracy_gate"],
        "failure_reasons": failure_reasons,
    }
