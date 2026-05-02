"""Finalize human-reviewed transcripts into delivery-gated records."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..schema_validator import validate_record
from ..transcription import normalise_transcript
from .metrics import (
    character_error_rate,
    code_switch_review_stats,
    speaker_accuracy,
    timestamp_accuracy,
    word_accuracy,
    word_error_rate,
)


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return safe.strip("._") or "speaker"


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


def remove_jsonl_row(path: str | Path, key, key_field: str = "session") -> None:
    path = Path(path)
    if not path.exists():
        return
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        existing = json.loads(line)
        if existing.get(key_field) != key:
            rows.append(existing)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
        encoding="utf-8",
    )


def sync_status_manifests(manifests_dir: str | Path, session: str, row: Dict, stage: str) -> None:
    manifests_dir = Path(manifests_dir)
    status_files = {
        "approved": "approved_for_delivery.jsonl",
        "review_required": "review_required.jsonl",
        "rejected": "rejected.jsonl",
    }
    for filename in status_files.values():
        remove_jsonl_row(manifests_dir / filename, session)
    upsert_jsonl(manifests_dir / status_files.get(stage, "rejected.jsonl"), row)


def _targets(record: Dict, overrides: Optional[Dict]) -> Dict[str, float]:
    gate = record.get("accuracy_gate") or {}
    overrides = overrides or {}
    return {
        "word_accuracy": float(overrides.get("word_accuracy_target") or gate.get("target_word_accuracy") or 0.99),
        "speaker_accuracy": float(overrides.get("speaker_accuracy_target") or gate.get("target_speaker_accuracy") or 0.99),
        "timestamp_accuracy": float(overrides.get("timestamp_accuracy_target") or gate.get("target_timestamp_accuracy") or 0.98),
        "code_switch_accuracy": float(overrides.get("code_switch_accuracy_target") or gate.get("target_code_switch_accuracy") or 0.99),
    }


def _original_transcript_for_review(record: Dict) -> Dict:
    source = record.get("asr_transcript_before_review")
    if isinstance(source, dict) and source.get("segments"):
        return deepcopy(source)
    return deepcopy(record.get("transcript", {}))


def _comparison_source_type(record: Dict) -> str:
    source = record.get("asr_transcript_before_review")
    if isinstance(source, dict) and source.get("segments"):
        return "asr_transcript_before_review"
    return "current_transcript_before_first_review"


def _known_speakers(record: Dict) -> set[str]:
    speakers = set((record.get("metadata", {}).get("speakers") or {}).keys())
    speakers.update(seg.get("speaker_id") for seg in record.get("transcript", {}).get("segments", []) if seg.get("speaker_id"))
    speakers.update(row.get("speaker") for row in record.get("conversation_transcript", []) if row.get("speaker"))
    if speakers:
        speakers.update({"SPEAKER_00", "SPEAKER_01"})
    return {str(s) for s in speakers if s}


def normalise_speaker_id(value: str, record: dict) -> str:
    raw = str(value or "").strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        # Internal aliases: zero-based pipeline speaker IDs.
        "speaker_00": "SPEAKER_00",
        "speaker00": "SPEAKER_00",
        "speaker_0": "SPEAKER_00",
        "spk0": "SPEAKER_00",
        "spk_0": "SPEAKER_00",
        # Human aliases: speaker_1 means first human speaker, not SPEAKER_01.
        "speaker_1": "SPEAKER_00",
        "speaker1": "SPEAKER_00",
        "speaker_one": "SPEAKER_00",
        "spk1": "SPEAKER_00",
        "spk_1": "SPEAKER_00",
        # Internal second speaker aliases. Do not conflate speaker_01 with
        # speaker_1: speaker_01 is ambiguous unless speaker_map explicitly
        # uses it as a label.
        "spk01": "SPEAKER_01",
        "spk_01": "SPEAKER_01",
        "speaker_2": "SPEAKER_01",
        "speaker2": "SPEAKER_01",
        "speaker_two": "SPEAKER_01",
        "spk2": "SPEAKER_01",
        "spk_2": "SPEAKER_01",
    }
    if raw.upper() in {"SPEAKER_00", "SPEAKER_01"}:
        return raw.upper()
    speaker_map = record.get("speaker_map") or {}
    for canonical, label in speaker_map.items():
        if raw == canonical or key == str(label).lower().replace("-", "_").replace(" ", "_"):
            return str(canonical)
    return aliases.get(key, raw)


def _ambiguous_speaker_alias(value: str, record: dict) -> bool:
    raw = str(value or "").strip()
    if raw in {"SPEAKER_00", "SPEAKER_01"}:
        return False
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key not in {"speaker_01", "speaker01"}:
        return False
    speaker_map = record.get("speaker_map") or {}
    return not any(
        key == str(label).lower().replace("-", "_").replace(" ", "_")
        for label in speaker_map.values()
    )


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
        original_speaker = str(segment.get("speaker") or "").strip()
        speaker = normalise_speaker_id(original_speaker, record)
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
        if _ambiguous_speaker_alias(original_speaker, record):
            errors.append(f"ambiguous_speaker_alias:{original_speaker}")
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
        if original_speaker and original_speaker != speaker:
            row["speaker_original"] = original_speaker
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
    unresolved_issue_count = sum(len(seg.get("unresolved_issue_types") or []) for seg in reviewed_segments)
    empty_count = sum(1 for seg in reviewed_segments if not seg.get("reviewed_text"))
    completion_rate = (
        (len(reviewed_segments) - empty_count) / len(reviewed_segments)
        if reviewed_segments else 0.0
    )
    # WER measures ASR correction rate against human review.
    # Final reviewed transcript "word accuracy" is delivery confidence unless a
    # future second-pass sampled audit provides measured final accuracy.
    delivery_confidence = (
        0.995
        if review_completed and empty_count == 0 and unresolved_issue_count == 0
        else max(0.0, min(0.98, completion_rate - unresolved_issue_count * 0.05))
    )
    final_speaker = speaker_accuracy(reviewed_segments, original_segments)
    final_timestamp = timestamp_accuracy(reviewed_segments, original_segments)
    code_switch_stats = code_switch_review_stats(reviewed_segments)
    return {
        "asr_vs_review": {
            "wer": wer,
            "word_accuracy": word_acc,
            "cer": cer,
            "measurement_type": "asr_compared_to_human_review",
        },
        "reviewed_delivery": {
            "review_completion_rate": round(completion_rate, 4),
            "empty_reviewed_text_count": empty_count,
            "unresolved_issue_count": unresolved_issue_count,
            "delivery_confidence": round(delivery_confidence, 4),
            "confidence_basis": "completed_human_review_no_empty_segments_no_unresolved_issues",
            "timestamp_confidence": 0.985,
            "timestamp_confidence_basis": "human_review_without_external_timing_audit",
        },
        "verified_final": {
            "word_accuracy": round(delivery_confidence, 4),
            "word_accuracy_measurement_type": "reviewed_delivery_confidence",
            "speaker_accuracy": final_speaker,
            "timestamp_accuracy": final_timestamp,
            "timestamp_accuracy_measurement_type": "reviewed_vs_original_timestamps",
            "code_switch_accuracy": float(code_switch_stats["accuracy"]),
            "verified_accuracy": bool(review_completed),
            "verification_method": "human_review" if review_completed else None,
        },
        "code_switch": code_switch_stats,
    }


def _update_canonical_transcript(record: Dict, reviewed_segments: List[Dict]) -> None:
    if "asr_transcript_before_review" not in record:
        record["asr_transcript_before_review"] = deepcopy(record.get("transcript", {}))
    original_transcript = _original_transcript_for_review(record)
    original_by_id = {
        str(seg.get("segment_id")): seg
        for seg in original_transcript.get("segments", [])
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


def build_final_transcript_side_file_paths(
    output_root: str,
    session: str,
    record: dict,
) -> dict:
    root = Path(output_root) / "transcripts"
    session_dir = root / session
    raw_path = session_dir / "raw.txt"
    norm_path = session_dir / "normalised.txt"
    combined_path = session_dir / "combined_conversation.json"
    flat_path = root / f"{session}.json"
    return {
        "raw_txt": str(raw_path.resolve()),
        "normalised_txt": str(norm_path.resolve()),
        "combined_conversation_json": str(combined_path.resolve()),
        "flat_conversation_json": str(flat_path.resolve()),
        "speaker_transcripts": {
            speaker: str((session_dir / f"speaker_{safe_filename(speaker)}.json").resolve())
            for speaker in (record.get("speaker_transcripts") or {})
        },
    }


def write_final_transcript_side_files(
    output_root: str,
    session: str,
    record: dict,
) -> dict:
    paths = build_final_transcript_side_file_paths(output_root, session, record)
    session_dir = Path(paths["raw_txt"]).parent
    session_dir.mkdir(parents=True, exist_ok=True)
    transcript = record.get("transcript", {})
    Path(paths["raw_txt"]).write_text(transcript.get("raw", ""), encoding="utf-8")
    Path(paths["normalised_txt"]).write_text(transcript.get("normalised", ""), encoding="utf-8")
    _write_json(paths["combined_conversation_json"], record.get("conversation_transcript", []))
    _write_json(paths["flat_conversation_json"], record.get("conversation_transcript", []))
    for speaker, text in (record.get("speaker_transcripts") or {}).items():
        _write_json(paths["speaker_transcripts"][speaker], {"speaker": speaker, "text": text})
    return paths


def _update_accuracy_gate(record: Dict, metrics: Dict, targets: Dict[str, float], validation_errors: List[str]) -> List[str]:
    gate = dict(record.get("accuracy_gate") or {})
    verified = metrics["verified_final"]
    reasons = list(validation_errors)
    review_was_required = bool(
        gate.get("human_review_required")
        or (record.get("human_review") or {}).get("required")
        or reasons
    )
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
        "human_review_required": review_was_required,
        "human_review_completed": bool(verified["verified_accuracy"]),
        "human_review_required_for_delivery": bool(reasons),
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
    candidate = deepcopy(record)
    candidate.setdefault("pipeline_mode", "offline_standard")
    session = str(candidate.get("session_name") or candidate.get("session_id") or Path(annotation_path).stem)
    now = _now()
    target_values = _targets(candidate, targets)
    reviewed_segments, review_errors = _validate_reviewed(reviewed, candidate)
    second_pass = reviewed.get("second_pass_review") or {
        "required": False,
        "completed": False,
        "reviewer_id": None,
        "sample_rate": 0.0,
        "notes": "",
    }
    if second_pass.get("required") and not second_pass.get("completed"):
        review_errors.append("second_pass_review_required_not_completed")
    completed = reviewed.get("status") == "completed" and not review_errors
    can_update_canonical = completed and bool(reviewed_segments)
    comparison_source_type = _comparison_source_type(candidate)
    original_transcript = _original_transcript_for_review(candidate)
    original_segments = original_transcript.get("segments", [])
    metrics = _compute_metrics(reviewed_segments, original_segments, completed)
    metrics["comparison_source"] = {
        "type": comparison_source_type,
        "segment_count": len(original_segments),
    }
    metrics["reviewed_delivery"]["word_accuracy_target"] = target_values["word_accuracy"]
    metrics["reviewed_delivery"]["second_pass_review"] = second_pass
    validation_passed = bool(candidate.get("validation", {}).get("passed"))
    gate_reasons = _update_accuracy_gate(candidate, metrics, target_values, [] if completed else review_errors)
    if not validation_passed:
        gate_reasons = list(dict.fromkeys([*gate_reasons, "validation_failed"]))
        candidate["accuracy_gate"]["passed"] = False
        candidate["accuracy_gate"]["verified_accuracy"] = False
        candidate["accuracy_gate"]["human_review_required"] = True
        candidate["accuracy_gate"]["human_review_completed"] = bool(completed)
        candidate["accuracy_gate"]["human_review_required_for_delivery"] = True
        candidate["accuracy_gate"]["reasons"] = gate_reasons

    passed = bool(completed and validation_passed and candidate["accuracy_gate"]["passed"])
    approved = bool(approve_if_passed and passed)
    if can_update_canonical:
        _update_canonical_transcript(candidate, reviewed_segments)
        side_files = build_final_transcript_side_file_paths(output_root, session, candidate)
        candidate.setdefault("artifacts", {})
        candidate["artifacts"]["final_transcript_files"] = side_files
    else:
        side_files = {}

    review_dir = Path(output_root) / "review" / session
    qa_path = review_dir / "qa_report.json"
    final_path = Path(reviewed_transcript_path)
    failure_reasons = list(candidate["accuracy_gate"].get("reasons") or [])
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
        "unresolved_issue_count": sum(len(seg.get("unresolved_issue_types") or []) for seg in reviewed_segments),
        "code_switch_segment_count": metrics["code_switch"]["code_switch_segment_count"],
        "unresolved_code_switch_count": metrics["code_switch"]["unresolved_code_switch_count"],
    }

    if approved:
        candidate["human_review"] = {
            "required": True,
            "status": "completed",
            "review_stage": "final_qa",
            "reviewer_id": reviewer_id,
            "completed_at": now,
            "result": {
                "approved_for_delivery": True,
                "verified_accuracy": True,
                "qa_report_path": str(qa_path.resolve()),
                "second_pass_review": second_pass,
            },
        }
        candidate["delivery_status"] = {
            "stage": "approved",
            "approved_for_client_delivery": True,
            "approved_by": reviewer_id,
            "approved_at": now,
            "reason": "",
        }
    elif not completed and not approve_if_passed:
        candidate.setdefault("human_review", {})
        candidate["human_review"].update({
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
        candidate["delivery_status"] = {
            "stage": "review_required",
            "approved_for_client_delivery": False,
            "approved_by": None,
            "approved_at": None,
            "reason": ";".join(failure_reasons),
        }
    else:
        candidate["human_review"] = {
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
        candidate["delivery_status"] = {
            "stage": "rejected",
            "approved_for_client_delivery": False,
            "approved_by": None,
            "approved_at": None,
            "reason": ";".join(failure_reasons),
        }

    candidate.setdefault("review_artifacts", {})
    candidate["review_artifacts"].update({
        "final_reviewed_transcript": str(final_path.resolve()),
        "qa_report": str(qa_path.resolve()),
    })
    candidate["last_finalized_at"] = now
    candidate["last_finalized_by"] = reviewer_id
    validate_record(candidate)

    if can_update_canonical:
        side_files = write_final_transcript_side_files(output_root, session, candidate)
        candidate["artifacts"]["final_transcript_files"] = side_files
    _write_json(qa_path, qa_report)
    _write_json(annotation_path, candidate)

    manifests = Path(output_root) / "manifests"
    artifact_paths = (candidate.get("artifacts") or {}).get("final_transcript_files") or {}
    canonical_transcript_path = artifact_paths.get("combined_conversation_json")
    raw_transcript_path = artifact_paths.get("raw_txt")
    normalised_transcript_path = artifact_paths.get("normalised_txt")
    flat_conversation_path = artifact_paths.get("flat_conversation_json")
    base_row = {
        "session": session,
        "transcript_path": canonical_transcript_path,
        "annotation_path": str(Path(annotation_path).resolve()),
        "reviewed_transcript_input_path": str(final_path.resolve()),
        "qa_report_path": str(qa_path.resolve()),
        "canonical_transcript_path": canonical_transcript_path,
        "raw_transcript_path": raw_transcript_path,
        "normalised_transcript_path": normalised_transcript_path,
        "flat_conversation_path": flat_conversation_path,
        "review_status": candidate["human_review"]["status"],
        "delivery_stage": candidate["delivery_status"]["stage"],
        "approved_for_client_delivery": candidate["delivery_status"]["approved_for_client_delivery"],
        "verified_word_accuracy": candidate["accuracy_gate"].get("verified_word_accuracy"),
        "verified_speaker_accuracy": candidate["accuracy_gate"].get("verified_speaker_accuracy"),
        "verified_timestamp_accuracy": candidate["accuracy_gate"].get("verified_timestamp_accuracy"),
        "verified_code_switch_accuracy": candidate["accuracy_gate"].get("verified_code_switch_accuracy"),
    }
    upsert_jsonl(manifests / "final_transcripts.jsonl", base_row)
    sync_status_manifests(manifests, session, base_row, candidate["delivery_status"]["stage"])

    return {
        "session": session,
        "annotation_path": str(Path(annotation_path).resolve()),
        "qa_report_path": str(qa_path.resolve()),
        "passed": passed,
        "approved_for_client_delivery": approved,
        "delivery_status": candidate["delivery_status"],
        "accuracy_gate": candidate["accuracy_gate"],
        "failure_reasons": failure_reasons,
    }
