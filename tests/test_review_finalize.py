from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema.exceptions import ValidationError

from pipeline.dataset_writer import DatasetWriter
from pipeline.output_formatter import DATASET_SCHEMA_VERSION
from pipeline.review.finalize import finalize_review
from pipeline.review.metrics import timestamp_accuracy
from pipeline.schema_validator import validate_record


def _record(*, validation_passed: bool = True) -> dict:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "session_id": "conversation_0001",
        "session_name": "conversation_0001",
        "input_mode": "speaker_folders",
        "pipeline_mode": "premium_accuracy",
        "metadata": {"speakers": {"SPEAKER_00": {}, "SPEAKER_01": {}}},
        "validation": {"passed": validation_passed, "issues": []},
        "accuracy_gate": {
            "target_word_accuracy": 0.99,
            "estimated_word_accuracy": 0.96,
            "target_speaker_accuracy": 0.99,
            "estimated_speaker_accuracy": 1.0,
            "target_timestamp_accuracy": 0.98,
            "estimated_timestamp_accuracy": 0.97,
            "target_code_switch_accuracy": 0.99,
            "estimated_code_switch_accuracy": 0.9,
            "estimated": True,
            "verified_accuracy": False,
            "passed": False,
            "human_review_required": True,
            "reasons": ["estimated_word_accuracy_below_target"],
        },
        "human_review": {"required": True, "status": "pending"},
        "delivery_status": {
            "stage": "review_required",
            "approved_for_client_delivery": False,
            "reason": "human_review_pending",
        },
        "transcript": {
            "raw": "hello world reply",
            "normalised": "hello world reply",
            "segments": [
                {
                    "segment_id": "conversation_0001_seg_00001",
                    "speaker_id": "SPEAKER_00",
                    "start": 0.0,
                    "end": 2.0,
                    "duration": 2.0,
                    "text": "hello world",
                    "text_normalized": "hello world",
                    "language": "en-IN",
                    "sample_rate": 16000,
                },
                {
                    "segment_id": "conversation_0001_seg_00002",
                    "speaker_id": "SPEAKER_01",
                    "start": 2.1,
                    "end": 3.0,
                    "duration": 0.9,
                    "text": "reply",
                    "text_normalized": "reply",
                    "language": "en-IN",
                    "sample_rate": 16000,
                },
            ],
        },
        "conversation_transcript": [
            {"segment_id": "conversation_0001_seg_00001", "speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "hello world"},
            {"segment_id": "conversation_0001_seg_00002", "speaker": "SPEAKER_01", "start": 2.1, "end": 3.0, "text": "reply"},
        ],
        "speaker_transcripts": {"SPEAKER_00": "hello world", "SPEAKER_01": "reply"},
        "transcript_candidates": [],
        "consensus": {},
        "review_artifacts": {},
    }


def _review(*, status: str = "completed", blank: bool = False, shifted: bool = False) -> dict:
    return {
        "status": status,
        "reviewer_id": "reviewer_001",
        "segments": [
            {
                "segment_id": "conversation_0001_seg_00001",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 2.0 if not shifted else 4.0,
                "asr_text": "hello world",
                "reviewed_text": "" if blank else "hello world",
                "language": "en-IN",
                "review_reasons": [],
                "resolved_issue_types": [],
                "unresolved_issue_types": [],
                "review_notes": "",
                "issue_types": [],
                "needs_review": False,
            },
            {
                "segment_id": "conversation_0001_seg_00002",
                "speaker": "SPEAKER_01",
                "start": 2.1 if not shifted else 4.2,
                "end": 3.0 if not shifted else 5.0,
                "asr_text": "reply",
                "reviewed_text": "reply",
                "language": "en-IN",
                "review_reasons": [],
                "resolved_issue_types": [],
                "unresolved_issue_types": [],
                "review_notes": "",
                "issue_types": [],
                "needs_review": False,
            },
        ],
    }


def _jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ReviewFinalizeTests(unittest.TestCase):
    def _write_inputs(self, root: Path, record: dict, review: dict):
        ann = root / "annotations" / "conversation_0001.json"
        rev = root / "review" / "conversation_0001" / "final_reviewed_transcript.json"
        ann.parent.mkdir(parents=True, exist_ok=True)
        rev.parent.mkdir(parents=True, exist_ok=True)
        ann.write_text(json.dumps(record), encoding="utf-8")
        rev.write_text(json.dumps(review), encoding="utf-8")
        return ann, rev

    def test_completed_review_passes_and_updates_transcript_side_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = _review()
            review["segments"][0]["reviewed_text"] = "reviewed hello world"
            ann, rev = self._write_inputs(root, _record(), review)
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            updated = json.loads(ann.read_text(encoding="utf-8"))
            manifest = (root / "manifests" / "approved_for_delivery.jsonl").read_text(encoding="utf-8")
            raw = (root / "transcripts" / "conversation_0001" / "raw.txt").read_text(encoding="utf-8")
            combined = json.loads((root / "transcripts" / "conversation_0001" / "combined_conversation.json").read_text(encoding="utf-8"))

        self.assertTrue(summary["approved_for_client_delivery"])
        self.assertEqual(updated["delivery_status"]["stage"], "approved")
        self.assertEqual(updated["final_transcript_source"], "human_review")
        self.assertIn("reviewed hello world", raw)
        self.assertEqual(combined[0]["text"], "reviewed hello world")
        self.assertIn("conversation_0001", manifest)

    def test_missing_reviewed_text_fails_without_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann, rev = self._write_inputs(root, _record(), _review(blank=True))
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001", approve_if_passed=False)
            updated = json.loads(ann.read_text(encoding="utf-8"))

        self.assertFalse(summary["approved_for_client_delivery"])
        self.assertEqual(updated["delivery_status"]["stage"], "review_required")

    def test_validation_failed_blocks_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann, rev = self._write_inputs(root, _record(validation_passed=False), _review())
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            updated = json.loads(ann.read_text(encoding="utf-8"))

        self.assertFalse(summary["approved_for_client_delivery"])
        self.assertEqual(updated["delivery_status"]["stage"], "rejected")
        self.assertIn("validation_failed", updated["delivery_status"]["reason"])

    def test_segment_ids_generated_in_review_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = _record()
            record["transcript"]["segments"][0]["segment_id"] = ""
            writer = DatasetWriter(str(root))
            writer.write_review_artifacts("conversation_0001", record)
            rows = json.loads((root / "review" / "conversation_0001" / "human_review_template.json").read_text(encoding="utf-8"))

        self.assertTrue(all(row["segment_id"] for row in rows))
        self.assertEqual(rows[0]["segment_id"], "conversation_0001_seg_00001")
        self.assertIn("review_reasons", rows[0])
        self.assertIn("unresolved_issue_types", rows[0])
        self.assertNotIn("issue_types", rows[0])

    def test_code_switch_review_reason_does_not_fail_if_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = _review()
            review["segments"][0]["language"] = "Hinglish"
            review["segments"][0]["review_reasons"] = ["code_switch_detected"]
            review["segments"][0]["unresolved_issue_types"] = []
            ann, rev = self._write_inputs(root, _record(), review)
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            qa = json.loads((root / "review" / "conversation_0001" / "qa_report.json").read_text(encoding="utf-8"))

        self.assertTrue(summary["approved_for_client_delivery"])
        self.assertEqual(qa["verified_final"]["code_switch_accuracy"], 0.99)
        self.assertEqual(qa["unresolved_code_switch_count"], 0)

    def test_unresolved_code_switch_issue_blocks_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = _review()
            review["segments"][0]["language"] = "Hinglish"
            review["segments"][0]["unresolved_issue_types"] = ["code_switch"]
            ann, rev = self._write_inputs(root, _record(), review)
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")

        self.assertFalse(summary["approved_for_client_delivery"])
        self.assertIn("verified_code_switch_accuracy_below_target", summary["failure_reasons"])

    def test_legacy_issue_types_does_not_fail_code_switch_qa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = _review()
            review["segments"][0]["language"] = "Hinglish"
            review["segments"][0]["issue_types"] = ["code_switch"]
            review["segments"][0]["unresolved_issue_types"] = []
            ann, rev = self._write_inputs(root, _record(), review)
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            qa = json.loads((root / "review" / "conversation_0001" / "qa_report.json").read_text(encoding="utf-8"))

        self.assertTrue(summary["approved_for_client_delivery"])
        self.assertEqual(qa["verified_final"]["code_switch_accuracy"], 0.99)

    def test_speaker_alias_spk1_is_normalised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = _review()
            review["segments"][0]["speaker"] = "spk1"
            ann, rev = self._write_inputs(root, _record(), review)
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            updated = json.loads(ann.read_text(encoding="utf-8"))

        self.assertTrue(summary["approved_for_client_delivery"])
        self.assertEqual(updated["transcript"]["segments"][0]["speaker_id"], "SPEAKER_00")

    def test_speaker_1_maps_to_speaker_00(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = _review()
            review["segments"][0]["speaker"] = "speaker_1"
            ann, rev = self._write_inputs(root, _record(), review)
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            updated = json.loads(ann.read_text(encoding="utf-8"))

        self.assertTrue(summary["approved_for_client_delivery"])
        self.assertEqual(updated["transcript"]["segments"][0]["speaker_id"], "SPEAKER_00")

    def test_speaker_2_maps_to_speaker_01(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = _review()
            review["segments"][1]["speaker"] = "speaker_2"
            ann, rev = self._write_inputs(root, _record(), review)
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            updated = json.loads(ann.read_text(encoding="utf-8"))

        self.assertTrue(summary["approved_for_client_delivery"])
        self.assertEqual(updated["transcript"]["segments"][1]["speaker_id"], "SPEAKER_01")

    def test_speaker_01_is_ambiguous_without_speaker_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = _review()
            review["segments"][1]["speaker"] = "speaker_01"
            ann, rev = self._write_inputs(root, _record(), review)
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")

        self.assertFalse(summary["approved_for_client_delivery"])
        self.assertIn("ambiguous_speaker_alias:speaker_01", summary["failure_reasons"])

    def test_speaker_01_allowed_by_explicit_speaker_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = _record()
            record["speaker_map"] = {"SPEAKER_01": "speaker_01"}
            review = _review()
            review["segments"][1]["speaker"] = "speaker_01"
            ann, rev = self._write_inputs(root, record, review)
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            updated = json.loads(ann.read_text(encoding="utf-8"))

        self.assertTrue(summary["approved_for_client_delivery"])
        self.assertEqual(updated["transcript"]["segments"][1]["speaker_id"], "SPEAKER_01")

    def test_second_pass_required_blocks_if_not_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = _review()
            review["second_pass_review"] = {"required": True, "completed": False, "reviewer_id": None, "sample_rate": 0.0, "notes": ""}
            ann, rev = self._write_inputs(root, _record(), review)
            summary = finalize_review(str(ann), str(rev), str(root), "reviewer_001")

        self.assertFalse(summary["approved_for_client_delivery"])
        self.assertIn("second_pass_review_required_not_completed", summary["failure_reasons"])

    def test_schema_premium_condition_uses_top_level_pipeline_mode(self) -> None:
        premium = _record()
        del premium["transcript_candidates"]
        with self.assertRaises(ValidationError):
            validate_record(premium)
        offline = _record()
        offline["pipeline_mode"] = "offline_standard"
        offline.pop("transcript_candidates")
        validate_record(offline)

    def test_schema_version_consistent(self) -> None:
        self.assertEqual(_record()["schema_version"], DATASET_SCHEMA_VERSION)

    def test_accuracy_below_target_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann, rev = self._write_inputs(root, _record(), _review(shifted=True))
            summary = finalize_review(
                str(ann),
                str(rev),
                str(root),
                "reviewer_001",
                targets={"timestamp_accuracy_target": 0.99},
            )

        self.assertFalse(summary["approved_for_client_delivery"])
        self.assertIn("verified_timestamp_accuracy_below_target", summary["failure_reasons"])

    def test_no_duplicate_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann, rev = self._write_inputs(root, _record(), _review())
            finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            rows = [
                json.loads(line)
                for line in (root / "manifests" / "final_transcripts.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session"], "conversation_0001")

    def test_human_review_state_semantics_for_approved_and_failed_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann, rev = self._write_inputs(root, _record(), _review())
            finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            approved = json.loads(ann.read_text(encoding="utf-8"))

            ann, rev = self._write_inputs(root, _record(), _review(blank=True))
            finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            failed = json.loads(ann.read_text(encoding="utf-8"))

        self.assertTrue(approved["accuracy_gate"]["human_review_required"])
        self.assertTrue(approved["accuracy_gate"]["human_review_completed"])
        self.assertFalse(approved["accuracy_gate"]["human_review_required_for_delivery"])
        self.assertTrue(failed["accuracy_gate"]["human_review_required_for_delivery"])

    def test_status_manifests_remove_stale_rows_when_rejected_then_approved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rejected_review = _review()
            rejected_review["segments"][1]["speaker"] = "speaker_01"
            ann, rev = self._write_inputs(root, _record(), rejected_review)
            finalize_review(str(ann), str(rev), str(root), "reviewer_001")

            rev.write_text(json.dumps(_review()), encoding="utf-8")
            finalize_review(str(ann), str(rev), str(root), "reviewer_001")

            approved_rows = _jsonl_rows(root / "manifests" / "approved_for_delivery.jsonl")
            rejected_rows = _jsonl_rows(root / "manifests" / "rejected.jsonl")

        self.assertEqual([row["session"] for row in approved_rows], ["conversation_0001"])
        self.assertNotIn("conversation_0001", [row["session"] for row in rejected_rows])

    def test_status_manifests_remove_stale_rows_when_review_required_then_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann, rev = self._write_inputs(root, _record(), _review(blank=True))
            finalize_review(str(ann), str(rev), str(root), "reviewer_001", approve_if_passed=False)

            rejected_review = _review()
            rejected_review["segments"][1]["speaker"] = "speaker_01"
            rev.write_text(json.dumps(rejected_review), encoding="utf-8")
            finalize_review(str(ann), str(rev), str(root), "reviewer_001")

            manifests = root / "manifests"
            approved_rows = _jsonl_rows(manifests / "approved_for_delivery.jsonl")
            review_rows = _jsonl_rows(manifests / "review_required.jsonl")
            rejected_rows = _jsonl_rows(manifests / "rejected.jsonl")

        self.assertNotIn("conversation_0001", [row["session"] for row in approved_rows])
        self.assertNotIn("conversation_0001", [row["session"] for row in review_rows])
        self.assertEqual([row["session"] for row in rejected_rows], ["conversation_0001"])

    def test_manifest_uses_delivery_oriented_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann, rev = self._write_inputs(root, _record(), _review())
            finalize_review(str(ann), str(rev), str(root), "reviewer_001")
            approved_row = _jsonl_rows(root / "manifests" / "approved_for_delivery.jsonl")[0]

            ann, rev = self._write_inputs(root, _record(), _review(blank=True))
            finalize_review(str(ann), str(rev), str(root), "reviewer_001", approve_if_passed=False)
            review_required_row = _jsonl_rows(root / "manifests" / "review_required.jsonl")[0]

        self.assertEqual(
            approved_row["canonical_transcript_path"],
            str((root / "transcripts" / "conversation_0001" / "combined_conversation.json").resolve()),
        )
        self.assertEqual(approved_row["transcript_path"], approved_row["canonical_transcript_path"])
        self.assertEqual(
            approved_row["reviewed_transcript_input_path"],
            str((root / "review" / "conversation_0001" / "final_reviewed_transcript.json").resolve()),
        )
        self.assertEqual(
            approved_row["qa_report_path"],
            str((root / "review" / "conversation_0001" / "qa_report.json").resolve()),
        )
        self.assertIsNone(review_required_row["canonical_transcript_path"])
        self.assertIsNone(review_required_row["raw_transcript_path"])
        self.assertIsNone(review_required_row["normalised_transcript_path"])
        self.assertIsNone(review_required_row["flat_conversation_path"])

    def test_timestamp_accuracy_can_reach_one_and_gate_on_measured_value(self) -> None:
        original = _record()["transcript"]["segments"]
        perfect = _review()["segments"]
        shifted = _review(shifted=True)["segments"]

        self.assertEqual(timestamp_accuracy(perfect, original), 1.0)
        self.assertLess(timestamp_accuracy(shifted, original), 0.99)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann, rev = self._write_inputs(root, _record(), _review())
            approved = finalize_review(
                str(ann),
                str(rev),
                str(root),
                "reviewer_001",
                targets={"timestamp_accuracy_target": 0.99},
            )

            ann, rev = self._write_inputs(root, _record(), _review(shifted=True))
            rejected = finalize_review(
                str(ann),
                str(rev),
                str(root),
                "reviewer_001",
                targets={"timestamp_accuracy_target": 0.99},
            )

        self.assertTrue(approved["approved_for_client_delivery"])
        self.assertFalse(rejected["approved_for_client_delivery"])
        self.assertIn("verified_timestamp_accuracy_below_target", rejected["failure_reasons"])

    def test_schema_validation_failure_leaves_no_partial_finalisation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_record = _record()
            review = _review()
            review["segments"][0]["segment_id"] = "conversation_0001_seg_missing"
            ann, rev = self._write_inputs(root, original_record, review)
            before = ann.read_text(encoding="utf-8")

            with self.assertRaises(ValidationError):
                finalize_review(str(ann), str(rev), str(root), "reviewer_001")

            self.assertEqual(ann.read_text(encoding="utf-8"), before)
            self.assertFalse((root / "review" / "conversation_0001" / "qa_report.json").exists())
            self.assertFalse((root / "transcripts" / "conversation_0001" / "raw.txt").exists())
            self.assertFalse((root / "transcripts" / "conversation_0001" / "combined_conversation.json").exists())
            self.assertFalse((root / "manifests").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
