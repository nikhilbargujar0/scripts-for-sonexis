#!/usr/bin/env python3
"""CLI for finalising human-reviewed Sonexis transcripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.review.finalize import finalize_review


def _bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {v!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finalize human-reviewed Sonexis transcript QA")
    p.add_argument("--annotation", required=True)
    p.add_argument("--reviewed_transcript", "--reviewed-transcript", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reviewer_id", "--reviewer-id", required=True)
    p.add_argument("--approve_if_passed", "--approve-if-passed", type=_bool, default=True)
    p.add_argument("--word_accuracy_target", "--word-accuracy-target", type=float, default=None)
    p.add_argument("--speaker_accuracy_target", "--speaker-accuracy-target", type=float, default=None)
    p.add_argument("--timestamp_accuracy_target", "--timestamp-accuracy-target", type=float, default=None)
    p.add_argument("--code_switch_accuracy_target", "--code-switch-accuracy-target", type=float, default=None)
    p.add_argument("--fail_fast", "--fail-fast", type=_bool, default=False)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = {
        k: v for k, v in {
            "word_accuracy_target": args.word_accuracy_target,
            "speaker_accuracy_target": args.speaker_accuracy_target,
            "timestamp_accuracy_target": args.timestamp_accuracy_target,
            "code_switch_accuracy_target": args.code_switch_accuracy_target,
        }.items()
        if v is not None
    }
    summary = finalize_review(
        annotation_path=args.annotation,
        reviewed_transcript_path=args.reviewed_transcript,
        output_root=args.output,
        reviewer_id=args.reviewer_id,
        approve_if_passed=args.approve_if_passed,
        targets=targets or None,
    )
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_fast and not summary.get("approved_for_client_delivery"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
