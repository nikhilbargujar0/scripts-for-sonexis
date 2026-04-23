#!/usr/bin/env python3
"""CLI only. Processing logic lives in pipeline.runner and pipeline.steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.runner import process_conversation


def _bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sonexis offline conversational dataset pipeline")
    p.add_argument("--input", required=True, help="conversation folder or audio file")
    p.add_argument("--output", required=True, help="dataset output directory")
    p.add_argument("--input_type", "--input-type", default="auto",
                   choices=["auto", "separate", "speaker_pair", "stereo", "mono"])
    p.add_argument("--output_mode", "--output-mode", default="both",
                   choices=["speaker_separated", "mono", "both"])
    p.add_argument("--offline_mode", "--offline-mode", default="true")
    p.add_argument("--model_dir", "--model-dir", default=None)
    p.add_argument("--model_size", "--model-size", default="small")
    p.add_argument("--compute_type", "--compute-type", default="int8")
    p.add_argument("--device", default="cpu")
    p.add_argument("--language", default=None)
    p.add_argument("--metadata_depth", "--metadata-depth", default="full", choices=["basic", "full"])
    p.add_argument("--enable_monologue_extraction", "--enable-monologue-extraction", default="true")
    p.add_argument("--random_seed", "--random-seed", type=int, default=0)
    p.add_argument("--fail_fast", "--fail-fast", default="false")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_type = "speaker_pair" if args.input_type == "separate" else args.input_type
    cfg = PipelineConfig(
        input_type=input_type,
        output_mode=args.output_mode,
        offline_mode=_bool(args.offline_mode),
        model_dir=args.model_dir,
        model_size=args.model_size,
        compute_type=args.compute_type,
        device=args.device,
        language=args.language,
        metadata_depth=args.metadata_depth,
        enable_monologue_extraction=_bool(args.enable_monologue_extraction),
        random_seed=args.random_seed,
        fail_fast=_bool(args.fail_fast),
        ask_metadata=False,
    )
    result = process_conversation(args.input, args.output, cfg)
    summary = {
        "output_path": result["output_path"],
        "records": len(result.get("records", [])),
        "validation_reports": result.get("validation_reports", []),
        "downloads": result.get("downloads", {}),
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
