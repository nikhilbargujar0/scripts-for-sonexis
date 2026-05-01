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
    value = str(v).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {v!r}")


def _load_premium_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    content = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(content)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("YAML premium config requires PyYAML or use a JSON file instead.") from exc
        loaded = yaml.safe_load(content) or {}
    if not isinstance(loaded, dict):
        raise ValueError("premium config must be a JSON/YAML object")
    return loaded


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sonexis offline conversational dataset pipeline")
    p.add_argument("--input", required=True, help="conversation folder or audio file")
    p.add_argument("--output", required=True, help="dataset output directory")
    p.add_argument("--input_type", "--input-type", default="auto",
                   choices=["auto", "separate", "speaker_pair", "stereo", "mono"])
    p.add_argument("--output_mode", "--output-mode", default="both",
                   choices=["speaker_separated", "mono", "both"])
    p.add_argument("--output_format", "--output-format", default="json",
                   choices=["json", "jsonl", "parquet"])
    p.add_argument("--offline_mode", "--offline-mode", type=_bool, default=True)
    p.add_argument("--model_dir", "--model-dir", default=None)
    p.add_argument("--model_size", "--model-size", default="small")
    p.add_argument("--compute_type", "--compute-type", default="int8")
    p.add_argument("--device", default="cpu")
    p.add_argument("--language", default=None)
    p.add_argument("--metadata_file", "--metadata-file", default=None)
    p.add_argument("--accent", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--dialect", default=None)
    p.add_argument("--domain", default=None)
    p.add_argument("--metadata_depth", "--metadata-depth", default="full", choices=["basic", "full"])
    p.add_argument("--enable_monologue_extraction", "--enable-monologue-extraction", type=_bool, default=True)
    p.add_argument("--alignment_min_confidence", "--alignment-min-confidence", type=float, default=0.35)
    p.add_argument("--pair_merge_gap_s", "--pair-merge-gap-s", type=float, default=0.15)
    p.add_argument("--pair_min_turn_duration_s", "--pair-min-turn-duration-s", type=float, default=0.08)
    p.add_argument("--random_seed", "--random-seed", type=int, default=0)
    p.add_argument("--fail_fast", "--fail-fast", type=_bool, default=False)
    p.add_argument("--pipeline_mode", "--pipeline-mode", default="offline_standard",
                   choices=["offline_standard", "premium_accuracy"])
    p.add_argument("--allow_paid_apis", "--allow-paid-apis", type=_bool, default=False)
    p.add_argument("--premium_config", "--premium-config", default=None)
    p.add_argument("--require_human_review", "--require-human-review", type=_bool, default=True)
    p.add_argument("--export_products", "--export-products", default="stt,diarisation,evaluation_gold")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_type = "speaker_pair" if args.input_type == "separate" else args.input_type
    premium_cfg = _load_premium_config(args.premium_config)
    cfg = PipelineConfig(
        input_type=input_type,
        output_mode=args.output_mode,
        output_format=args.output_format,
        offline_mode=args.offline_mode,
        model_dir=args.model_dir,
        model_size=args.model_size,
        compute_type=args.compute_type,
        device=args.device,
        language=args.language,
        metadata_file=args.metadata_file,
        accent=args.accent,
        region=args.region,
        dialect=args.dialect,
        domain=args.domain,
        metadata_depth=args.metadata_depth,
        enable_monologue_extraction=args.enable_monologue_extraction,
        alignment_min_confidence=args.alignment_min_confidence,
        pair_merge_gap_s=args.pair_merge_gap_s,
        pair_min_turn_duration_s=args.pair_min_turn_duration_s,
        random_seed=args.random_seed,
        fail_fast=args.fail_fast,
        ask_metadata=False,
        pipeline_mode=args.pipeline_mode,
        allow_paid_apis=args.allow_paid_apis,
        require_human_review=args.require_human_review,
        export_products=[item.strip() for item in args.export_products.split(",") if item.strip()],
        premium=premium_cfg or PipelineConfig().premium,
    )
    if cfg.pipeline_mode == "premium_accuracy":
        cfg.premium["enabled"] = True
        cfg.premium["allow_paid_apis"] = bool(cfg.allow_paid_apis)
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
