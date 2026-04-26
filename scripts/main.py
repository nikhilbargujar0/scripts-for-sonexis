#!/usr/bin/env python3
"""CLI only. Processing logic lives in pipeline.runner and pipeline.steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import PipelineConfig, _detect_device
from pipeline.runner import process_conversation


def _bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def _load_premium_config(path: str | None) -> dict:
    if not path:
        return {}
    content = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(content)
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("YAML premium config requires PyYAML or use a JSON file instead.") from exc
    return yaml.safe_load(content) or {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sonexis offline conversational dataset pipeline")
    p.add_argument("--input", required=True, help="conversation folder or audio file")
    p.add_argument("--output", required=True, help="dataset output directory")
    p.add_argument("--input_type", "--input-type", default="auto",
                   choices=["auto", "separate", "speaker_pair", "stereo", "mono"])
    p.add_argument("--output_mode", "--output-mode", default="both",
                   choices=["speaker_separated", "mono", "both"])
    p.add_argument("--colab", default="false",
                   help="Shorthand for Colab: sets offline_mode=false, device=auto, "
                        "compute_type=float16, ask_metadata=false. "
                        "Individual flags still override these defaults.")
    p.add_argument("--offline_mode", "--offline-mode", default=None,
                   help="default: true normally, false when --colab is set")
    p.add_argument("--model_dir", "--model-dir", default=None)
    p.add_argument("--model_size", "--model-size", default="small")
    p.add_argument("--compute_type", "--compute-type", default=None,
                   help="default: int8 (cpu) or float16 (cuda). "
                        "Pass 'auto' or omit to let the pipeline choose.")
    p.add_argument("--device", default=None,
                   help="cpu | cuda | auto  (default: cpu, or auto when --colab)")
    p.add_argument("--language", default=None)
    p.add_argument("--metadata_file", "--metadata-file", default=None)
    p.add_argument("--accent", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--dialect", default=None)
    p.add_argument("--domain", default=None)
    p.add_argument("--metadata_depth", "--metadata-depth", default="full", choices=["basic", "full"])
    p.add_argument("--enable_monologue_extraction", "--enable-monologue-extraction", default="true")
    p.add_argument("--alignment_min_confidence", "--alignment-min-confidence", type=float, default=0.35)
    p.add_argument("--pair_merge_gap_s", "--pair-merge-gap-s", type=float, default=0.15)
    p.add_argument("--pair_min_turn_duration_s", "--pair-min-turn-duration-s", type=float, default=0.08)
    p.add_argument("--random_seed", "--random-seed", type=int, default=0)
    p.add_argument("--fail_fast", "--fail-fast", default="false")
    p.add_argument("--beam_size", "--beam-size", type=int, default=5,
                   help="Whisper beam size (5=default quality, 1=greedy/fast)")
    p.add_argument("--asr_batched", "--asr-batched", default="false",
                   help="Use BatchedInferencePipeline for ~2-3x faster decoding on GPU")
    p.add_argument("--denoise", default="false",
                   help="Apply noise reduction before ASR (helps with background noise)")
    p.add_argument("--initial_prompt", "--initial-prompt", default=None,
                   help="Whisper conditioning prompt. Auto-set for hi/ta/te/mr/bn/gu/pa.")
    p.add_argument("--no_speech_threshold", "--no-speech-threshold", type=float, default=0.6,
                   help="Whisper no-speech probability cutoff (0-1, default 0.6)")
    p.add_argument("--compression_ratio_threshold", "--compression-ratio-threshold",
                   type=float, default=2.4,
                   help="Whisper compression ratio threshold (default 2.4; raise for code-switch)")
    p.add_argument("--log_prob_threshold", "--log-prob-threshold", type=float, default=-1.0,
                   help="Whisper avg log-prob floor (default -1.0)")
    p.add_argument("--condition_on_previous_text", "--condition-on-previous-text",
                   default="false",
                   help="Feed previous segment text as context (default false)")
    p.add_argument("--quality_score_threshold", "--quality-score-threshold",
                   type=float, default=0.35,
                   help="Segment quality score below this = low quality (default 0.35)")
    p.add_argument("--pipeline_mode", "--pipeline-mode", default="offline_standard",
                   choices=["offline_standard", "premium_accuracy"])
    p.add_argument("--allow_paid_apis", "--allow-paid-apis", default="false")
    p.add_argument("--premium_config", "--premium-config", default=None)
    p.add_argument("--require_human_review", "--require-human-review", default="true")
    p.add_argument("--export_products", "--export-products", default="stt,diarisation,evaluation_gold")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_type = "speaker_pair" if args.input_type == "separate" else args.input_type
    premium_cfg = _load_premium_config(args.premium_config)

    # --colab sets sensible Colab defaults; individual flags still win.
    colab = _bool(args.colab)
    offline_mode = _bool(args.offline_mode) if args.offline_mode is not None else (not colab)
    device = args.device if args.device is not None else ("auto" if colab else "cpu")
    # Resolve "auto" device now so compute_type default can react to it.
    resolved_device = _detect_device() if device == "auto" else device
    if args.compute_type is not None and args.compute_type != "auto":
        compute_type = args.compute_type
    else:
        compute_type = "float16" if resolved_device == "cuda" else "int8"

    cfg = PipelineConfig(
        input_type=input_type,
        output_mode=args.output_mode,
        offline_mode=offline_mode,
        model_dir=args.model_dir,
        model_size=args.model_size,
        compute_type=compute_type,
        device=resolved_device,
        language=args.language,
        metadata_file=args.metadata_file,
        accent=args.accent,
        region=args.region,
        dialect=args.dialect,
        domain=args.domain,
        metadata_depth=args.metadata_depth,
        enable_monologue_extraction=_bool(args.enable_monologue_extraction),
        alignment_min_confidence=args.alignment_min_confidence,
        pair_merge_gap_s=args.pair_merge_gap_s,
        pair_min_turn_duration_s=args.pair_min_turn_duration_s,
        random_seed=args.random_seed,
        fail_fast=_bool(args.fail_fast),
        beam_size=args.beam_size,
        asr_batched=_bool(args.asr_batched),
        denoise=_bool(args.denoise),
        initial_prompt=args.initial_prompt,
        no_speech_threshold=args.no_speech_threshold,
        compression_ratio_threshold=args.compression_ratio_threshold,
        log_prob_threshold=args.log_prob_threshold,
        condition_on_previous_text=_bool(args.condition_on_previous_text),
        quality_score_threshold=args.quality_score_threshold,
        ask_metadata=False,
        pipeline_mode=args.pipeline_mode,
        allow_paid_apis=_bool(args.allow_paid_apis),
        require_human_review=_bool(args.require_human_review),
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
