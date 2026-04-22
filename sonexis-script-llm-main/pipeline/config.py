"""config.py

PipelineConfig: single configuration object for the entire pipeline.

Replaces the ad-hoc argparse.Namespace passed around in main.py.
Can be constructed from:
  - kwargs (direct instantiation)
  - a plain dict (e.g. loaded from YAML)
  - argparse.Namespace (from CLI)
  - load_config() — parses sys.argv / explicit argv, optionally merging a YAML file

Choosing output_mode:
  "both"             → write speaker_separated WAVs + mono mixed (recommended)
  "speaker_separated"→ write only per-speaker WAVs
  "mono"             → write only the mixed mono WAV

Choosing input_type:
  "auto"     → detect from folder structure (default)
  "speaker_pair" → two per-speaker files per session
  "stereo"   → one stereo WAV per session (left=spk1, right=spk2)
  "mono"     → single mixed file (existing mono pipeline)
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class PipelineConfig:
    # ── input ─────────────────────────────────────────────────────────
    input_type: str = "auto"          # auto | speaker_pair | stereo | mono

    # ── output ────────────────────────────────────────────────────────
    output_mode: str = "both"         # both | speaker_separated | mono
    output_format: str = "json"       # json | jsonl | parquet (annotation format)
    dataset_name: str = "dataset"

    # ── audio processing ──────────────────────────────────────────────
    sample_rate_target: int = 16_000
    denoise: bool = False

    # ── ASR ───────────────────────────────────────────────────────────
    model_size: str = "small"         # tiny | base | small | medium | large-v3
    compute_type: str = "int8"
    device: str = "cpu"               # cpu | cuda | auto
    language: Optional[str] = None   # None = auto-detect

    # ── VAD ───────────────────────────────────────────────────────────
    vad_backend: str = "webrtc"       # webrtc | silero

    # ── diarisation (mono path only; speaker_pair uses ground-truth) ──
    diarisation_backend: str = "kmeans"   # kmeans | pyannote
    max_speakers: int = 4
    min_speakers: int = 1
    hf_token: Optional[str] = None

    # ── offline mode ──────────────────────────────────────────────────
    offline_mode: bool = True
    model_dir: Optional[str] = None

    # ── language classifier ───────────────────────────────────────────
    classifier: str = "auto"          # auto | on | off
    classifier_cache: Optional[str] = None
    fasttext_model: Optional[str] = None

    # ── quality thresholds ────────────────────────────────────────────
    max_duration_mismatch_s: float = 60.0  # warn above this
    min_audio_duration_s: float = 1.0

    # ── I/O paths (populated by load_config / run_pipeline) ──────────
    input_dir: str = ""
    output_dir: str = ""

    # ── misc ──────────────────────────────────────────────────────────
    fail_fast: bool = False
    verbose: bool = False

    # ── interaction metadata ──────────────────────────────────────────
    interruption_threshold_s: float = 0.5  # overlaps shorter than this = interruption

    # ── alignment ─────────────────────────────────────────────────────
    alignment_enabled: bool = True          # run cross-correlation alignment for speaker_pair
    alignment_max_offset_s: float = 5.0    # max plausible clock drift to search
    alignment_min_confidence: float = 0.10  # below this → skip alignment, log warning

    # ── performance / speed ───────────────────────────────────────────
    # num_workers: parallel session workers (>1 → ProcessPoolExecutor)
    # Each worker loads its own Whisper model; best for batches of files.
    num_workers: int = 1
    # ASR beam size: 5 = default quality, 1 = greedy (~2× faster, mild quality drop)
    beam_size: int = 5
    # asr_batched: use BatchedInferencePipeline (faster-whisper >= 1.1, ~2-3× faster)
    asr_batched: bool = False
    asr_batch_size: int = 16
    # cpu_threads: 0 = all cores; set to N to limit Whisper thread usage
    asr_cpu_threads: int = 0
    # skip_sha1: skip SHA1 hash of audio files (saves I/O on large files)
    skip_sha1: bool = False
    # Global deterministic seed for clustering and any future stochastic helpers.
    random_seed: int = 0
    # Stable default keeps JSON records deterministic unless the user overrides it.
    generated_at: str = "1970-01-01T00:00:00+00:00"
    include_runtime_metrics: bool = False
    metadata_file: Optional[str] = None
    ask_metadata: bool = True
    dialect: Optional[str] = None
    region: Optional[str] = None
    gender: Optional[str] = None
    age_band: Optional[str] = None
    recording_context: Optional[str] = None
    consent_status: Optional[str] = None
    user_metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_namespace(cls, ns) -> "PipelineConfig":
        """Build from argparse.Namespace (handles kebab→snake rename)."""
        d = {k.replace("-", "_"): v for k, v in vars(ns).items()}
        # argparse historically exposed --input-mode as input_mode while the
        # config field is input_type. Keep both names working.
        if "input_mode" in d and "input_type" not in d:
            d["input_type"] = d["input_mode"]
        if "interruption_threshold" in d and "interruption_threshold_s" not in d:
            d["interruption_threshold_s"] = d["interruption_threshold"]
        # offline_mode may arrive as string "true"/"false" from CLI
        if "offline_mode" in d and isinstance(d["offline_mode"], str):
            d["offline_mode"] = d["offline_mode"].lower() in ("true", "1", "yes")
        # Populate I/O paths from argparse "input" / "output" keys.
        if "input" in d:
            d.setdefault("input_dir", d["input"])
        if "output" in d:
            d.setdefault("output_dir", d["output"])
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
#  Public loader — the recommended programmatic entry point
# ---------------------------------------------------------------------------

def load_config(
    argv: Optional[List[str]] = None,
    yaml_path: Optional[str] = None,
) -> PipelineConfig:
    """Parse CLI arguments and/or a YAML config file and return PipelineConfig.

    Priority (highest wins):
        1. Explicit CLI arguments supplied via *argv* or sys.argv[1:]
        2. Values from *yaml_path* (or ``--config`` CLI flag)
        3. PipelineConfig defaults

    ``config.input_dir`` and ``config.output_dir`` are populated from
    ``--input`` / ``--output`` so that ``run_pipeline(config)`` works without
    passing extra arguments::

        config = load_config()
        run_pipeline(config)

    Parameters
    ----------
    argv:
        Explicit argument list. Defaults to ``sys.argv[1:]``.
    yaml_path:
        Path to a YAML file whose keys map to PipelineConfig fields.
        Also honoured via ``--config <path>`` in *argv*.
    """
    import argparse as _argparse

    _argv: List[str] = list(argv) if argv is not None else sys.argv[1:]

    # ── Step 1: extract --config before the full parse ──────────────
    _pre = _argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args(_argv)
    _yaml_path = yaml_path or _pre_args.config

    # ── Step 2: load YAML base (lowest priority) ─────────────────────
    yaml_base: dict = {}
    if _yaml_path:
        try:
            import yaml as _yaml  # optional dependency
            with open(_yaml_path, "r", encoding="utf-8") as _f:
                yaml_base = _yaml.safe_load(_f) or {}
        except ImportError:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "PyYAML not installed; ignoring --config / yaml_path"
            )
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning("load_config: YAML load failed: %s", _e)

    # ── Step 3: build base config from YAML ──────────────────────────
    cfg = PipelineConfig.from_dict(yaml_base)

    # ── Step 4: full CLI parse (deferred import avoids circular) ─────
    # pipeline.main imports pipeline.config, so we defer the import to
    # *function call time* (not module import time) to break the cycle.
    try:
        from pipeline.main import _parse_args  # type: ignore[attr-defined]
    except ImportError:
        # Fallback for editable-install / different working directory.
        from .main import _parse_args  # type: ignore[attr-defined]

    _full_args = _parse_args(_argv)
    _cli_cfg = PipelineConfig.from_namespace(_full_args)

    # Merge: CLI values override YAML defaults for fields that differ from default.
    _defaults = PipelineConfig()
    for _field in PipelineConfig.__dataclass_fields__:
        _cli_val = getattr(_cli_cfg, _field)
        _default_val = getattr(_defaults, _field)
        if _cli_val != _default_val:
            setattr(cfg, _field, _cli_val)

    # Always take input_dir / output_dir from CLI if provided.
    if _cli_cfg.input_dir:
        cfg.input_dir = _cli_cfg.input_dir
    if _cli_cfg.output_dir:
        cfg.output_dir = _cli_cfg.output_dir

    return cfg
