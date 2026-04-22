"""pipeline/runner.py

Production entry point for the Sonexis conversational-audio pipeline.

Recommended usage::

    from pipeline.config import load_config
    from pipeline.runner import run_pipeline

    config = load_config()          # reads CLI args / YAML
    exit_code = run_pipeline(config)

``run_pipeline`` validates that ``config.input_dir`` and
``config.output_dir`` are set, enforces offline-mode model path checks,
seeds the RNG for determinism, and then delegates to the battle-tested
``pipeline.main.run`` implementation.

Offline mode enforcement
------------------------
When ``config.offline_mode = True`` (the default), this function checks
that the required model directories exist **before** starting any session
processing.  A missing model raises :class:`pipeline.offline.OfflineModeError`
immediately instead of failing silently mid-batch.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("sonexis")


# ── offline pre-flight check ───────────────────────────────────────────────

def _check_offline_models(cfg: PipelineConfig) -> None:
    """Raise OfflineModeError if any required model is missing.

    Only checked when ``cfg.offline_mode = True``.  Skipped when
    ``cfg.model_dir`` is not set (legacy behaviour: let downstream modules
    fail with their own error messages).
    """
    if not cfg.offline_mode:
        return
    model_dir = cfg.model_dir
    if not model_dir:
        return

    from .offline import (
        OfflineModeError,
        fasttext_local_path,
        require_model_dir,
        whisper_local_path,
    )

    # Whisper model directory
    whisper_path = whisper_local_path(model_dir, cfg.model_size)
    require_model_dir(
        whisper_path,
        label=f"whisper/{cfg.model_size}",
        download_hint="python download_models.py --whisper",
    )

    # FastText language ID (optional — only required if classifier is not "off")
    if cfg.classifier != "off" and cfg.fasttext_model is None:
        ft_path = fasttext_local_path(model_dir)
        if not os.path.isfile(ft_path):
            log.warning(
                "[offline_mode] FastText model not found at %s — "
                "language detection will use lexicon heuristics only.",
                ft_path,
            )

    log.info("offline pre-flight: all required models present under %s", model_dir)


# ── argv builder ───────────────────────────────────────────────────────────

def _config_to_argv(cfg: PipelineConfig) -> List[str]:
    """Convert a :class:`PipelineConfig` to a ``pipeline.main.run`` argv list.

    Only fields that deviate from the default *or* are required (input/output)
    are emitted, keeping the argv short and debuggable.
    """
    defaults = PipelineConfig()
    argv: List[str] = [
        "--input",  cfg.input_dir,
        "--output", cfg.output_dir,
    ]

    def _add(flag: str, value, default=None):
        if default is None or value != default:
            argv.extend([flag, str(value)])

    def _add_bool(flag: str, value: bool, default: bool = False):
        if value and not default:
            argv.append(flag)

    _add("--input-mode",      cfg.input_type,      defaults.input_type)
    _add("--output-mode",     cfg.output_mode,     defaults.output_mode)
    _add("--output-format",   cfg.output_format,   defaults.output_format)
    _add("--dataset-name",    cfg.dataset_name,    defaults.dataset_name)
    _add("--model-size",      cfg.model_size,      defaults.model_size)
    _add("--compute-type",    cfg.compute_type,    defaults.compute_type)
    _add("--device",          cfg.device,          defaults.device)
    _add("--vad-backend",     cfg.vad_backend,     defaults.vad_backend)
    _add("--diarisation-backend", cfg.diarisation_backend, defaults.diarisation_backend)
    _add("--max-speakers",    cfg.max_speakers,    defaults.max_speakers)
    _add("--min-speakers",    cfg.min_speakers,    defaults.min_speakers)
    _add("--offline-mode",    str(cfg.offline_mode).lower(),
         str(defaults.offline_mode).lower())
    _add("--interruption-threshold", cfg.interruption_threshold_s,
         defaults.interruption_threshold_s)
    _add("--num-workers",     cfg.num_workers,     defaults.num_workers)
    _add("--beam-size",       cfg.beam_size,       defaults.beam_size)
    _add("--asr-batch-size",  cfg.asr_batch_size,  defaults.asr_batch_size)
    _add("--asr-cpu-threads", cfg.asr_cpu_threads, defaults.asr_cpu_threads)
    _add("--random-seed",     cfg.random_seed,     defaults.random_seed)
    _add("--generated-at",    cfg.generated_at,    defaults.generated_at)

    if cfg.language:
        argv.extend(["--language", cfg.language])
    if cfg.model_dir:
        argv.extend(["--model-dir", cfg.model_dir])
    if cfg.fasttext_model:
        argv.extend(["--fasttext-model", cfg.fasttext_model])
    if cfg.classifier and cfg.classifier != defaults.classifier:
        argv.extend(["--classifier", cfg.classifier])
    if cfg.classifier_cache:
        argv.extend(["--classifier-cache", cfg.classifier_cache])
    if cfg.hf_token:
        argv.extend(["--hf-token", cfg.hf_token])
    if cfg.metadata_file:
        argv.extend(["--metadata-file", cfg.metadata_file])

    _add_bool("--denoise",                cfg.denoise)
    _add_bool("--verbose",                cfg.verbose)
    _add_bool("--fail-fast",              cfg.fail_fast)
    _add_bool("--skip-sha1",              cfg.skip_sha1)
    _add_bool("--asr-batched",            cfg.asr_batched)
    _add_bool("--include-runtime-metrics", cfg.include_runtime_metrics)

    if not cfg.ask_metadata:
        argv.append("--no-ask-metadata")

    for field_name in ("dialect", "region", "gender", "age_band",
                       "recording_context", "consent_status"):
        value: Optional[str] = getattr(cfg, field_name, None)
        if value:
            argv.extend([f"--{field_name.replace('_', '-')}", value])

    return argv


# ── public entry point ─────────────────────────────────────────────────────

def run_pipeline(config: PipelineConfig) -> int:
    """Run the Sonexis pipeline with a fully-populated :class:`PipelineConfig`.

    Parameters
    ----------
    config :
        Must have ``input_dir`` and ``output_dir`` set.  All other fields
        default to the values documented on :class:`PipelineConfig`.

    Returns
    -------
    int
        ``0`` — all sessions processed successfully.
        ``1`` — one or more sessions failed (partial output written).
        ``2`` — fatal error (no sessions found, missing model, etc.).

    Raises
    ------
    ValueError
        If ``config.input_dir`` or ``config.output_dir`` is empty.
    pipeline.offline.OfflineModeError
        If ``offline_mode=True`` and a required model directory is absent.
    """
    if not config.input_dir:
        raise ValueError(
            "config.input_dir must be set before calling run_pipeline().  "
            "Use load_config() or set it explicitly."
        )
    if not config.output_dir:
        raise ValueError(
            "config.output_dir must be set before calling run_pipeline().  "
            "Use load_config() or set it explicitly."
        )

    # Determinism: seed RNG before any pipeline work.
    np.random.seed(config.random_seed)

    # Offline model pre-flight (raises OfflineModeError if models missing).
    _check_offline_models(config)

    # Delegate to the production run() function.
    from .main import run  # deferred to avoid circular import at module level
    argv = _config_to_argv(config)
    return run(argv)
