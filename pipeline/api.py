"""Callable backend API for internal script integrations.

Internal scripts can call ``process_conversation(input_path, output_path,
config)`` and receive structured paths, records, logs, and previews.
"""
from __future__ import annotations

import io
import logging
import os
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .audio_loader import (
    detect_and_group_pairs,
    detect_stereo_files,
    iter_audio_files,
    load_audio,
    load_speaker_pair,
    load_stereo_as_pair,
)
from .batch_writer import BatchWriter
from .config import PipelineConfig
from .dataset_writer import DatasetWriter
from .language_detection import FastTextLID
from .main import (
    _apply_user_metadata,
    _build_asr_cfg,
    _load_user_metadata_file,
    _normalise_user_metadata_block,
    _process_single,
    _process_speaker_pair,
)
from .offline import default_model_dir, fasttext_local_path
from .roman_indic_classifier import RomanIndicClassifier
from .transcription import Transcriber

ProgressFn = Optional[Callable[[str], None]]


class ProcessingError(RuntimeError):
    """Raised when input validation or processing fails cleanly."""


def _log_progress(progress: ProgressFn, message: str) -> None:
    if progress:
        progress(message)
    logging.getLogger("sonexis.api").info(message)


def _normalise_config(config: PipelineConfig | Dict | None) -> PipelineConfig:
    if isinstance(config, PipelineConfig):
        cfg = PipelineConfig.from_dict(config.to_dict())
    elif isinstance(config, dict):
        cfg = PipelineConfig.from_dict(config)
    elif config is None:
        cfg = PipelineConfig()
    else:
        raise TypeError("config must be a PipelineConfig, dict, or None")

    cfg.ask_metadata = False
    if cfg.metadata_depth not in ("basic", "full"):
        raise ValueError("metadata_depth must be 'basic' or 'full'")
    if cfg.metadata_file and not cfg.user_metadata:
        cfg.user_metadata = _load_user_metadata_file(cfg.metadata_file)
    cli_values = {
        field: getattr(cfg, field)
        for field in ("dialect", "region", "gender", "age_band", "recording_context", "consent_status")
    }
    cli_block = _normalise_user_metadata_block(cli_values, "api_config")
    if cli_block:
        cfg.user_metadata.setdefault("*", {}).update(cli_block)
    return cfg


def _model_cache_key(cfg: PipelineConfig, model_dir: Optional[str]) -> Tuple:
    return (
        cfg.model_size,
        cfg.compute_type,
        cfg.device,
        cfg.language,
        cfg.offline_mode,
        model_dir,
        cfg.beam_size,
        cfg.asr_batched,
        cfg.asr_batch_size,
        cfg.asr_cpu_threads,
        cfg.classifier,
        cfg.classifier_cache,
        cfg.fasttext_model,
    )


@lru_cache(maxsize=4)
def _load_models_cached(key: Tuple) -> Tuple[Transcriber, FastTextLID, Optional[RomanIndicClassifier]]:
    (
        model_size,
        compute_type,
        device,
        language,
        offline_mode,
        model_dir,
        beam_size,
        asr_batched,
        asr_batch_size,
        asr_cpu_threads,
        classifier_mode,
        classifier_cache,
        fasttext_model,
    ) = key

    cfg = PipelineConfig(
        model_size=model_size,
        compute_type=compute_type,
        device=device,
        language=language,
        offline_mode=offline_mode,
        model_dir=model_dir,
        beam_size=beam_size,
        asr_batched=asr_batched,
        asr_batch_size=asr_batch_size,
        asr_cpu_threads=asr_cpu_threads,
        classifier=classifier_mode,
        classifier_cache=classifier_cache,
        fasttext_model=fasttext_model,
    )
    transcriber = Transcriber(_build_asr_cfg(cfg, model_dir))
    transcriber._load()

    ft_path = fasttext_model
    if ft_path is None and model_dir:
        ft_path = fasttext_local_path(model_dir)
    ft_lid = FastTextLID(path=ft_path)

    classifier = None
    if classifier_mode != "off":
        try:
            c = RomanIndicClassifier(cache_path=classifier_cache)
            classifier = c if c.available() else None
        except Exception:
            classifier = None
    return transcriber, ft_lid, classifier


def load_models(cfg: PipelineConfig, model_dir: Optional[str] = None):
    """Load ASR/language models once per compatible config."""
    return _load_models_cached(_model_cache_key(cfg, model_dir))


def _resolve_input(input_path: str, cfg: PipelineConfig):
    input_path = os.path.abspath(input_path)
    input_mode = cfg.input_type
    pairs: List = []

    if input_mode in ("speaker_pair", "auto"):
        pairs = detect_and_group_pairs(input_path)
        if pairs:
            input_mode = "speaker_pair"
        elif input_mode == "speaker_pair":
            raise ProcessingError("Separate speaker mode selected, but no two-file speaker sessions found.")

    stereo_files: List[str] = []
    if input_mode == "auto":
        stereo_files = detect_stereo_files(input_path)
        input_mode = "stereo" if stereo_files else "mono"

    if input_mode == "stereo":
        work_items = stereo_files or detect_stereo_files(input_path) or list(iter_audio_files(input_path))
    elif input_mode == "speaker_pair":
        work_items = pairs
    else:
        work_items = list(iter_audio_files(input_path))

    if not work_items:
        raise ProcessingError(f"No supported audio sessions found under {input_path!r}.")
    return input_mode, work_items


def _preview(record: Dict) -> Dict:
    speakers = record.get("metadata", {}).get("speakers", {})
    language = record.get("metadata", {}).get("language", {})
    interaction = record.get("metadata", {}).get("interaction", {})
    transcript = record.get("transcript", {})
    return {
        "session": record.get("session_name"),
        "transcript_preview": (transcript.get("raw") or "")[:2000],
        "detected_languages": {
            "primary_language": language.get("primary_language"),
            "dominant_language": language.get("dominant_language"),
            "switching_score": language.get("switching_score"),
            "language_segments": language.get("language_segments", [])[:12],
        },
        "speaker_stats": speakers,
        "interaction": interaction,
        "validation": record.get("validation", {}),
    }


def zip_directory(source_dir: str, zip_path: str, subdir: Optional[str] = None) -> str:
    """Create a zip file from a directory or one subdirectory."""
    root = Path(source_dir)
    target = root / subdir if subdir else root
    if not target.exists():
        raise FileNotFoundError(str(target))
    zip_path = os.path.abspath(zip_path)
    Path(zip_path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in target.rglob("*"):
            if path.is_file() and path.resolve() != Path(zip_path).resolve():
                zf.write(path, path.relative_to(root))
    return zip_path


def process_conversation(
    input_path: str,
    output_path: str,
    config: PipelineConfig | Dict | None = None,
    progress_callback: ProgressFn = None,
) -> Dict:
    """Process conversational audio into a dataset.

    Args:
        input_path: File or directory containing mono, stereo, or speaker-pair audio.
        output_path: Dataset output directory.
        config: ``PipelineConfig`` or dict.
        progress_callback: Optional callable receiving status strings.
    """
    from .runner import process_conversation as runner_process_conversation
    return runner_process_conversation(input_path, output_path, config, progress_callback)

    cfg = _normalise_config(config)
    np.random.seed(cfg.random_seed)

    model_dir = cfg.model_dir or (default_model_dir() if cfg.offline_mode else None)
    output_path = os.path.abspath(output_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    logs = io.StringIO()
    handler = logging.StreamHandler(logs)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)

    try:
        with redirect_stdout(logs), redirect_stderr(logs):
            _log_progress(progress_callback, "Resolving input")
            input_mode, work_items = _resolve_input(input_path, cfg)
            _log_progress(progress_callback, f"Detected input mode: {input_mode}")

            _log_progress(progress_callback, "Loading models")
            transcriber, ft_lid, classifier = load_models(cfg, model_dir)

            dataset_writer = DatasetWriter(
                output_root=output_path,
                output_mode=cfg.output_mode,
                output_format=cfg.output_format,
                dataset_name=cfg.dataset_name,
            )
            batch_writer = BatchWriter(
                output_dir=output_path,
                fmt=cfg.output_format,
                dataset_name=cfg.dataset_name,
            )

            shared = dict(
                transcriber=transcriber,
                ft_lid=ft_lid,
                classifier=classifier,
                cfg=cfg,
                dataset_writer=dataset_writer,
                model_dir=model_dir,
            )

            for idx, item in enumerate(work_items, 1):
                _log_progress(progress_callback, f"Processing session {idx}/{len(work_items)}")
                if input_mode == "stereo":
                    session_name = os.path.splitext(os.path.basename(item))[0]
                    pair = load_stereo_as_pair(item, session_name=session_name)
                    if pair is None:
                        raise ProcessingError(f"Unable to load stereo file: {item}")
                    record = _process_speaker_pair(pair, **shared)
                elif input_mode == "speaker_pair":
                    session_name, p1, l1, p2, l2 = item
                    pair = load_speaker_pair(p1, l1, p2, l2, session_name=session_name)
                    if pair is None:
                        raise ProcessingError(f"Unable to load speaker pair: {session_name}")
                    record = _process_speaker_pair(pair, **shared)
                else:
                    clip = load_audio(item)
                    if clip is None:
                        raise ProcessingError(f"Unable to load audio file: {item}")
                    record = _process_single(clip, **shared)

                records.append(record)
                batch_writer.write(record)

            final_batch = batch_writer.close()
            if final_batch:
                _log_progress(progress_callback, f"Batch file written: {final_batch}")
            _log_progress(progress_callback, "Done")

    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)

    full_zip = zip_directory(output_path, os.path.join(output_path, "dataset.zip"))
    downloads = {
        "dataset_zip": full_zip,
    }
    for name in ("transcripts", "annotations", "audio", "manifests", "logs"):
        path = Path(output_path) / name
        if path.exists():
            downloads[f"{name}_zip"] = zip_directory(
                output_path,
                os.path.join(output_path, f"{name}.zip"),
                subdir=name,
            )

    return {
        "input_mode": input_mode if records else cfg.input_type,
        "output_path": output_path,
        "config": asdict(cfg),
        "records": records,
        "metadata_json": records[0] if records else {},
        "previews": [_preview(r) for r in records],
        "logs": logs.getvalue(),
        "downloads": downloads,
    }
