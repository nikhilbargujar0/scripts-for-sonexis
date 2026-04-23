"""Production runner: orchestration only, step logic lives under pipeline.steps."""
from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .api import _load_models_cached, _model_cache_key, _normalise_config, _preview
from .batch_writer import BatchWriter
from .config import PipelineConfig
from .dataset_writer import DatasetWriter
from .offline import default_model_dir
from .processors.mono_processor import process_single
from .processors.pair_processor import process_speaker_pair
from .steps.audio_processing import (
    detect_and_group_pairs,
    detect_stereo_files,
    iter_audio_files,
    load_audio,
    load_speaker_pair,
    load_stereo_as_pair,
)
from .steps.validation import validate_record_against_schema, write_validation_report

ProgressFn = Optional[Callable[[str], None]]


class ProcessingError(RuntimeError):
    """Raised when input validation or processing fails cleanly."""


def _log(progress: ProgressFn, message: str) -> None:
    if progress:
        progress(message)
    logging.getLogger("sonexis.runner").info(message)


def load_models_once(cfg: PipelineConfig, model_dir: Optional[str] = None):
    return _load_models_cached(_model_cache_key(cfg, model_dir))


def _resolve_input(input_path: str, cfg: PipelineConfig):
    input_path = os.path.abspath(input_path)
    input_type = cfg.input_type
    pairs: List = []
    if input_type in ("separate", "speaker_pair"):
        input_type = "speaker_pair"

    if input_type in ("speaker_pair", "auto"):
        pairs = detect_and_group_pairs(input_path)
        if pairs:
            input_type = "speaker_pair"
        elif input_type == "speaker_pair":
            raise ProcessingError("separate input selected, but no two-file speaker session found")

    stereo_files: List[str] = []
    if input_type == "auto":
        stereo_files = detect_stereo_files(input_path)
        input_type = "stereo" if stereo_files else "mono"

    if input_type == "stereo":
        work_items = stereo_files or detect_stereo_files(input_path) or list(iter_audio_files(input_path))
    elif input_type == "speaker_pair":
        work_items = pairs
    else:
        work_items = list(iter_audio_files(input_path))

    if not work_items:
        raise ProcessingError(f"no supported audio found under {input_path!r}")
    return input_type, work_items


def _zip_directory(source_dir: str, zip_path: str, subdir: Optional[str] = None) -> str:
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
    """Run offline conversational audio pipeline and enforce dataset schema."""
    cfg = _normalise_config(config)
    np.random.seed(cfg.random_seed)
    if cfg.offline_mode and not (cfg.model_dir or default_model_dir()):
        raise ProcessingError("offline_mode true, but no local model directory configured")

    model_dir = cfg.model_dir or (default_model_dir() if cfg.offline_mode else None)
    output_path = os.path.abspath(output_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    logs = io.StringIO()
    handler = logging.StreamHandler(logs)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)

    records: List[Dict] = []
    written_validation_reports: List[str] = []
    input_type = cfg.input_type
    try:
        with redirect_stdout(logs), redirect_stderr(logs):
            _log(progress_callback, "resolve input")
            input_type, work_items = _resolve_input(input_path, cfg)
            _log(progress_callback, f"input type: {input_type}")

            _log(progress_callback, "load local models")
            transcriber, ft_lid, classifier = load_models_once(cfg, model_dir)
            dataset_writer = DatasetWriter(output_root=output_path, output_mode=cfg.output_mode,
                                           output_format=cfg.output_format, dataset_name=cfg.dataset_name)
            batch_writer = BatchWriter(output_dir=output_path, fmt=cfg.output_format,
                                       dataset_name=cfg.dataset_name)

            shared = dict(transcriber=transcriber, ft_lid=ft_lid, classifier=classifier,
                          cfg=cfg, dataset_writer=dataset_writer, model_dir=model_dir)

            for idx, item in enumerate(work_items, 1):
                _log(progress_callback, f"process session {idx}/{len(work_items)}")
                if input_type == "stereo":
                    session_name = os.path.splitext(os.path.basename(item))[0]
                    pair = load_stereo_as_pair(item, session_name=session_name)
                    if pair is None:
                        raise ProcessingError(f"cannot load stereo file: {item}")
                    record = process_speaker_pair(pair, **shared)
                elif input_type == "speaker_pair":
                    session_name, p1, l1, p2, l2 = item
                    pair = load_speaker_pair(p1, l1, p2, l2, session_name=session_name)
                    if pair is None:
                        raise ProcessingError(f"cannot load speaker pair: {session_name}")
                    record = process_speaker_pair(pair, **shared)
                else:
                    clip = load_audio(item)
                    if clip is None:
                        raise ProcessingError(f"cannot load audio file: {item}")
                    record = process_single(clip, **shared)

                validate_record_against_schema(record)
                if cfg.fail_fast and not record.get("validation", {}).get("passed", True):
                    raise ProcessingError(
                        f"validation failed for {record.get('session_name', 'session')}: "
                        f"{record.get('validation', {}).get('issues', [])}"
                    )
                written_validation_reports.append(
                    write_validation_report(record.get("validation", {}), output_path, record.get("session_name", "session"))
                )
                records.append(record)
                batch_writer.write(record)

            final_batch = batch_writer.close()
            if final_batch:
                _log(progress_callback, f"batch written: {final_batch}")
            _log(progress_callback, "done")
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)

    downloads = {"dataset_zip": _zip_directory(output_path, os.path.join(output_path, "dataset.zip"))}
    for name in ("transcripts", "annotations", "audio", "manifests", "logs"):
        path = Path(output_path) / name
        if path.exists():
            downloads[f"{name}_zip"] = _zip_directory(output_path, os.path.join(output_path, f"{name}.zip"), subdir=name)

    return {
        "input_mode": input_type,
        "output_path": output_path,
        "config": asdict(cfg),
        "records": records,
        "metadata_json": records[0] if records else {},
        "previews": [_preview(r) for r in records],
        "logs": logs.getvalue(),
        "validation_reports": written_validation_reports,
        "downloads": downloads,
    }


def write_example_output(path: str) -> None:
    example = {
        "schema_version": "3.0",
        "generated_at": "1970-01-01T00:00:00+00:00",
        "input_mode": "speaker_pair",
        "session_name": "conversation_0001",
        "metadata": {
            "audio": {"duration_s": 31.2},
            "language": {"segments": [{"start": 0.0, "end": 2.1, "lang": "Hindi"}]},
            "speakers": {},
            "conversation": {},
            "interaction": {"turn_count": 12, "interruptions": 3, "overlap_duration": 4.1},
        },
        "transcript": {"raw": "haan so I was saying..."},
        "speaker_segmentation": [],
        "validation": {"passed": True, "issue_count": 0, "issues": [], "checks": {}},
        "processing": {"offline_mode": True, "random_seed": 0},
    }
    Path(path).write_text(json.dumps(example, indent=2, sort_keys=True) + "\n", encoding="utf-8")
