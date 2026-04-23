"""Callable backend API for internal script integrations."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

from .config import PipelineConfig
from .language_detection import FastTextLID
from .offline import fasttext_local_path
from .processors.downstream import build_asr_cfg, load_user_metadata_file, normalise_user_metadata_block
from .roman_indic_classifier import RomanIndicClassifier
from .transcription import Transcriber

ProgressFn = Optional[Callable[[str], None]]


class ProcessingError(RuntimeError):
    """Raised when input validation or processing fails cleanly."""


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
        cfg.user_metadata = load_user_metadata_file(cfg.metadata_file)
    cli_values = {
        field: getattr(cfg, field)
        for field in ("dialect", "region", "gender", "age_band", "recording_context", "consent_status")
    }
    cli_block = normalise_user_metadata_block(cli_values, "api_config")
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
    transcriber = Transcriber(build_asr_cfg(cfg, model_dir))
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


def zip_directory(source_dir: str, zip_path: str, subdir: Optional[str] = None) -> str:
    from .runner import _zip_directory
    return _zip_directory(source_dir, zip_path, subdir)
