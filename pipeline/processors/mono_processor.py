"""Mono session processor."""
from __future__ import annotations

import os
import time
from typing import Optional

from ..audio_loader import LoadedAudio
from ..config import PipelineConfig
from ..dataset_writer import DatasetWriter
from ..diarisation import DiarisationConfig, diarise, diarise_pyannote
from ..offline import pyannote_local_path
from ..preprocessing import PreprocessConfig, preprocess
from ..processors.downstream import build_asr_cfg, run_downstream
from ..quality_checker import check_mono
from ..roman_indic_classifier import RomanIndicClassifier
from ..transcription import Transcriber
from ..validation import build_validation_report
from ..vad import VADConfig, detect_speech
from ..language_detection import FastTextLID
from ..output_formatter import build_record


def _processing_report(
    cfg: PipelineConfig,
    requested_diarisation_backend: str,
    effective_diarisation_backend: str,
) -> dict:
    return {
        "random_seed": cfg.random_seed,
        "offline_mode": bool(cfg.offline_mode),
        "asr": {
            "backend": "faster-whisper",
            "model_size": cfg.model_size,
            "compute_type": cfg.compute_type,
            "device": cfg.device,
            "beam_size": cfg.beam_size,
            "batched": bool(cfg.asr_batched),
            "batch_size": cfg.asr_batch_size,
        },
        "vad": {"backend": cfg.vad_backend},
        "diarisation": {
            "requested_backend": requested_diarisation_backend,
            "effective_backend": effective_diarisation_backend,
        },
        "language": {
            "fasttext_model": cfg.fasttext_model,
            "roman_indic_classifier": cfg.classifier,
        },
        "include_runtime_metrics": bool(cfg.include_runtime_metrics),
    }


def process_single(
    clip: LoadedAudio,
    transcriber: Transcriber,
    ft_lid: FastTextLID,
    classifier: Optional[RomanIndicClassifier],
    cfg: PipelineConfig,
    dataset_writer: DatasetWriter,
    model_dir: Optional[str] = None,
) -> dict:
    t0 = time.time()

    qreport = check_mono(clip, min_duration_s=cfg.min_audio_duration_s)
    if not qreport.passed and cfg.fail_fast:
        raise RuntimeError(f"Quality check failed: {qreport.errors}")

    pcfg = PreprocessConfig(denoise=cfg.denoise)
    wav = preprocess(clip.waveform, clip.sample_rate, pcfg)
    speech = detect_speech(wav, clip.sample_rate, backend=cfg.vad_backend, cfg=VADConfig())

    requested_diarisation_backend = cfg.diarisation_backend
    effective_diarisation_backend = cfg.diarisation_backend
    if cfg.diarisation_backend == "pyannote":
        try:
            local_pyannote = pyannote_local_path(model_dir) if model_dir else None
            dia = diarise_pyannote(
                wav,
                clip.sample_rate,
                hf_token=cfg.hf_token,
                min_speakers=cfg.min_speakers if cfg.min_speakers > 1 else None,
                max_speakers=cfg.max_speakers,
                local_model_dir=local_pyannote,
                offline_mode=cfg.offline_mode,
            )
        except RuntimeError:
            effective_diarisation_backend = "kmeans_fallback"
            dia = diarise(
                wav,
                clip.sample_rate,
                speech,
                cfg=DiarisationConfig(
                    max_speakers=cfg.max_speakers,
                    min_speakers=cfg.min_speakers,
                    random_state=cfg.random_seed,
                ),
            )
    else:
        dia = diarise(
            wav,
            clip.sample_rate,
            speech,
            cfg=DiarisationConfig(
                max_speakers=cfg.max_speakers,
                min_speakers=cfg.min_speakers,
                random_state=cfg.random_seed,
            ),
        )

    asr_cfg = build_asr_cfg(cfg, model_dir)
    (
        transcript,
        lang_report,
        audio_meta,
        speaker_meta,
        conv_meta,
        monologue,
        monologues,
        interaction_meta,
        speaker_lang,
        _overlaps,
    ) = run_downstream(
        wav,
        clip.sample_rate,
        speech,
        dia,
        transcriber,
        ft_lid,
        classifier,
        asr_cfg,
        cfg=cfg,
    )

    session_name = os.path.splitext(os.path.basename(clip.path))[0]
    record = build_record(
        audio_path=clip.path,
        transcript=transcript,
        turns=dia,
        language=lang_report,
        audio_meta=audio_meta,
        speaker_meta=speaker_meta,
        conversation_meta=conv_meta,
        monologue=monologue,
        interaction_meta=interaction_meta,
        monologues=monologues,
        speaker_lang=speaker_lang,
        quality=qreport.to_dict(),
        validation=build_validation_report(
            quality_report=qreport,
            transcript=transcript,
            requested_diarisation_backend=requested_diarisation_backend,
            effective_diarisation_backend=effective_diarisation_backend,
            speech_segments=speech,
            turns=dia,
            session_duration_s=float(len(wav) / clip.sample_rate) if clip.sample_rate else 0.0,
        ),
        processing=_processing_report(cfg, requested_diarisation_backend, effective_diarisation_backend),
        input_mode="mono",
        session_name=session_name,
        source_files=[clip.path],
        generated_at=cfg.generated_at,
        skip_sha1=cfg.skip_sha1,
    )
    if cfg.include_runtime_metrics:
        record["processing_time_s"] = round(time.time() - t0, 3)

    dataset_writer.write_session(
        session_name=session_name,
        record=record,
        turns=dia,
        mixed_wav=wav,
        sample_rate=clip.sample_rate,
        monologues=monologues,
        source_paths=[clip.path],
    )
    return record
