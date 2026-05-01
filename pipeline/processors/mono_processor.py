"""Mono session processor."""
from __future__ import annotations

import os
import time
from typing import List, Optional

from ..audio_loader import LoadedAudio
from ..config import PipelineConfig
from ..dataset_writer import DatasetWriter
from ..diarisation import DiarisationConfig, diarise, diarise_pyannote
from ..metadata_extraction import extract_audio_metadata
from ..offline import pyannote_local_path
from ..preprocessing import PreprocessConfig, preprocess
from ..processors.downstream import build_asr_cfg, run_downstream
from ..premium.alignment_router import refine_timestamps
from ..premium.asr_router import PremiumASRRouter
from ..premium.consensus import choose_consensus
from ..premium.adapters.whisper_local import WhisperLocalAdapter
from ..premium.quality import (
    build_code_switch_metadata,
    build_premium_processing,
    build_quality_metrics,
    build_quality_targets,
    build_tts_suitability,
)
from ..premium.review import build_human_review
from ..quality_checker import check_mono
from ..roman_indic_classifier import RomanIndicClassifier
from ..transcription import Transcriber
from ..utils.premium_routing import normalize_recording_condition
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
            "denoise": bool(cfg.denoise),
            "initial_prompt": getattr(cfg, "initial_prompt", None),
            "no_speech_threshold": getattr(cfg, "no_speech_threshold", 0.6),
            "compression_ratio_threshold": getattr(cfg, "compression_ratio_threshold", 2.4),
            "condition_on_previous_text": bool(getattr(cfg, "condition_on_previous_text", False)),
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


def _dataset_products(cfg: PipelineConfig) -> List[str]:
    products = list(getattr(cfg, "export_products", []) or [])
    if cfg.output_mode in ("both", "mono"):
        products.append("mono_mixed")
    if cfg.output_mode in ("both", "speaker_separated"):
        products.append("speaker_separated")
    seen = set()
    out: List[str] = []
    for product in products:
        if product and product not in seen:
            seen.add(product)
            out.append(product)
    return out


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
    preview_audio_meta = extract_audio_metadata(wav, clip.sample_rate, speech)
    pipeline_mode = getattr(cfg, "pipeline_mode", "offline_standard")
    transcript_override = None
    transcript_candidates = []
    routing_decision = None
    consensus_result = None
    alignment_result = None
    selected_candidate = None
    if pipeline_mode == "premium_accuracy":
        router = PremiumASRRouter(
            cfg=cfg,
            whisper_adapter=WhisperLocalAdapter(
                transcriber=transcriber,
                asr_cfg=asr_cfg,
                fasttext_lid=ft_lid,
                roman_indic_classifier=classifier,
            ),
            fasttext_lid=ft_lid,
            roman_indic_classifier=classifier,
        )
        routed = router.run(
            wav,
            clip.sample_rate,
            audio_meta=preview_audio_meta,
            overlap_duration_s=0.0,
        )
        transcript_candidates = list(routed["candidates"])
        routing_decision = routed["routing_decision"]
        selected_candidate, consensus_result = choose_consensus(
            transcript_candidates,
            audio_condition=normalize_recording_condition(preview_audio_meta),
        )
        alignment_result = refine_timestamps(
            selected_candidate,
            wav=wav,
            sample_rate=clip.sample_rate,
            cfg=cfg,
        )
        transcript_override = alignment_result.transcript

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
        source_audios=[clip],
        transcript_override=transcript_override,
    )

    code_switch = build_code_switch_metadata(
        language_report=lang_report.to_dict(),
        candidate=selected_candidate,
    )
    quality_targets = build_quality_targets(cfg)
    quality_metrics = build_quality_metrics()
    human_review = build_human_review(
        pipeline_mode=pipeline_mode,
        require_human_review=bool(getattr(cfg, "require_human_review", True)),
        consensus=consensus_result,
        alignment=alignment_result,
        routing=routing_decision,
        audio_condition=normalize_recording_condition(audio_meta),
        code_switch=code_switch,
    )
    tts_suitability = build_tts_suitability(
        record_or_meta=audio_meta,
        review_status=str(human_review.get("status") or "pending"),
        overlap_duration_s=0.0,
        speaker_count=len(speaker_meta),
        alignment=alignment_result,
    )
    if routing_decision is not None and consensus_result is not None and alignment_result is not None:
        premium_processing = build_premium_processing(
            pipeline_mode=pipeline_mode,
            routing=routing_decision,
            consensus=consensus_result,
            alignment=alignment_result,
            human_review_required=bool(human_review.get("required")),
        )
    else:
        premium_processing = {
            "pipeline_mode": pipeline_mode,
            "paid_api_used": False,
            "engines_used": ["whisper_local"],
            "consensus_applied": False,
            "timestamp_refinement_applied": False,
            "human_review_required": bool(human_review.get("required")),
        }

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
            quality_score_threshold=getattr(cfg, "quality_score_threshold", 0.35),
        ),
        processing=_processing_report(cfg, requested_diarisation_backend, effective_diarisation_backend),
        input_mode="mono",
        session_name=session_name,
        source_files=[clip.path],
        generated_at=cfg.generated_at,
        quality_targets=quality_targets,
        quality_metrics=quality_metrics,
        cfg=cfg,
        total_speech_duration_sec=sum(end - start for start, end in speech),
        original_sample_rate=getattr(clip, "original_sample_rate", None) or clip.sample_rate,
        wav=wav,
        skip_sha1=cfg.skip_sha1,
    )
    record["human_review"] = human_review
    record["code_switch"] = code_switch
    record["tts_suitability"] = tts_suitability
    record["dataset_products"] = _dataset_products(cfg)
    record["premium_processing"] = premium_processing
    record["pipeline_mode"] = pipeline_mode
    if transcript_candidates and bool(getattr(cfg, "store_transcript_candidates", True)):
        record["transcript_candidates"] = [
            candidate.to_dict(
                include_segments=bool(getattr(cfg, "store_candidate_segments", True)),
                include_words=bool(getattr(cfg, "store_candidate_words", False)),
            )
            for candidate in transcript_candidates
        ]
    if routing_decision is not None:
        record["routing_decision"] = routing_decision.to_dict()
    if consensus_result is not None:
        record["consensus"] = consensus_result.to_dict()
    if alignment_result is not None:
        record["timestamp_method"] = alignment_result.timestamp_method
        record["timestamp_confidence"] = round(float(alignment_result.timestamp_confidence), 4)
        record["timestamp_refinement"] = alignment_result.to_dict()
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
