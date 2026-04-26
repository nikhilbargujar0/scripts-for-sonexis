"""Dual-speaker processor with real alignment application."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..audio_loader import SpeakerPairAudio
from ..config import PipelineConfig
from ..dataset_writer import DatasetWriter
from ..diarisation import SpeakerTurn, diarise_from_speaker_vad
from ..language_detection import FastTextLID
from ..metadata_extraction import extract_audio_metadata
from ..output_formatter import build_record
from ..preprocessing import PreprocessConfig, preprocess
from ..processors.downstream import build_asr_cfg, run_downstream
from ..premium.adapters.whisper_local import WhisperLocalAdapter
from ..premium.alignment_router import refine_timestamps
from ..premium.asr_router import PremiumASRRouter
from ..premium.consensus import choose_consensus
from ..premium.quality import (
    build_code_switch_metadata,
    build_premium_processing,
    build_quality_metrics,
    build_quality_targets,
    build_tts_suitability,
)
from ..premium.review import build_human_review
from ..quality_checker import check_speaker_pair
from ..roman_indic_classifier import RomanIndicClassifier
from ..steps.alignment import AlignmentError, align_pair
from ..steps.audio_processing import mix_mono_with_metadata
from ..transcription import Transcriber
from ..utils.premium_routing import normalize_recording_condition
from ..validation import build_validation_report
from ..vad import VADConfig, detect_speech


def _vad_parallel(
    wavs: Dict[str, np.ndarray],
    sample_rate: int,
    vad_backend: str,
) -> Dict[str, List[Tuple[float, float]]]:
    results: Dict[str, List[Tuple[float, float]]] = {}
    for label, wav in wavs.items():
        results[label] = detect_speech(wav, sample_rate, backend=vad_backend, cfg=VADConfig())
    return results


def _vad_union(
    speaker_vad: Dict[str, List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    all_segs = [seg for segs in speaker_vad.values() for seg in segs]
    if not all_segs:
        return []
    all_segs.sort(key=lambda s: s[0])
    merged: List[Tuple[float, float]] = [all_segs[0]]
    for s, e in all_segs[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged


def _overlap_duration(segments_a: List[Tuple[float, float]], segments_b: List[Tuple[float, float]]) -> float:
    total = 0.0
    for sa, ea in segments_a:
        for sb, eb in segments_b:
            overlap = min(ea, eb) - max(sa, sb)
            if overlap > 0:
                total += overlap
    return round(total, 3)


def _build_alignment_report(
    pair: SpeakerPairAudio,
    speaker_wavs: Dict[str, np.ndarray],
    cfg: PipelineConfig,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    labels = list(pair.speakers)
    if pair.recording_type == "stereo":
        return speaker_wavs, {
            "method": "shared_stereo_timeline",
            "offset_ms": 0,
            "confidence": 1.0,
            "passed": True,
            "applied": False,
            "note": "stereo channels already share timeline",
        }
    if len(labels) < 2:
        return speaker_wavs, {
            "method": "cross_correlation",
            "offset_ms": 0,
            "confidence": 0.0,
            "passed": False,
            "applied": False,
            "note": "not enough speaker channels for alignment",
        }

    first, second = labels[0], labels[1]
    try:
        aligned_a, aligned_b, report = align_pair(
            speaker_wavs[first],
            speaker_wavs[second],
            pair.mixed.sample_rate,
            min_confidence=cfg.alignment_min_confidence,
            fail_unreliable=cfg.fail_fast,
        )
    except AlignmentError:
        raise

    durations = {
        label: round(float(audio.duration), 3)
        for label, audio in pair.speakers.items()
    }
    values = list(durations.values())
    mismatch = (max(values) - min(values)) if len(values) >= 2 else 0.0
    applied = bool(report.get("passed"))
    out_wavs = dict(speaker_wavs)
    if applied:
        out_wavs[first] = aligned_a
        out_wavs[second] = aligned_b
    report = {
        "method": "cross_correlation",
        "offset_ms": int(report.get("offset_ms", 0) or 0),
        "confidence": round(float(report.get("confidence", 0.0) or 0.0), 4),
        "passed": bool(report.get("passed")),
        "applied": applied,
        "note": "positive offset means second speaker file was delayed relative to first",
        "speaker_durations_s": durations,
        "duration_mismatch_s": round(float(mismatch), 3),
        "max_duration_mismatch_s": float(cfg.max_duration_mismatch_s),
    }
    return out_wavs, report


def _processing_report(cfg: PipelineConfig) -> Dict:
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
            "requested_backend": "speaker_vad",
            "effective_backend": "speaker_vad",
            "merge_gap_s": cfg.pair_merge_gap_s,
            "min_turn_duration_s": cfg.pair_min_turn_duration_s,
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


def process_speaker_pair(
    pair: SpeakerPairAudio,
    transcriber: Transcriber,
    ft_lid: FastTextLID,
    classifier: Optional[RomanIndicClassifier],
    cfg: PipelineConfig,
    dataset_writer: DatasetWriter,
    model_dir: Optional[str] = None,
) -> dict:
    t0 = time.time()

    qreport = check_speaker_pair(
        pair,
        min_duration_s=cfg.min_audio_duration_s,
        max_duration_mismatch_s=cfg.max_duration_mismatch_s,
    )

    pcfg = PreprocessConfig(denoise=cfg.denoise)
    speaker_wavs: Dict[str, np.ndarray] = {
        label: preprocess(audio.waveform, audio.sample_rate, pcfg)
        for label, audio in pair.speakers.items()
    }
    speaker_wavs, alignment = _build_alignment_report(pair, speaker_wavs, cfg)

    speaker_vad = _vad_parallel(speaker_wavs, pair.mixed.sample_rate, cfg.vad_backend)
    mixed_speech = _vad_union(speaker_vad)
    expected_overlap_s = 0.0
    if len(speaker_vad) >= 2:
        labels = list(speaker_vad)
        expected_overlap_s = _overlap_duration(speaker_vad[labels[0]], speaker_vad[labels[1]])

    dia, speaker_map = diarise_from_speaker_vad(
        speaker_vad=speaker_vad,
        merge_gap_s=cfg.pair_merge_gap_s,
        min_turn_duration_s=cfg.pair_min_turn_duration_s,
        preserve_overlaps=True,
    )

    mixed_wav, mono_mix = mix_mono_with_metadata(speaker_wavs.values())
    asr_cfg = build_asr_cfg(cfg, model_dir)
    preview_audio_meta = extract_audio_metadata(mixed_wav, pair.mixed.sample_rate, mixed_speech)
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
            mixed_wav,
            pair.mixed.sample_rate,
            audio_meta=preview_audio_meta,
            overlap_duration_s=expected_overlap_s,
        )
        transcript_candidates = list(routed["candidates"])
        routing_decision = routed["routing_decision"]
        selected_candidate, consensus_result = choose_consensus(
            transcript_candidates,
            audio_condition=normalize_recording_condition(preview_audio_meta),
        )
        alignment_result = refine_timestamps(
            selected_candidate,
            wav=mixed_wav,
            sample_rate=pair.mixed.sample_rate,
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
        overlaps,
    ) = run_downstream(
        mixed_wav,
        pair.mixed.sample_rate,
        mixed_speech,
        dia,
        transcriber,
        ft_lid,
        classifier,
        asr_cfg,
        speaker_map=speaker_map,
        cfg=cfg,
        source_audios=(
            [pair.mixed]
            if pair.recording_type == "stereo"
            else list(pair.speakers.values())
        ),
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
        overlap_duration_s=expected_overlap_s,
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

    speaker_sources: Dict[str, Dict] = {}
    for pipeline_label, human_label in speaker_map.items():
        src_audio = pair.speakers.get(human_label)
        speaker_sources[pipeline_label] = {
            "label": human_label,
            "file": src_audio.path if src_audio else "unknown",
            "duration_s": round(src_audio.duration, 3) if src_audio else 0.0,
            "vad_segments": len(speaker_vad.get(human_label, [])),
        }

    if pair.recording_type == "stereo":
        source_paths = [pair.mixed.path] if os.path.isfile(pair.mixed.path) else []
    else:
        source_paths = [a.path for a in pair.speakers.values() if os.path.isfile(a.path)]

    input_mode = "stereo" if pair.recording_type == "stereo" else "speaker_pair"
    record = build_record(
        audio_path=pair.mixed.path,
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
            requested_diarisation_backend="speaker_vad",
            effective_diarisation_backend="speaker_vad",
            speech_segments=mixed_speech,
            turns=dia,
            input_alignment=alignment,
            mono_mix=mono_mix,
            interaction_meta=interaction_meta,
            expected_overlap_duration_s=expected_overlap_s,
            session_duration_s=float(len(mixed_wav) / pair.mixed.sample_rate) if pair.mixed.sample_rate else 0.0,
            alignment_required=input_mode == "speaker_pair",
            quality_score_threshold=getattr(cfg, "quality_score_threshold", 0.35),
        ),
        processing=_processing_report(cfg),
        input_mode=input_mode,
        speaker_sources=speaker_sources,
        session_name=pair.session_name,
        speaker_map=speaker_map,
        source_files=source_paths,
        generated_at=cfg.generated_at,
        quality_targets=quality_targets,
        quality_metrics=quality_metrics,
        cfg=cfg,
        total_speech_duration_sec=sum(end - start for start, end in mixed_speech),
        original_sample_rate=pair.mixed.source_sample_rate or pair.mixed.sample_rate,
        wav=mixed_wav,
        skip_sha1=cfg.skip_sha1,
        input_alignment=alignment,
        mono_mix=mono_mix,
    )
    record["human_review"] = human_review
    record["code_switch"] = code_switch
    record["tts_suitability"] = tts_suitability
    record["dataset_products"] = _dataset_products(cfg)
    record["premium_processing"] = premium_processing
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

    spk_wav_by_id: Dict[str, np.ndarray] = {
        pipeline_label: speaker_wavs[human_label]
        for pipeline_label, human_label in speaker_map.items()
        if human_label in speaker_wavs
    }

    dataset_writer.write_session(
        session_name=pair.session_name,
        record=record,
        turns=dia,
        speaker_wavs=spk_wav_by_id,
        mixed_wav=mixed_wav,
        sample_rate=pair.mixed.sample_rate,
        speaker_map=speaker_map,
        monologues=monologues,
        source_paths=source_paths,
        speaker_lang={k: v.to_dict() for k, v in speaker_lang.items()},
    )
    return record
