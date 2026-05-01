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
    build_accuracy_gate,
    build_premium_processing,
    build_quality_metrics,
    build_quality_targets,
    build_tts_suitability,
)
from ..premium.review import build_human_review
from ..premium.types import AlignmentResult, ConsensusResult
from ..quality_checker import check_speaker_pair
from ..roman_indic_classifier import RomanIndicClassifier
from ..steps.alignment import AlignmentError, align_pair
from ..steps.audio_processing import mix_mono_with_metadata
from ..transcription import Transcriber, Transcript
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
    if pair.recording_type == "studio_speaker_folders":
        durations = {
            label: round(float(audio.duration), 3)
            for label, audio in pair.speakers.items()
        }
        values = list(durations.values())
        mismatch = (max(values) - min(values)) if len(values) >= 2 else 0.0
        return speaker_wavs, {
            "method": "shared_studio_timeline",
            "offset_ms": 0,
            "confidence": 1.0,
            "passed": True,
            "applied": False,
            "note": "speaker folders provide ground-truth identity; no diarisation or waveform shift was run",
            "speaker_durations_s": durations,
            "duration_mismatch_s": round(float(mismatch), 3),
            "max_duration_mismatch_s": float(cfg.max_duration_mismatch_s),
        }
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


def _merge_speaker_transcripts(
    speaker_transcripts: Dict[str, Transcript],
    speaker_map: Dict[str, str],
) -> Transcript:
    segments = []
    language_scores: Dict[str, float] = {}
    for pipeline_label, human_label in speaker_map.items():
        transcript = speaker_transcripts.get(human_label)
        if transcript is None:
            continue
        language_scores[transcript.language] = max(
            language_scores.get(transcript.language, 0.0),
            float(transcript.language_probability or 0.0),
        )
        for seg in transcript.segments:
            setattr(seg, "ground_truth_speaker", pipeline_label)
            segments.append(seg)
    segments.sort(key=lambda s: (float(s.start), float(s.end)))
    primary_language = max(language_scores, key=language_scores.get) if language_scores else "unknown"
    duration = max((float(seg.end) for seg in segments), default=0.0)
    return Transcript(
        language=primary_language,
        language_probability=float(language_scores.get(primary_language, 0.0)),
        duration=duration,
        segments=segments,
    )


def _mark_transcript_speaker(transcript: Transcript, pipeline_label: str) -> Transcript:
    for seg in transcript.segments:
        setattr(seg, "ground_truth_speaker", pipeline_label)
    return transcript


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
    per_speaker_transcripts: Dict[str, Transcript] = {}
    if pair.recording_type == "studio_speaker_folders":
        transcriber.cfg = asr_cfg
        if pipeline_mode == "premium_accuracy":
            per_speaker_consensus: Dict[str, ConsensusResult] = {}
            per_speaker_alignment: Dict[str, AlignmentResult] = {}
            for pipeline_label, human_label in speaker_map.items():
                wav = speaker_wavs[human_label]
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
                    pair.mixed.sample_rate,
                    audio_meta=preview_audio_meta,
                    overlap_duration_s=expected_overlap_s,
                )
                speaker_candidates = list(routed["candidates"])
                for candidate in speaker_candidates:
                    candidate.adapter_metadata["speaker_label"] = human_label
                    candidate.adapter_metadata["speaker_id"] = pipeline_label
                    transcript_candidates.append(candidate)
                routing_decision = routed["routing_decision"]
                selected, consensus = choose_consensus(
                    speaker_candidates,
                    audio_condition=normalize_recording_condition(preview_audio_meta),
                )
                refined = refine_timestamps(selected, wav=wav, sample_rate=pair.mixed.sample_rate, cfg=cfg)
                _mark_transcript_speaker(refined.transcript, pipeline_label)
                per_speaker_transcripts[human_label] = refined.transcript
                per_speaker_consensus[human_label] = consensus
                per_speaker_alignment[human_label] = refined
            transcript_override = _merge_speaker_transcripts(per_speaker_transcripts, speaker_map)
            consensus_scores = [c.consensus_score for c in per_speaker_consensus.values()]
            timestamp_scores = [a.timestamp_confidence for a in per_speaker_alignment.values()]
            selected_candidate = transcript_candidates[0] if transcript_candidates else None
            consensus_result = ConsensusResult(
                transcript=transcript_override,
                selected_engine="speaker_consensus",
                consensus_score=round(float(min(consensus_scores) if consensus_scores else 0.0), 4),
                engines_compared=list(dict.fromkeys(c.engine for c in transcript_candidates)),
                transcript_strategy="per_speaker_consensus",
                review_recommended=bool(any(c.review_recommended for c in per_speaker_consensus.values())),
                rationale=["speaker_folder_premium_consensus", "speaker_separated_asr_candidates_preserved"],
                candidate_rationales={
                    f"{human_label}:{engine}": rationale
                    for human_label, consensus in per_speaker_consensus.items()
                    for engine, rationale in consensus.candidate_rationales.items()
                },
                disagreement_flags=list(dict.fromkeys(
                    flag
                    for consensus in per_speaker_consensus.values()
                    for flag in consensus.disagreement_flags
                )),
            )
            alignment_result = AlignmentResult(
                transcript=transcript_override,
                timestamp_method="per_speaker_alignment",
                timestamp_confidence=round(float(min(timestamp_scores) if timestamp_scores else 0.0), 4),
                word_timestamps_available=any(seg.words for seg in transcript_override.segments),
                segment_timestamps_available=any(seg.end > seg.start for seg in transcript_override.segments),
                refinement_applied=any(a.refinement_applied for a in per_speaker_alignment.values()),
                synthetic_word_timestamps=any(a.synthetic_word_timestamps for a in per_speaker_alignment.values()),
                timing_quality="high" if timestamp_scores and min(timestamp_scores) >= 0.98 else "medium",
                notes=["per_speaker_premium_alignment"],
            )
        else:
            for human_label, wav in speaker_wavs.items():
                per_speaker_transcripts[human_label] = transcriber.transcribe(wav, pair.mixed.sample_rate)
            transcript_override = _merge_speaker_transcripts(per_speaker_transcripts, speaker_map)
    elif pipeline_mode == "premium_accuracy":
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
    metadata = dict(getattr(pair, "conversation_metadata", {}) or {})
    if metadata:
        primary = {
            "conversation_id": metadata.get("conversation_id") or pair.session_name,
            "scenario_id": metadata.get("scenario_id"),
            "scenario_name": metadata.get("scenario_name"),
            "topic": metadata.get("topic"),
            "sub_topic": metadata.get("sub_topic"),
            "conversation_style": metadata.get("conversation_style"),
            "scripted": metadata.get("scripted"),
            "language_mix": metadata.get("language_mix"),
            "speaker_roles": metadata.get("speaker_roles") or metadata.get("speaker roles"),
            "speaker_profiles": metadata.get("speaker_profiles") or metadata.get("speaker profile metadata"),
            "recording_setup": metadata.get("recording_setup") or metadata.get("recording setup"),
            "consent_status": metadata.get("consent_status") or metadata.get("consent status"),
        }
        conv_meta["provided_metadata"] = {
            **dict(conv_meta.get("provided_metadata") or {}),
            **{k: v for k, v in primary.items() if v is not None},
        }
        for key in ("topic", "sub_topic", "scenario_id", "scenario_name", "conversation_style", "scripted", "language_mix"):
            if primary.get(key) is not None:
                conv_meta[key] = {"value": primary[key], "source": "metadata_json", "confidence": 1.0}
        if primary.get("recording_setup") is not None:
            audio_meta["recording_setup"] = {
                "value": primary["recording_setup"],
                "source": "metadata_json",
                "confidence": 1.0,
            }
    quality_targets = build_quality_targets(cfg)
    quality_metrics = build_quality_metrics()
    accuracy_gate = build_accuracy_gate(
        cfg=cfg,
        consensus=consensus_result,
        alignment=alignment_result,
        code_switch=code_switch,
        speaker_attribution_confidence=1.0,
    )
    quality_metrics["estimated_word_accuracy"] = accuracy_gate["estimated_word_accuracy"]
    quality_metrics["estimated_timestamp_accuracy"] = accuracy_gate["estimated_timestamp_accuracy"]
    quality_metrics["estimated_code_switch_accuracy"] = accuracy_gate.get("estimated_code_switch_accuracy")
    human_review = build_human_review(
        pipeline_mode=pipeline_mode,
        require_human_review=bool(getattr(cfg, "require_human_review", True)),
        consensus=consensus_result,
        alignment=alignment_result,
        routing=routing_decision,
        audio_condition=normalize_recording_condition(audio_meta),
        code_switch=code_switch,
        accuracy_gate=accuracy_gate,
        targets=quality_targets,
        speaker_attribution_confidence=1.0,
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

    if pair.recording_type == "stereo":
        input_mode = "stereo"
    elif pair.recording_type == "studio_speaker_folders":
        input_mode = "speaker_folders"
    else:
        input_mode = "speaker_pair"
    requested_diarisation = "none" if input_mode == "speaker_folders" else "speaker_vad"
    effective_diarisation = "none" if input_mode == "speaker_folders" else "speaker_vad"
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
            requested_diarisation_backend=requested_diarisation,
            effective_diarisation_backend=effective_diarisation,
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
        original_sample_rate=getattr(pair.mixed, "original_sample_rate", None) or pair.mixed.sample_rate,
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
    record["accuracy_gate"] = accuracy_gate
    if input_mode == "speaker_folders":
        record["metadata_json"] = metadata
        record["inferred_metadata"] = {
            "conversation": {
                k: v for k, v in conv_meta.items()
                if isinstance(v, dict) and v.get("source") != "metadata_json" and "confidence" in v
            },
            "speakers": {
                spk: {
                    k: v for k, v in meta.items()
                    if isinstance(v, dict) and v.get("source") != "metadata_json" and "confidence" in v
                }
                for spk, meta in speaker_meta.items()
            },
        }
        record["validation"]["checks"]["speaker_folder_structure"] = pair.validation_context.get("structure", {})
        record["validation"]["checks"]["source_audio_format"] = pair.validation_context.get("audio_format", {})
        for warning in pair.validation_context.get("warnings", []):
            record["validation"]["issues"].append({
                "severity": "warning",
                "code": "source_audio_format_warning",
                "message": warning,
                "confidence": 1.0,
                "details": {},
            })
        record["validation"]["issue_count"] = len(record["validation"].get("issues", []))
        record["validation"]["passed"] = not any(
            issue.get("severity") == "error" for issue in record["validation"].get("issues", [])
        )
        record["processing"]["diarisation"] = {
            "requested_backend": "none",
            "effective_backend": "none",
            "fallback_available_for_input_type": "mono",
            "note": "speaker folders are ground-truth speaker identity; diarisation was not run",
        }
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
    review_required = bool(record.get("human_review", {}).get("required"))
    delivery_passed = bool(
        record.get("accuracy_gate", {}).get("passed")
        and record.get("validation", {}).get("passed")
        and not review_required
    )
    record["delivery_status"] = {
        "stage": "approved" if delivery_passed else ("review_required" if review_required else "rejected"),
        "approved_for_client_delivery": delivery_passed,
        "reason": "" if delivery_passed else ";".join(
            list(record.get("accuracy_gate", {}).get("reasons") or [])
            or (["human_review_pending"] if review_required else [])
            or (["validation_failed"] if not record.get("validation", {}).get("passed") else [])
        ),
    }
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
        original_sources=pair.speakers if input_mode == "speaker_folders" else None,
        speaker_lang={k: v.to_dict() for k, v in speaker_lang.items()},
    )
    return record
