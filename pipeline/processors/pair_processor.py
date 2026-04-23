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
from ..output_formatter import build_record
from ..preprocessing import PreprocessConfig, preprocess
from ..processors.downstream import build_asr_cfg, run_downstream
from ..quality_checker import check_speaker_pair
from ..roman_indic_classifier import RomanIndicClassifier
from ..steps.alignment import AlignmentError, align_pair
from ..steps.audio_processing import mix_mono_with_metadata
from ..transcription import Transcriber
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
    )

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
        ),
        processing=_processing_report(cfg),
        input_mode=input_mode,
        speaker_sources=speaker_sources,
        session_name=pair.session_name,
        speaker_map=speaker_map,
        source_files=source_paths,
        generated_at=cfg.generated_at,
        skip_sha1=cfg.skip_sha1,
        input_alignment=alignment,
        mono_mix=mono_mix,
    )
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
