"""main.py — CLI entrypoint for the offline conversational-audio pipeline.

Input modes
-----------
  auto (default)    Detect from folder structure; prefer speaker_pair when found.
  mono              Single mixed audio file per session.
  speaker_pair      Two per-speaker files per session (e.g. Host.wav + Guest.wav).
  stereo            One stereo WAV per session (left = spk1, right = spk2).

Speed tips
----------
  --beam-size 1          Greedy decode (~2× faster, mild quality drop)
  --asr-batched          BatchedInferencePipeline (~2-3× faster, needs faster-whisper >= 1.1)
  --model-size base      Smaller model (~3× faster than small, lower accuracy)
  --device cuda          GPU (10-50× faster than CPU when available)
  --num-workers 4        Parallel sessions (each worker loads its own model copy)
  --skip-sha1            Skip file hashing (saves I/O on large files)
  --asr-cpu-threads 4    Limit CPU threads per worker (useful with --num-workers)

Usage
-----
  # Auto-detect, write to ./dataset
  python -m pipeline.main --input ./audio --output ./dataset

  # Fast batch run on 8-core machine
  python -m pipeline.main --input ./audio --output ./dataset \\
      --beam-size 1 --asr-batched --num-workers 4 --skip-sha1

  # Fully offline
  python -m pipeline.main --input ./audio --output ./dataset \\
      --offline_mode true --model-dir ./models
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio_loader import (
    LoadedAudio,
    SpeakerPairAudio,
    detect_and_group_pairs,
    detect_stereo_files,
    iter_audio_files,
    load_audio,
    load_speaker_pair,
    load_stereo_as_pair,
)
from .batch_writer import SUPPORTED_FORMATS, BatchWriter
from .config import PipelineConfig
from .dataset_writer import DatasetWriter
from .diarisation import (
    DiarisationConfig,
    SpeakerTurn,
    diarise,
    diarise_from_speaker_vad,
    diarise_pyannote,
)
from .interaction_metadata import extract_interaction_metadata
from .language_detection import FastTextLID, detect_language, detect_language_per_speaker
from .metadata_extraction import (
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)
from .monologue_extractor import MonologueConfig, extract_monologue, extract_monologues_per_speaker
from .output_formatter import build_record
from .preprocessing import PreprocessConfig, preprocess
from .quality_checker import check_mono, check_speaker_pair
from .roman_indic_classifier import RomanIndicClassifier
from .transcription import ASRConfig, FILLERS, Transcriber
from .vad import VADConfig, detect_speech
from .validation import build_validation_report
from .steps.alignment import estimate_offset

log = logging.getLogger("sonexis")

USER_METADATA_FIELDS = (
    "dialect",
    "region",
    "gender",
    "age_band",
    "recording_context",
    "consent_status",
)


# ── logging ────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ── argument parsing ───────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline conversational-audio → structured dataset pipeline",
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--input-mode", dest="input_type", default="auto",
                   choices=["auto", "mono", "speaker_pair", "stereo"])
    p.add_argument("--output-mode", default="both",
                   choices=["both", "speaker_separated", "mono"])
    p.add_argument("--mode", dest="output_mode", default=argparse.SUPPRESS,
                   choices=["both", "speaker_separated", "mono"],
                   help="Alias for --output-mode. Example: --mode both.")
    p.add_argument("--output-format", default="json",
                   choices=list(SUPPORTED_FORMATS))
    p.add_argument("--dataset-name", default="dataset")
    p.add_argument("--model-size", default="small",
                   choices=["tiny", "base", "small", "medium", "large-v3"])
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--language", default=None)
    p.add_argument("--vad-backend", default="webrtc", choices=["webrtc", "silero"])
    p.add_argument("--diarisation-backend", default="kmeans",
                   choices=["kmeans", "pyannote"])
    p.add_argument("--hf-token", default=None)
    p.add_argument("--max-speakers", type=int, default=4)
    p.add_argument("--min-speakers", type=int, default=1)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--fasttext-model", default=None)
    p.add_argument("--classifier", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--classifier-cache", default=None)
    p.add_argument("--offline-mode", "--offline_mode", dest="offline_mode", default="true",
                   help="true by default: load models from --model-dir only.")
    p.add_argument("--allow-model-downloads", action="store_true",
                   help="Allow faster-whisper/HF downloads at runtime. Off by default.")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--interruption-threshold", type=float, default=0.5)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    # ── speed args ─────────────────────────────────────────────────────
    p.add_argument("--num-workers", type=int, default=1,
                   help="Parallel session workers (each loads its own model). "
                        "Best for batches of many files. Default: 1.")
    p.add_argument("--beam-size", type=int, default=5,
                   help="ASR beam size. 1=greedy (~2× faster). Default: 5.")
    p.add_argument("--asr-batched", action="store_true",
                   help="Use BatchedInferencePipeline (faster-whisper >= 1.1, ~2-3× faster).")
    p.add_argument("--asr-batch-size", type=int, default=16)
    p.add_argument("--asr-cpu-threads", type=int, default=0,
                   help="Whisper CPU threads per worker. 0=all. "
                        "Set to cpu_count/num_workers when using --num-workers.")
    p.add_argument("--skip-sha1", action="store_true",
                   help="Skip SHA1 hashing of audio files (saves I/O for large files).")
    p.add_argument("--random-seed", type=int, default=0,
                   help="Deterministic seed for clustering. Default: 0.")
    p.add_argument("--generated-at", default="1970-01-01T00:00:00+00:00",
                   help="Stable ISO timestamp stored in JSON records.")
    p.add_argument("--include-runtime-metrics", action="store_true",
                   help="Include wall-clock processing_time_s in records.")
    p.add_argument("--enable-monologues", dest="enable_monologue_extraction",
                   action="store_true", default=True,
                   help="Extract best monologue clip per speaker.")
    p.add_argument("--no-monologues", dest="enable_monologue_extraction",
                   action="store_false",
                   help="Skip monologue extraction.")
    p.add_argument("--metadata-depth", choices=["basic", "full"], default="full",
                   help="Metadata depth requested by CLI/API. Full is default.")
    p.add_argument("--metadata-file", default=None,
                   help="JSON file with user-provided metadata. Supports '*' and speaker labels.")
    p.add_argument("--ask-metadata", dest="ask_metadata", action="store_true", default=True,
                   help="Ask interactively for missing speaker metadata when stdin is a TTY.")
    p.add_argument("--no-ask-metadata", dest="ask_metadata", action="store_false",
                   help="Disable interactive metadata prompts.")
    p.add_argument("--dialect", default=None, help="User-provided dialect; applied globally.")
    p.add_argument("--region", default=None, help="User-provided region; applied globally.")
    p.add_argument("--gender", default=None, help="User-provided gender; applied globally.")
    p.add_argument("--age-band", default=None, help="User-provided age band; applied globally.")
    p.add_argument("--recording-context", default=None,
                   help="User-provided recording context; applied globally.")
    p.add_argument("--consent-status", default=None,
                   help="User-provided consent status; applied globally.")
    return p.parse_args(argv)


# ── helpers ────────────────────────────────────────────────────────────────

def _build_asr_cfg(cfg: PipelineConfig, model_dir: Optional[str]) -> ASRConfig:
    from .offline import whisper_local_path
    model_path = whisper_local_path(model_dir, cfg.model_size) if model_dir else None
    return ASRConfig(
        model_size=cfg.model_size,
        compute_type=cfg.compute_type,
        device=cfg.device if cfg.device != "auto" else "cpu",
        language=cfg.language,
        offline_mode=cfg.offline_mode,
        model_path=model_path,
        beam_size=cfg.beam_size,
        batched=cfg.asr_batched,
        batch_size=cfg.asr_batch_size,
        cpu_threads=cfg.asr_cpu_threads,
    )


def _vad_parallel(
    wavs: Dict[str, np.ndarray],
    sample_rate: int,
    vad_backend: str,
) -> Dict[str, List[Tuple[float, float]]]:
    """Run VAD on multiple waveforms in parallel threads.

    faster than sequential when wavs has 2+ entries because webrtcvad
    releases the GIL in its C extension.
    """
    results: Dict[str, List[Tuple[float, float]]] = {}
    if len(wavs) <= 1:
        for label, wav in wavs.items():
            results[label] = detect_speech(wav, sample_rate,
                                           backend=vad_backend, cfg=VADConfig())
        return results

    with ThreadPoolExecutor(max_workers=len(wavs)) as pool:
        future_map = {
            pool.submit(detect_speech, wav, sample_rate, vad_backend, VADConfig()): label
            for label, wav in wavs.items()
        }
        for fut in as_completed(future_map):
            label = future_map[fut]
            results[label] = fut.result()
    return results


def _vad_union(
    speaker_vad: Dict[str, List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """Merge per-speaker VAD segments into a single sorted union.

    Avoids running VAD a third time on the mixed signal — the union
    of per-speaker segments is equivalent and already computed.
    """
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


def _speaker_pair_alignment_report(pair: SpeakerPairAudio, max_mismatch_s: float) -> Dict:
    """Estimate shared-timeline offset and duration mismatch."""
    durations = {
        label: round(float(audio.duration), 3)
        for label, audio in pair.speakers.items()
    }
    values = list(durations.values())
    mismatch = (max(values) - min(values)) if len(values) >= 2 else 0.0
    labels = list(pair.speakers)
    offset = {"offset_ms": 0, "confidence": 0.0, "lag_samples": 0, "passed": False}
    if len(labels) >= 2:
        a = pair.speakers[labels[0]]
        b = pair.speakers[labels[1]]
        if a.sample_rate == b.sample_rate:
            offset = estimate_offset(a.waveform, b.waveform, a.sample_rate)
    return {
        "method": "cross_correlation",
        "offset_ms": offset["offset_ms"],
        "confidence": offset["confidence"],
        "passed": bool(offset["passed"]),
        "note": (
            "positive offset means second speaker file is delayed relative to first"
            if len(labels) >= 2 else "not enough speakers for alignment"
        ),
        "speaker_start_offsets_s": {label: 0.0 for label in pair.speakers},
        "speaker_durations_s": durations,
        "duration_mismatch_s": round(float(mismatch), 3),
        "max_duration_mismatch_s": float(max_mismatch_s),
    }


def _processing_report(
    cfg: PipelineConfig,
    requested_diarisation_backend: str,
    effective_diarisation_backend: str,
) -> Dict:
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


def _clean_metadata_value(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _provided_field(value: str, source: str = "user_provided") -> Dict:
    return {
        "value": value,
        "confidence": 1.0 if value != "unknown" else 0.0,
        "source": source,
    }


def _normalise_user_metadata_block(raw: Dict, source: str) -> Dict:
    out: Dict[str, Dict] = {}
    for field in USER_METADATA_FIELDS:
        value = _clean_metadata_value(raw.get(field))
        if value:
            out[field] = _provided_field(value, source)
    return out


def _load_user_metadata_file(path: Optional[str]) -> Dict[str, Dict]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--metadata-file must contain a JSON object")

    # Supported shapes:
    # {"*": {"region": "Rajasthan"}, "Speaker_L": {"gender": "female"}}
    # or flat {"region": "Rajasthan", "dialect": "Marwadi"}.
    if any(k in USER_METADATA_FIELDS for k in data):
        return {"*": _normalise_user_metadata_block(data, "metadata_file")}

    out: Dict[str, Dict] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out[str(key)] = _normalise_user_metadata_block(value, "metadata_file")
    return out


def _speaker_labels_for_metadata(input_mode: str, work_items: List) -> List[str]:
    labels = set()
    if input_mode == "speaker_pair":
        for _, _p1, l1, _p2, l2 in work_items:
            labels.add(str(l1))
            labels.add(str(l2))
    elif input_mode == "stereo":
        labels.update({"Speaker_L", "Speaker_R"})
    else:
        labels.add("*")
    return sorted(labels)


def _prompt_metadata(labels: List[str], existing: Dict[str, Dict]) -> Dict[str, Dict]:
    if not sys.stdin.isatty():
        return existing

    out = {k: dict(v) for k, v in existing.items()}
    print("\nSpeaker metadata. Press Enter for unknown. No assumptions made.", file=sys.stderr)
    for label in labels:
        key = label or "*"
        out.setdefault(key, {})
        prompt_label = "all detected speakers" if key == "*" else key
        for field in USER_METADATA_FIELDS:
            if field in out[key]:
                continue
            answer = input(f"{prompt_label} {field.replace('_', ' ')}: ").strip()
            value = answer or "unknown"
            out[key][field] = _provided_field(value, "interactive_prompt")
    return out


def _prepare_user_metadata(cfg: PipelineConfig, input_mode: str, work_items: List) -> Dict:
    metadata = _load_user_metadata_file(cfg.metadata_file)
    cli_values = {field: getattr(cfg, field) for field in USER_METADATA_FIELDS}
    cli_block = _normalise_user_metadata_block(cli_values, "cli")
    if cli_block:
        metadata.setdefault("*", {}).update(cli_block)

    labels = _speaker_labels_for_metadata(input_mode, work_items)
    if cfg.ask_metadata:
        metadata = _prompt_metadata(labels, metadata)
    return metadata


def _apply_user_metadata(
    speaker_meta: Dict[str, Dict],
    user_metadata: Dict,
    speaker_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    if not user_metadata:
        return speaker_meta
    speaker_map = speaker_map or {}
    enriched: Dict[str, Dict] = {}
    for speaker_id, meta in speaker_meta.items():
        label = speaker_map.get(speaker_id, speaker_id)
        provided = {}
        for key in ("*", speaker_id, label):
            provided.update(user_metadata.get(key, {}))
        entry = dict(meta)
        if provided:
            entry["provided_metadata"] = provided
        enriched[speaker_id] = entry
    return enriched


def _apply_metadata_depth(
    speaker_meta: Dict[str, Dict],
    conversation_meta: Dict,
    depth: str,
) -> Tuple[Dict[str, Dict], Dict]:
    if depth != "basic":
        return speaker_meta, conversation_meta

    speaker_keys = {
        "label",
        "total_speaking_time_s",
        "word_count",
        "wpm",
        "pause_frequency_per_min",
        "speaking_style",
        "turn_count",
        "speech_ratio",
        "dominance_score",
        "provided_metadata",
        "language",
    }
    slim_speakers = {
        spk: {k: v for k, v in meta.items() if k in speaker_keys}
        for spk, meta in speaker_meta.items()
    }
    conversation_keys = {
        "turn_count",
        "speaker_count",
        "avg_turn_length_s",
        "total_speech_time_s",
        "quality",
    }
    slim_conversation = {
        k: v for k, v in conversation_meta.items() if k in conversation_keys
    }
    slim_conversation["metadata_depth"] = "basic"
    return slim_speakers, slim_conversation


def _run_downstream(
    wav: np.ndarray,
    sample_rate: int,
    speech_segments: List[Tuple[float, float]],
    turns: List[SpeakerTurn],
    transcriber: Transcriber,
    ft_lid: FastTextLID,
    classifier: Optional[RomanIndicClassifier],
    asr_cfg: ASRConfig,
    speaker_map: Optional[Dict[str, str]] = None,
    cfg: Optional[PipelineConfig] = None,
) -> Tuple:
    """ASR → language → interaction → metadata → monologues."""
    pipeline_cfg = cfg or PipelineConfig()

    transcriber.cfg = asr_cfg
    transcript = transcriber.transcribe(wav, sample_rate)
    log.info("  whisper language=%s segments=%d",
             transcript.language, len(transcript.segments))

    lang_report = detect_language(
        full_text=transcript.text,
        fasttext_lid=ft_lid,
        roman_indic_classifier=classifier,
        transcript_segments=transcript.segments,
        total_duration_s=transcript.duration,
    )

    speaker_lang = detect_language_per_speaker(
        turns=turns,
        transcript_segments=transcript.segments,
        fasttext_lid=ft_lid,
        roman_indic_classifier=classifier,
    )

    interaction_meta, overlap_ratios, _ = extract_interaction_metadata(
        turns, interruption_threshold_s=pipeline_cfg.interruption_threshold_s
    )

    audio_meta = extract_audio_metadata(wav, sample_rate, speech_segments)

    dominance = interaction_meta.get("dominance", {})
    total_dur = float(wav.size / sample_rate)
    speaker_meta = extract_speaker_metadata(
        transcript=transcript,
        turns=turns,
        language=lang_report.primary_language,
        scripts=lang_report.scripts,
        filler_lexicon=FILLERS,
        speaker_labels=speaker_map,
        overlap_ratios=overlap_ratios,
        dominance=dominance,
        total_audio_duration_s=total_dur,
    )
    speaker_meta = _apply_user_metadata(
        speaker_meta,
        pipeline_cfg.user_metadata,
        speaker_map=speaker_map,
    )

    conv_meta = extract_conversation_metadata(transcript, turns)
    speaker_meta, conv_meta = _apply_metadata_depth(
        speaker_meta,
        conv_meta,
        pipeline_cfg.metadata_depth,
    )
    if pipeline_cfg.enable_monologue_extraction:
        monologues = extract_monologues_per_speaker(transcript, turns, MonologueConfig())
        monologue = extract_monologue(transcript, turns, MonologueConfig())
    else:
        monologues = {}
        monologue = None

    return (transcript, lang_report, audio_meta, speaker_meta, conv_meta,
            monologue, monologues, interaction_meta, speaker_lang)


# ── mono processing ────────────────────────────────────────────────────────

def _process_single(
    clip: LoadedAudio,
    transcriber: Transcriber,
    ft_lid: FastTextLID,
    classifier: Optional[RomanIndicClassifier],
    cfg: PipelineConfig,
    dataset_writer: DatasetWriter,
    model_dir: Optional[str] = None,
) -> dict:
    t0 = time.time()
    log.info("processing %s (%.2fs) [mono]", clip.filename, clip.duration)

    qreport = check_mono(clip, min_duration_s=cfg.min_audio_duration_s)
    if not qreport.passed:
        log.warning("  quality errors: %s", qreport.errors)
        if cfg.fail_fast:
            raise RuntimeError(f"Quality check failed: {qreport.errors}")
    if qreport.warnings:
        log.warning("  quality warnings: %s", qreport.warnings)

    pcfg = PreprocessConfig(denoise=cfg.denoise)
    wav = preprocess(clip.waveform, clip.sample_rate, pcfg)
    speech = detect_speech(wav, clip.sample_rate,
                           backend=cfg.vad_backend, cfg=VADConfig())
    log.info("  VAD segments: %d", len(speech))

    requested_diarisation_backend = cfg.diarisation_backend
    effective_diarisation_backend = cfg.diarisation_backend
    if cfg.diarisation_backend == "pyannote":
        try:
            from .offline import pyannote_local_path
            local_pyannote = pyannote_local_path(model_dir) if model_dir else None
            dia = diarise_pyannote(
                wav, clip.sample_rate,
                hf_token=cfg.hf_token,
                min_speakers=cfg.min_speakers if cfg.min_speakers > 1 else None,
                max_speakers=cfg.max_speakers,
                local_model_dir=local_pyannote,
                offline_mode=cfg.offline_mode,
            )
        except RuntimeError as err:
            log.warning("pyannote failed (%s); using kmeans", err)
            effective_diarisation_backend = "kmeans_fallback"
            dia = diarise(wav, clip.sample_rate, speech,
                          cfg=DiarisationConfig(max_speakers=cfg.max_speakers,
                                                min_speakers=cfg.min_speakers,
                                                random_state=cfg.random_seed))
    else:
        dia = diarise(wav, clip.sample_rate, speech,
                      cfg=DiarisationConfig(max_speakers=cfg.max_speakers,
                                            min_speakers=cfg.min_speakers,
                                            random_state=cfg.random_seed))
    log.info("  turns: %d speakers: %d", len(dia), len({t.speaker for t in dia}))

    asr_cfg = _build_asr_cfg(cfg, model_dir)
    (transcript, lang_report, audio_meta, speaker_meta, conv_meta,
     monologue, monologues, interaction_meta, speaker_lang) = _run_downstream(
        wav, clip.sample_rate, speech, dia,
        transcriber, ft_lid, classifier, asr_cfg, cfg=cfg,
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
        ),
        processing=_processing_report(
            cfg, requested_diarisation_backend, effective_diarisation_backend
        ),
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


# ── speaker-pair / stereo processing ──────────────────────────────────────

def _process_speaker_pair(
    pair: SpeakerPairAudio,
    transcriber: Transcriber,
    ft_lid: FastTextLID,
    classifier: Optional[RomanIndicClassifier],
    cfg: PipelineConfig,
    dataset_writer: DatasetWriter,
    model_dir: Optional[str] = None,
) -> dict:
    t0 = time.time()
    log.info("processing session '%s' [%s: %s]",
             pair.session_name, pair.recording_type,
             " + ".join(pair.speakers.keys()))

    qreport = check_speaker_pair(
        pair,
        min_duration_s=cfg.min_audio_duration_s,
        max_duration_mismatch_s=cfg.max_duration_mismatch_s,
    )
    if not qreport.passed:
        log.warning("  quality errors: %s", qreport.errors)
    if qreport.warnings:
        log.warning("  quality warnings: %s", qreport.warnings)
    alignment = _speaker_pair_alignment_report(pair, cfg.max_duration_mismatch_s)

    pcfg = PreprocessConfig(denoise=cfg.denoise)
    speaker_wavs: Dict[str, np.ndarray] = {
        label: preprocess(audio.waveform, audio.sample_rate, pcfg)
        for label, audio in pair.speakers.items()
    }
    mixed_wav = preprocess(pair.mixed.waveform, pair.mixed.sample_rate, pcfg)

    # Parallel VAD on both speakers simultaneously (2× faster than sequential).
    speaker_vad = _vad_parallel(speaker_wavs, pair.mixed.sample_rate, cfg.vad_backend)
    for label, segs in speaker_vad.items():
        log.info("  VAD [%s]: %d segments", label, len(segs))

    # Derive mixed speech from union of speaker VADs — skip the 3rd VAD call.
    mixed_speech = _vad_union(speaker_vad)
    log.info("  VAD [mixed union]: %d segments", len(mixed_speech))

    dia, speaker_map = diarise_from_speaker_vad(speaker_vad=speaker_vad, merge_gap_s=0.4)
    log.info("  turns: %d speakers: %d (ground-truth)", len(dia), len(speaker_map))

    asr_cfg = _build_asr_cfg(cfg, model_dir)
    (transcript, lang_report, audio_meta, speaker_meta, conv_meta,
     monologue, monologues, interaction_meta, speaker_lang) = _run_downstream(
        mixed_wav, pair.mixed.sample_rate, mixed_speech, dia,
        transcriber, ft_lid, classifier, asr_cfg,
        speaker_map=speaker_map, cfg=cfg,
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
        ),
        processing=_processing_report(cfg, "speaker_vad", "speaker_vad"),
        input_mode=input_mode,
        speaker_sources=speaker_sources,
        session_name=pair.session_name,
        speaker_map=speaker_map,
        source_files=source_paths,
        generated_at=cfg.generated_at,
        skip_sha1=cfg.skip_sha1,
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


# ── worker-process globals (set by initializer) ────────────────────────────
# These are process-local — each worker in the pool gets its own copy.

_W_TRANSCRIBER: Optional[Transcriber] = None
_W_FT_LID: Optional[FastTextLID] = None
_W_CLASSIFIER = None
_W_CFG: Optional[PipelineConfig] = None
_W_DATASET_WRITER: Optional[DatasetWriter] = None
_W_MODEL_DIR: Optional[str] = None


def _worker_init(cfg_dict: dict, model_dir: Optional[str], output_root: str,
                 ft_path: Optional[str]) -> None:
    """Runs once per worker process. Loads heavy models into process-local globals."""
    global _W_TRANSCRIBER, _W_FT_LID, _W_CLASSIFIER, _W_CFG
    global _W_DATASET_WRITER, _W_MODEL_DIR

    _W_CFG = PipelineConfig.from_dict(cfg_dict)
    _W_MODEL_DIR = model_dir
    np.random.seed(_W_CFG.random_seed)

    # ASR — pre-warm so first session doesn't pay model load cost.
    asr_cfg = _build_asr_cfg(_W_CFG, model_dir)
    _W_TRANSCRIBER = Transcriber(asr_cfg)
    _W_TRANSCRIBER._load()

    _W_FT_LID = FastTextLID(path=ft_path)

    _W_CLASSIFIER = None
    if _W_CFG.classifier != "off":
        try:
            from .roman_indic_classifier import RomanIndicClassifier
            c = RomanIndicClassifier(cache_path=_W_CFG.classifier_cache)
            _W_CLASSIFIER = c if c.available() else None
        except Exception:
            pass

    _W_DATASET_WRITER = DatasetWriter(
        output_root=output_root,
        output_mode=_W_CFG.output_mode,
        output_format=_W_CFG.output_format,
        dataset_name=_W_CFG.dataset_name,
    )


def _worker_run_mono(path: str) -> dict:
    clip = load_audio(path)
    if clip is None:
        return {"error": f"load_failed:{path}"}
    try:
        return _process_single(
            clip, _W_TRANSCRIBER, _W_FT_LID, _W_CLASSIFIER,
            _W_CFG, _W_DATASET_WRITER, _W_MODEL_DIR,
        )
    except Exception as exc:
        log.exception("worker failed on %s: %s", path, exc)
        return {"error": str(exc), "path": path}


def _worker_run_pair(item: tuple) -> dict:
    session_name, p1, l1, p2, l2 = item
    pair = load_speaker_pair(p1, l1, p2, l2, session_name=session_name)
    if pair is None:
        return {"error": f"load_failed:{session_name}"}
    try:
        return _process_speaker_pair(
            pair, _W_TRANSCRIBER, _W_FT_LID, _W_CLASSIFIER,
            _W_CFG, _W_DATASET_WRITER, _W_MODEL_DIR,
        )
    except Exception as exc:
        log.exception("worker failed on session '%s': %s", session_name, exc)
        return {"error": str(exc), "session": session_name}


def _worker_run_stereo(f: str) -> dict:
    session_name = os.path.splitext(os.path.basename(f))[0]
    pair = load_stereo_as_pair(f, session_name=session_name)
    if pair is None:
        return {"error": f"load_failed:{session_name}"}
    try:
        return _process_speaker_pair(
            pair, _W_TRANSCRIBER, _W_FT_LID, _W_CLASSIFIER,
            _W_CFG, _W_DATASET_WRITER, _W_MODEL_DIR,
        )
    except Exception as exc:
        log.exception("worker failed stereo session '%s': %s", session_name, exc)
        return {"error": str(exc), "session": session_name}


# ── main run loop ──────────────────────────────────────────────────────────

def run(argv=None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    cfg = PipelineConfig.from_namespace(args)
    if getattr(args, "allow_model_downloads", False):
        cfg.offline_mode = False
    np.random.seed(cfg.random_seed)

    from .offline import default_model_dir, fasttext_local_path
    model_dir: Optional[str] = cfg.model_dir or (
        default_model_dir() if cfg.offline_mode else None
    )
    if cfg.offline_mode:
        log.info("OFFLINE MODE — models must be under %s", model_dir)

    os.makedirs(args.output, exist_ok=True)

    # Resolve input mode + collect session list
    input_mode = cfg.input_type
    pairs: List = []

    if input_mode in ("speaker_pair", "auto"):
        pairs = detect_and_group_pairs(args.input)
        if pairs:
            log.info("detected %d speaker-pair session(s)", len(pairs))
            if input_mode == "auto":
                input_mode = "speaker_pair"
        elif input_mode == "speaker_pair":
            log.error("speaker_pair mode: no pairs found under %s", args.input)
            return 2

    if input_mode == "auto":
        stereo_files = detect_stereo_files(args.input)
        if stereo_files:
            log.info("detected %d stereo session file(s)", len(stereo_files))
            input_mode = "stereo"
        else:
            input_mode = "mono"

    log.info("input_mode=%s  output_mode=%s  num_workers=%d  beam_size=%d  batched=%s",
             input_mode, cfg.output_mode, cfg.num_workers, cfg.beam_size, cfg.asr_batched)

    # Resolve fasttext path once (used in initializer too)
    ft_path = cfg.fasttext_model
    if ft_path is None and model_dir:
        ft_path = fasttext_local_path(model_dir)

    # Collect work items
    if input_mode == "stereo":
        work_items = (
            stereo_files if "stereo_files" in locals()
            else detect_stereo_files(args.input) or list(iter_audio_files(args.input))
        )
        worker_fn = _worker_run_stereo
    elif input_mode == "speaker_pair":
        work_items = pairs
        worker_fn = _worker_run_pair
    else:
        work_items = list(iter_audio_files(args.input))
        worker_fn = _worker_run_mono

    if not work_items:
        log.error("no sessions found under %s", args.input)
        return 2

    cfg.user_metadata = _prepare_user_metadata(cfg, input_mode, work_items)

    total = len(work_items)
    log.info("sessions to process: %d", total)

    legacy_writer = BatchWriter(
        output_dir=args.output,
        fmt=cfg.output_format,
        dataset_name=cfg.dataset_name,
    )

    n_ok = n_fail = 0

    if cfg.num_workers > 1 and total > 1:
        # ── Multi-process mode ─────────────────────────────────────────
        # Each worker loads its own Whisper model (process-local globals).
        # Divide cpu_threads evenly if not explicitly set.
        if cfg.asr_cpu_threads == 0:
            import os as _os
            available = _os.cpu_count() or 1
            per_worker = max(1, available // cfg.num_workers)
            cfg_for_workers = PipelineConfig.from_dict({
                **cfg.to_dict(),
                "asr_cpu_threads": per_worker,
            })
        else:
            cfg_for_workers = cfg

        log.info("launching %d worker processes", cfg.num_workers)
        with ProcessPoolExecutor(
            max_workers=cfg.num_workers,
            initializer=_worker_init,
            initargs=(cfg_for_workers.to_dict(), model_dir, args.output, ft_path),
        ) as pool:
            future_to_item = {pool.submit(worker_fn, item): item
                              for item in work_items}
            for i, fut in enumerate(as_completed(future_to_item), 1):
                try:
                    record = fut.result()
                except Exception as exc:
                    log.error("[%d/%d] worker exception: %s", i, total, exc)
                    n_fail += 1
                    if cfg.fail_fast:
                        pool.shutdown(wait=False, cancel_futures=True)
                        return 1
                    continue

                if "error" in record:
                    log.error("[%d/%d] %s", i, total, record["error"])
                    n_fail += 1
                    if cfg.fail_fast:
                        pool.shutdown(wait=False, cancel_futures=True)
                        return 1
                else:
                    legacy_writer.write(record)
                    n_ok += 1
                    log.info("[%d/%d] done: %s  (%.1fs)",
                             i, total,
                             record.get("session_name", "?"),
                             record.get("processing_time_s", 0))

    else:
        # ── Single-process mode (default) ──────────────────────────────
        # Create shared components once.
        transcriber = Transcriber()
        ft_lid = FastTextLID(path=ft_path)

        classifier: Optional[RomanIndicClassifier] = None
        if cfg.classifier != "off":
            try:
                classifier = RomanIndicClassifier(cache_path=cfg.classifier_cache)
                if not classifier.available():
                    classifier = None
            except Exception as err:
                log.warning("classifier init failed (%s); lexicon only", err)

        dataset_writer = DatasetWriter(
            output_root=args.output,
            output_mode=cfg.output_mode,
            output_format=cfg.output_format,
            dataset_name=cfg.dataset_name,
        )

        # Pre-warm ASR model so first session doesn't pay cold-start cost.
        asr_cfg = _build_asr_cfg(cfg, model_dir)
        transcriber.cfg = asr_cfg
        log.info("pre-warming ASR model ...")
        transcriber._load()

        shared = dict(transcriber=transcriber, ft_lid=ft_lid,
                      classifier=classifier, cfg=cfg,
                      dataset_writer=dataset_writer, model_dir=model_dir)

        try:
            for i, item in enumerate(work_items, 1):
                log.info("[%d/%d] %s", i, total,
                         item if isinstance(item, str) else item[0])
                try:
                    if input_mode == "stereo":
                        session_name = os.path.splitext(os.path.basename(item))[0]
                        pair = load_stereo_as_pair(item, session_name=session_name)
                        if pair is None:
                            raise RuntimeError(f"failed to load stereo {item}")
                        record = _process_speaker_pair(pair, **shared)
                    elif input_mode == "speaker_pair":
                        session_name, p1, l1, p2, l2 = item
                        pair = load_speaker_pair(p1, l1, p2, l2,
                                                 session_name=session_name)
                        if pair is None:
                            raise RuntimeError(f"failed to load pair {session_name}")
                        record = _process_speaker_pair(pair, **shared)
                    else:
                        clip = load_audio(item)
                        if clip is None:
                            raise RuntimeError(f"failed to load {item}")
                        record = _process_single(clip, **shared)
                except Exception as err:
                    log.exception("failed: %s", err)
                    n_fail += 1
                    if cfg.fail_fast:
                        return 1
                    continue
                legacy_writer.write(record)
                n_ok += 1
                log.info("  done (%.1fs)", record.get("processing_time_s", 0))
        finally:
            closed = legacy_writer.close()
            if closed:
                log.info("final output: %s", closed)

    if cfg.num_workers > 1:
        legacy_writer.close()

    log.info("done. processed=%d  failed=%d", n_ok, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
