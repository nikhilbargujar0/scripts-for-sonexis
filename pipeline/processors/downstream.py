"""Shared downstream processing helpers."""
from __future__ import annotations

import json
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import PipelineConfig
from ..interaction_metadata import OverlapSegment
from ..language_detection import FastTextLID, detect_language, detect_language_per_speaker
from ..metadata_extraction import (
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)
from ..monologue_extractor import MonologueConfig, extract_monologue, extract_monologues_per_speaker
from ..offline import whisper_local_path
from ..roman_indic_classifier import RomanIndicClassifier
from ..steps.interaction import compute_interaction
from ..transcription import ASRConfig, FILLERS, Transcript, Transcriber
from ..diarisation import SpeakerTurn
from ..utils.metadata_fields import provided_field

USER_METADATA_FIELDS = (
    "dialect",
    "region",
    "gender",
    "age_band",
    "recording_context",
    "consent_status",
)


def build_asr_cfg(cfg: PipelineConfig, model_dir: Optional[str]) -> ASRConfig:
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


def _clean_metadata_value(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalise_user_metadata_block(raw: Dict, source: str) -> Dict:
    out: Dict[str, Dict] = {}
    for field in USER_METADATA_FIELDS:
        value = _clean_metadata_value(raw.get(field))
        if value:
            out[field] = provided_field(value, source)
    return out


def load_user_metadata_file(path: Optional[str]) -> Dict[str, Dict]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--metadata-file must contain a JSON object")
    if any(k in USER_METADATA_FIELDS for k in data):
        return {"*": normalise_user_metadata_block(data, "metadata_file")}

    out: Dict[str, Dict] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out[str(key)] = normalise_user_metadata_block(value, "metadata_file")
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
            out[key][field] = provided_field(value, "interactive_prompt")
    return out


def prepare_user_metadata(cfg: PipelineConfig, input_mode: str, work_items: List) -> Dict:
    metadata = load_user_metadata_file(cfg.metadata_file)
    cli_values = {field: getattr(cfg, field) for field in USER_METADATA_FIELDS}
    cli_block = normalise_user_metadata_block(cli_values, "cli")
    if cli_block:
        metadata.setdefault("*", {}).update(cli_block)

    labels = _speaker_labels_for_metadata(input_mode, work_items)
    if cfg.ask_metadata:
        metadata = _prompt_metadata(labels, metadata)
    return metadata


def apply_user_metadata(
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


def apply_metadata_depth(
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
        "accent",
        "gender",
        "age_band",
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
        "topic",
        "sub_topic",
    }
    slim_conversation = {
        k: v for k, v in conversation_meta.items() if k in conversation_keys
    }
    slim_conversation["metadata_depth"] = "basic"
    return slim_speakers, slim_conversation


def run_downstream(
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
):
    """ASR -> language -> interaction -> metadata -> monologues."""
    pipeline_cfg = cfg or PipelineConfig()

    transcriber.cfg = asr_cfg
    transcript = transcriber.transcribe(wav, sample_rate)

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

    interaction_meta, overlap_ratios, overlaps = compute_interaction(
        turns, interruption_threshold_s=pipeline_cfg.interruption_threshold_s
    )

    audio_meta = extract_audio_metadata(wav, sample_rate, speech_segments)

    dominance = interaction_meta.get("dominance", {})
    total_dur = float(wav.size / sample_rate) if sample_rate else 0.0
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
    speaker_meta = apply_user_metadata(
        speaker_meta,
        pipeline_cfg.user_metadata,
        speaker_map=speaker_map,
    )

    conv_meta = extract_conversation_metadata(transcript, turns)
    speaker_meta, conv_meta = apply_metadata_depth(
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

    return (
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
    )
