"""Shared downstream processing helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..audio_loader import LoadedAudio
from ..code_switch import enrich_code_switch_segments
from ..config import PipelineConfig, _detect_device
from ..confidence import annotate_segments_with_confidence
from ..interaction_metadata import OverlapSegment
from ..language_detection import FastTextLID, detect_language, detect_language_per_speaker
from ..metadata_extraction import (
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)
from ..monologue_extractor import MonologueConfig, extract_monologue, extract_monologues_per_speaker
from ..offline import whisper_local_path
from ..overlap import annotate_segments_with_overlaps, detect_overlaps
from ..punctuation import apply_punctuation_metadata
from ..roman_indic_classifier import RomanIndicClassifier
from ..steps.interaction import compute_interaction
from ..snr import annotate_segments_with_snr
from ..transcription import ASRConfig, FILLERS, Transcript, Transcriber
from ..diarisation import SpeakerTurn
from ..utils.metadata_fields import inferred_field, provided_field

SPEAKER_METADATA_FIELDS = (
    "accent",
    "dialect",
    "region",
    "gender",
    "age_band",
    "recording_context",
    "consent_status",
)
CONVERSATION_METADATA_FIELDS = ("domain",)
USER_METADATA_FIELDS = SPEAKER_METADATA_FIELDS + CONVERSATION_METADATA_FIELDS


def compute_total_speech_duration(speech_segments) -> float:
    """Sum VAD speech segment durations in seconds."""
    return float(sum(float(end) - float(start) for start, end in speech_segments))


_LANG_PROMPTS: Dict[str, str] = {
    # Pure Hindi — short, neutral, no code-switch bias.
    "hi": (
        "यह एक हिंदी बातचीत है। "
        "वक्ता हिंदी में बोल रहे हैं।"
    ),
    # Hinglish — explicitly signals Hindi-English mixing so Whisper does
    # not hallucinate one language over the other.
    "hinglish": (
        "यह हिंदी और अंग्रेजी मिश्रित (Hinglish) बातचीत है। "
        "वक्ता हिंदी और अंग्रेजी दोनों में बोलते हैं। "
        "This is a Hindi-English mixed conversation."
    ),
    # Marwadi — Devanagari script; Whisper uses the Hindi model.
    "mwr": (
        "यह एक राजस्थानी / मारवाड़ी बातचीत है। "
        "वक्ता मारवाड़ी और हिंदी मिश्रित भाषा में बोल सकते हैं।"
    ),
    "ta": "இது தமிழ் மற்றும் ஆங்கிலம் கலந்த உரையாடல்.",
    "te": "ఇది తెలుగు మరియు ఇంగ్లీష్ మిశ్రిత సంభాషణ.",
    "mr": "हे मराठी आणि इंग्रजी मिश्रित संभाषण आहे.",
    "bn": "এটি বাংলা এবং ইংরেজি মিশ্রিত কথোপকথন।",
    "gu": "આ ગુજરાતી અને અંગ્રેજી મિશ્રિત વાર્તાલાપ છે.",
    "pa": (
        "ਇਹ ਇੱਕ ਪੰਜਾਬੀ ਗੱਲਬਾਤ ਹੈ। "
        "ਬੁਲਾਰੇ ਪੰਜਾਬੀ ਅਤੇ ਕਦੇ-ਕਦੇ ਅੰਗਰੇਜ਼ੀ ਵਿੱਚ ਬੋਲ ਸਕਦੇ ਹਨ।"
    ),
}


_GENERIC_MULTILINGUAL_PROMPT = (
    "This conversation may include Hindi, English, or other Indian languages. "
    "यह बातचीत हिंदी, अंग्रेजी या अन्य भारतीय भाषाओं में हो सकती है।"
)

# Map logical / dialect codes → BCP-47 codes accepted by faster-whisper.
# Whisper has no "hinglish" or "mwr" model; Hindi is the closest.
_WHISPER_LANG_ALIASES: Dict[str, str] = {
    "hinglish": "hi",
    "mwr": "hi",   # Marwadi written in Devanagari; Hindi model is closest
}


def _normalise_whisper_language(language: Optional[str]) -> Optional[str]:
    """Return the faster-whisper-compatible language code for *language*."""
    if not language:
        return language
    return _WHISPER_LANG_ALIASES.get(language.lower(), language)


def _resolve_initial_prompt(cfg: PipelineConfig) -> Optional[str]:
    if getattr(cfg, "initial_prompt", None):
        return cfg.initial_prompt
    lang = (cfg.language or "").lower().split("-")[0]
    if lang:
        return _LANG_PROMPTS.get(lang)
    # language=None (auto-detect): inject generic multilingual prompt so
    # Whisper doesn't default to English-only decoding on mixed-script audio.
    return _GENERIC_MULTILINGUAL_PROMPT


def build_asr_cfg(cfg: PipelineConfig, model_dir: Optional[str]) -> ASRConfig:
    model_path = whisper_local_path(model_dir, cfg.model_size) if model_dir else None
    device = cfg.device if cfg.device != "auto" else _detect_device()
    # int8 is CPU-only quantisation; CUDA needs float16 (or float32).
    compute_type = cfg.compute_type
    if device == "cuda" and compute_type == "int8":
        compute_type = "float16"
    return ASRConfig(
        model_size=cfg.model_size,
        compute_type=compute_type,
        device=device,
        language=_normalise_whisper_language(cfg.language),
        offline_mode=cfg.offline_mode,
        model_path=model_path,
        beam_size=cfg.beam_size,
        batched=cfg.asr_batched,
        batch_size=cfg.asr_batch_size,
        cpu_threads=cfg.asr_cpu_threads,
        initial_prompt=_resolve_initial_prompt(cfg),
        no_speech_threshold=getattr(cfg, "no_speech_threshold", 0.6),
        compression_ratio_threshold=getattr(cfg, "compression_ratio_threshold", 2.4),
        log_prob_threshold=getattr(cfg, "log_prob_threshold", -1.0),
        condition_on_previous_text=getattr(cfg, "condition_on_previous_text", False),
    )


def _clean_metadata_value(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalise_user_metadata_block(raw: Dict, source: str, fields: Tuple[str, ...]) -> Dict:
    out: Dict[str, Dict] = {}
    for field in fields:
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
        out: Dict[str, Dict] = {}
        speaker_block = normalise_user_metadata_block(
            data,
            "metadata_file",
            SPEAKER_METADATA_FIELDS,
        )
        conversation_block = normalise_user_metadata_block(
            data,
            "metadata_file",
            CONVERSATION_METADATA_FIELDS,
        )
        if speaker_block:
            out["*"] = speaker_block
        if conversation_block:
            out["__conversation__"] = conversation_block
        return out

    out: Dict[str, Dict] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            if str(key) == "conversation":
                out["__conversation__"] = normalise_user_metadata_block(
                    value,
                    "metadata_file",
                    CONVERSATION_METADATA_FIELDS,
                )
            else:
                out[str(key)] = normalise_user_metadata_block(
                    value,
                    "metadata_file",
                    SPEAKER_METADATA_FIELDS,
                )
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
        for field in SPEAKER_METADATA_FIELDS:
            if field in out[key]:
                continue
            answer = input(f"{prompt_label} {field.replace('_', ' ')}: ").strip()
            value = answer or "unknown"
            out[key][field] = provided_field(value, "interactive_prompt")
    out.setdefault("__conversation__", {})
    if "domain" not in out["__conversation__"]:
        answer = input("conversation domain: ").strip()
        if answer:
            out["__conversation__"]["domain"] = provided_field(
                answer,
                "interactive_prompt",
            )
    return out


def prepare_user_metadata(cfg: PipelineConfig, input_mode: str, work_items: List) -> Dict:
    metadata = load_user_metadata_file(cfg.metadata_file)
    speaker_cli_values = {
        field: getattr(cfg, field)
        for field in SPEAKER_METADATA_FIELDS
    }
    conversation_cli_values = {
        field: getattr(cfg, field)
        for field in CONVERSATION_METADATA_FIELDS
    }
    speaker_cli_block = normalise_user_metadata_block(
        speaker_cli_values,
        "cli",
        SPEAKER_METADATA_FIELDS,
    )
    conversation_cli_block = normalise_user_metadata_block(
        conversation_cli_values,
        "cli",
        CONVERSATION_METADATA_FIELDS,
    )
    if speaker_cli_block:
        metadata.setdefault("*", {}).update(speaker_cli_block)
    if conversation_cli_block:
        metadata.setdefault("__conversation__", {}).update(conversation_cli_block)

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
            for field, value in provided.items():
                entry[field] = value
        entry.setdefault("accent", inferred_field("unknown", 0.0))
        entry.setdefault("region", inferred_field("unknown", 0.0))
        entry.setdefault("dialect", inferred_field("unknown", 0.0))
        enriched[speaker_id] = entry
    return enriched


def apply_conversation_metadata(conversation_meta: Dict, user_metadata: Dict) -> Dict:
    if not user_metadata:
        conversation_meta.setdefault("domain", inferred_field("unknown", 0.0))
        return conversation_meta
    provided = dict(user_metadata.get("__conversation__", {}))
    entry = dict(conversation_meta)
    if provided:
        entry["provided_metadata"] = {
            **dict(entry.get("provided_metadata") or {}),
            **provided,
        }
        for field, value in provided.items():
            entry[field] = value
    entry.setdefault("domain", inferred_field("unknown", 0.0))
    return entry


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
        "speaker_id",
        "accent",
        "region",
        "dialect",
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
        "domain",
    }
    slim_conversation = {
        k: v for k, v in conversation_meta.items() if k in conversation_keys
    }
    slim_conversation["metadata_depth"] = "basic"
    return slim_speakers, slim_conversation


def _audio_source_info(
    source_audios: Optional[List[LoadedAudio]],
    processed_sample_rate_hz: int,
) -> Dict:
    clips = [clip for clip in (source_audios or []) if clip is not None]
    source_rates = [int(clip.source_sample_rate or clip.sample_rate) for clip in clips]
    bit_depths = [int(clip.sample_width_bits) for clip in clips if clip.sample_width_bits]
    channels = [int(clip.channels) for clip in clips if clip.channels]
    codecs = sorted({
        str(clip.encoding).strip().lower()
        for clip in clips
        if str(clip.encoding or "").strip()
    })
    formats = sorted({
        Path(str(clip.path)).suffix.lower().lstrip(".")
        for clip in clips
        if Path(str(clip.path)).suffix
    })

    sample_rate_hz = None
    if source_rates:
        sample_rate_hz = source_rates[0] if len(set(source_rates)) == 1 else max(source_rates)

    return {
        "sample_rate_hz": int(sample_rate_hz or processed_sample_rate_hz),
        "processed_sample_rate_hz": int(processed_sample_rate_hz),
        "bit_depth": int(bit_depths[0]) if len(set(bit_depths)) == 1 and bit_depths else (
            int(max(bit_depths)) if bit_depths else None
        ),
        "channels": int(channels[0]) if len(set(channels)) == 1 and channels else (
            int(max(channels)) if channels else 1
        ),
        "codec": codecs[0] if len(codecs) == 1 else ("mixed" if codecs else None),
        "container_format": formats[0] if len(formats) == 1 else ("mixed" if formats else None),
    }


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
    source_audios: Optional[List[LoadedAudio]] = None,
    transcript_override: Optional[Transcript] = None,
):
    """ASR -> language -> interaction -> metadata -> monologues."""
    pipeline_cfg = cfg or PipelineConfig()

    if transcript_override is None:
        transcriber.cfg = asr_cfg
        transcript = transcriber.transcribe(wav, sample_rate)
    else:
        transcript = transcript_override

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
    apply_punctuation_metadata(
        transcript.segments,
        enabled=bool(getattr(pipeline_cfg, "punctuation_enabled", True)),
    )
    enrich_code_switch_segments(transcript.segments)

    interaction_meta, overlap_ratios, overlaps = compute_interaction(
        turns, interruption_threshold_s=pipeline_cfg.interruption_threshold_s
    )

    audio_meta = extract_audio_metadata(
        wav,
        sample_rate,
        speech_segments,
        source_info=_audio_source_info(source_audios, sample_rate),
    )
    # Build VAD mask for SNR-04 (speech-only frame estimation).
    vad_mask: Optional[np.ndarray] = None
    if speech_segments:
        n_samples = len(wav)
        _mask = np.zeros(n_samples, dtype=bool)
        for seg_start, seg_end in speech_segments:
            s = max(0, int(float(seg_start) * sample_rate))
            e = min(n_samples, int(float(seg_end) * sample_rate))
            _mask[s:e] = True
        vad_mask = _mask

    annotate_segments_with_snr(transcript.segments, wav, sample_rate, vad_mask=vad_mask)

    # OVER-01..04: detect overlapping speech regions and annotate segments.
    overlap_regions = detect_overlaps(turns, segments=transcript.segments)
    annotate_segments_with_overlaps(transcript.segments, overlap_regions)

    annotate_segments_with_confidence(transcript.segments, overlap_regions)

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
    conv_meta = apply_conversation_metadata(conv_meta, pipeline_cfg.user_metadata)
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
