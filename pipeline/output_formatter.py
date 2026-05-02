"""output_formatter.py

Assemble the final JSON dataset record for one session (schema v3.0.0).

Schema additions over v1:
  - metadata.interaction  ← InteractionMetadata block
  - timeline              ← [{speaker, label, start, end}] flat list
  - speaker_transcripts   ← {speaker_id: plain text}
  - conversation_transcript ← [{speaker, label, start, end, text}]
  - monologues            ← {speaker_id: MonologueSample | null}
  - quality               ← QualityReport dict
  - per-speaker language in metadata.speakers[*].language
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

from .diarisation import SpeakerTurn
from .language_detection import LanguageReport
from .monologue_extractor import Monologue
from .steps.language import build_code_switch_report
from .transcription import Transcript, normalise_transcript
from .utils.dataset_purpose import infer_dataset_purpose


SCHEMA_VERSION = "3.0.0"
PIPELINE_VERSION = "1.0.0"
DATASET_SCHEMA_VERSION = "3.0.0"

_LANG_TO_BCP47: Dict[str, str] = {
    "hi": "hi-Deva",
    "pa": "pa-Guru",
    "mr": "mr-Deva",
    "en": "en-IN",
    "hinglish": "hi-Latn",
    "ur": "ur-Arab",
    "gu": "gu-Gujr",
    "bn": "bn-Beng",
    "ta": "ta-Taml",
    "te": "te-Telu",
    "kn": "kn-Knda",
    "ml": "ml-Mlym",
    "mwr": "mwr-Latn",
}


def _bcp47_normalize(lang: str) -> str:
    if not lang or "-" in str(lang):
        return str(lang or "")
    return _LANG_TO_BCP47.get(str(lang), str(lang))


def _compute_config_hash(cfg_dict: Dict) -> str:
    canonical = json.dumps(cfg_dict or {}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_audio_fingerprint(wav: "np.ndarray", sample_rate: int) -> str:
    """AUDQ-03: SHA-256 of the first 5 seconds of waveform bytes.

    Stable across re-encodings of the same audio content; useful for
    detecting train/test leakage across sessions.
    """
    import numpy as _np
    five_s = int(5 * sample_rate)
    snippet = _np.asarray(wav[:five_s], dtype=_np.float32)
    return hashlib.sha256(snippet.tobytes()).hexdigest()


def _collect_model_versions(cfg_dict: Dict) -> Dict:
    versions: Dict[str, str] = {}
    for cfg_key, model_key in (
        ("model_size", "whisper"),
        ("diarisation_backend", "diarisation"),
        ("vad_backend", "vad"),
        ("fasttext_model", "fasttext_lid"),
        ("language_profile", "language_profile"),
    ):
        value = (cfg_dict or {}).get(cfg_key)
        if value:
            versions[model_key] = str(value)
    return versions


def _interaction_aliases(meta: Optional[Dict]) -> Dict:
    out = dict(meta or {})
    response = out.get("response_latency") or {}
    out.setdefault("turn_count", int(out.get("total_turns") or 0))
    out.setdefault("interruptions", int(out.get("interruption_count") or 0))
    out.setdefault("overlap_duration", float(out.get("overlap_duration_s") or 0.0))
    out.setdefault("avg_response_latency", response.get("mean_s"))
    out.setdefault("speaker_dominance", out.get("dominance") or {})
    return out


def _language_aliases(report: LanguageReport) -> Dict:
    out = report.to_dict()
    raw_segments = out.get("language_segments") or out.get("per_segment") or []
    segments: List[Dict] = []
    for seg in raw_segments:
        lang = seg.get("lang") or seg.get("language") or seg.get("label") or "unknown"
        segments.append({
            "start": round(float(seg.get("start", 0.0) or 0.0), 3),
            "end": round(float(seg.get("end", 0.0) or 0.0), 3),
            "lang": str(lang),
            "confidence": round(float(seg.get("confidence", out.get("confidence", 0.5)) or 0.5), 3),
        })
    patterns: List[str] = []
    switch_count = 0
    for a, b in zip(segments, segments[1:]):
        if a["lang"] != b["lang"]:
            switch_count += 1
            pattern = f"{a['lang']}→{b['lang']}"
            if pattern not in patterns:
                patterns.append(pattern)
    out.setdefault("segments", segments)
    out.setdefault("switch_count", switch_count)
    out.setdefault("patterns", patterns)
    return out


def _file_descriptor(path: str, skip_sha1: bool = False) -> Dict:
    """Stat + optional SHA1 for path. Virtual paths handled gracefully.

    skip_sha1=True skips reading the entire file just to hash it — saves
    significant I/O for large audio files (1-hour WAV ≈ 1 GB).
    """
    if not os.path.isfile(path):
        return {
            "path": os.path.abspath(path),
            "name": os.path.basename(path),
            "size_bytes": None,
            "sha1": None,
            "note": "virtual_mixed_audio",
        }
    stat = os.stat(path)
    sha = None
    if not skip_sha1:
        with open(path, "rb") as f:
            sha = hashlib.sha1(f.read()).hexdigest()  # noqa: S324
    return {
        "path": os.path.abspath(path),
        "name": os.path.basename(path),
        "size_bytes": int(stat.st_size),
        "sha1": sha,
    }


def _build_timeline(
    turns: List[SpeakerTurn],
    speaker_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Flat sorted list of {speaker, label, start, end} from turns."""
    sm = speaker_map or {}
    return [
        {
            "speaker": t.speaker,
            "label": sm.get(t.speaker, t.speaker),
            "start": round(t.start, 3),
            "end": round(t.end, 3),
            "confidence": round(t.confidence, 4),
        }
        for t in sorted(turns, key=lambda x: x.start)
    ]


def _build_conversation_transcript(
    transcript: Transcript,
    turns: List[SpeakerTurn],
    speaker_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """[{speaker, label, start, end, text}] — one entry per transcript segment
    mapped to its diarised speaker.
    """
    sm = speaker_map or {}
    sorted_turns = sorted(turns, key=lambda t: t.start)

    def find_speaker(mid: float) -> str:
        for turn in sorted_turns:
            if turn.start <= mid <= turn.end:
                return turn.speaker
        if sorted_turns:
            return min(sorted_turns,
                       key=lambda t: min(abs(t.start - mid), abs(t.end - mid))).speaker
        return "SPEAKER_00"

    result = []
    for seg in transcript.segments:
        if not seg.text.strip():
            continue
        forced_speaker = getattr(seg, "ground_truth_speaker", None)
        mid = (seg.start + seg.end) / 2.0
        spk = forced_speaker or find_speaker(mid)
        result.append({
            "speaker": spk,
            "label": sm.get(spk, spk),
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "language": _bcp47_normalize(seg.language),
            "quality_score": round(float(seg.quality_score), 4),
        })
    return result


def _build_speaker_transcripts(
    conversation_transcript: List[Dict],
) -> Dict[str, str]:
    """Concatenate transcript entries per speaker."""
    chunks: Dict[str, List[str]] = {}
    for entry in conversation_transcript:
        spk = entry["speaker"]
        chunks.setdefault(spk, []).append(entry["text"])
    return {spk: " ".join(texts) for spk, texts in chunks.items()}


def _build_annotations(
    transcript: Transcript,
    turns: List[SpeakerTurn],
    language: LanguageReport,
    monologue: Optional[Monologue],
    monologues: Optional[Dict[str, Optional[Monologue]]],
    validation: Optional[Dict],
) -> Dict:
    return {
        "transcript_available": bool(str(transcript.text or "").strip()),
        "speaker_turns_available": bool(turns),
        "timestamps_available": bool(transcript.segments),
        "language_segments_available": bool(
            language.language_segments or language.per_segment
        ),
        "monologue_available": bool(monologue) or (
            any(monologues.values()) if monologues else False
        ),
        "validation_report_available": True,
    }


def _default_quality_metrics() -> Dict:
    return {
        "estimated_word_accuracy": None,
        "estimated_timestamp_accuracy": None,
        "estimated_code_switch_accuracy": None,
        "benchmark_evaluated": False,
        "human_review_completed": False,
    }


def _default_human_review(required: bool = True) -> Dict:
    return {
        "required": bool(required),
        "status": "pending",
        "review_stage": "transcript_review",
        "priority": {
            "level": "normal",
            "score": 0.0,
            "reasons": [],
        },
        "reviewer_id": None,
        "notes": None,
    }


def _default_code_switch(language_meta: Dict) -> Dict:
    report = build_code_switch_report(language_meta.get("segments") or language_meta.get("language_segments") or [])
    dominant_languages: List[str] = []
    for segment in report.get("segments", []):
        lang = segment.get("lang")
        if lang and lang not in dominant_languages:
            dominant_languages.append(lang)
    return {
        "detected": bool(report.get("switch_count")),
        "dominant_languages": dominant_languages,
        "switch_count": int(report.get("switch_count") or 0),
        "switch_patterns": list(report.get("patterns") or []),
        "matrix_languages": {},
        "switch_points": [],
        "cs_density": 0.0,
        "review_required": bool(report.get("switch_count")),
    }


def _enrich_code_switch_from_segments(base: Dict, transcript: Transcript) -> Dict:
    out = dict(base or {})
    matrix_languages: Dict[str, int] = {}
    switch_points: List[Dict] = []
    total_words = 0
    total_switches = 0
    for idx, segment in enumerate(transcript.segments):
        matrix = getattr(segment, "matrix_language", None)
        if matrix:
            matrix_languages[matrix] = matrix_languages.get(matrix, 0) + 1
        points = list(getattr(segment, "switch_points", []) or [])
        for point in points:
            payload = dict(point)
            payload["segment_idx"] = idx
            switch_points.append(payload)
        total_switches += len(points)
        total_words += len(getattr(segment, "words", []) or []) or len(str(segment.text).split())
    density = total_switches / max(total_words, 1) * 100.0
    out["matrix_languages"] = matrix_languages
    out["switch_points"] = switch_points
    out["cs_density"] = round(float(density), 4)
    out["detected"] = bool(out.get("detected") or total_switches)
    out["switch_count"] = max(int(out.get("switch_count") or 0), total_switches)
    return out


def _default_premium_processing() -> Dict:
    return {
        "pipeline_mode": "offline_standard",
        "paid_api_used": False,
        "engines_used": ["whisper_local"],
        "consensus_applied": False,
        "timestamp_refinement_applied": False,
        "human_review_required": True,
    }


def _default_routing_decision() -> Dict:
    return {
        "pipeline_mode": "offline_standard",
        "paid_api_allowed": False,
        "local_first": True,
        "difficulty_score": 0.0,
        "should_escalate": False,
        "escalated_to_paid": False,
        "reasons": [],
        "attempted_engines": ["whisper_local"],
        "skipped_engines": [],
        "engines_attempted": ["whisper_local"],
        "engines_skipped": [],
        "engines_used": ["whisper_local"],
    }


def _default_timestamp_refinement() -> Dict:
    return {
        "timestamp_method": None,
        "timestamp_confidence": None,
        "word_timestamps_available": False,
        "segment_timestamps_available": False,
        "refinement_applied": False,
        "synthetic_word_timestamps": False,
        "timing_quality": "low",
        "notes": [],
    }


def _default_accuracy_gate() -> Dict:
    return {
        "target_word_accuracy": 0.99,
        "estimated_word_accuracy": 0.0,
        "verified_word_accuracy": None,
        "target_speaker_accuracy": 0.99,
        "estimated_speaker_accuracy": 0.0,
        "verified_speaker_accuracy": None,
        "target_timestamp_accuracy": 0.98,
        "estimated_timestamp_accuracy": 0.0,
        "verified_timestamp_accuracy": None,
        "target_code_switch_accuracy": 0.99,
        "estimated_code_switch_accuracy": 0.0,
        "verified_code_switch_accuracy": None,
        "estimated": True,
        "verified_accuracy": False,
        "passed": False,
        "human_review_required": True,
        "human_review_completed": False,
        "human_review_required_for_delivery": True,
        "reasons": ["accuracy_gate_not_evaluated"],
    }


def _default_delivery_status() -> Dict:
    return {
        "stage": "review_required",
        "approved_for_client_delivery": False,
        "reason": "accuracy_gate_not_evaluated",
    }


def build_record(
    *,
    audio_path: str,
    transcript: Transcript,
    turns: List[SpeakerTurn],
    language: LanguageReport,
    audio_meta: Dict,
    speaker_meta: Dict[str, Dict],
    conversation_meta: Dict,
    monologue: Optional[Monologue],
    # rich annotation blocks
    interaction_meta: Optional[Dict] = None,
    monologues: Optional[Dict[str, Optional[Monologue]]] = None,
    speaker_lang: Optional[Dict[str, LanguageReport]] = None,
    quality: Optional[Dict] = None,
    validation: Optional[Dict] = None,
    processing: Optional[Dict] = None,
    # dual-mode fields
    input_mode: str = "mono",
    speaker_sources: Optional[Dict[str, Dict]] = None,
    session_name: Optional[str] = None,
    speaker_map: Optional[Dict[str, str]] = None,
    source_files: Optional[List[str]] = None,
    generated_at: Optional[str] = None,
    input_alignment: Optional[Dict] = None,
    mono_mix: Optional[Dict] = None,
    quality_targets: Optional[Dict] = None,
    quality_metrics: Optional[Dict] = None,
    human_review: Optional[Dict] = None,
    code_switch: Optional[Dict] = None,
    transcript_candidates: Optional[List[Dict]] = None,
    routing_decision: Optional[Dict] = None,
    timestamp_method: Optional[str] = None,
    timestamp_confidence: Optional[float] = None,
    timestamp_refinement: Optional[Dict] = None,
    tts_suitability: Optional[Dict] = None,
    dataset_products: Optional[List[str]] = None,
    premium_processing: Optional[Dict] = None,
    cfg: Optional[object] = None,
    model_versions: Optional[Dict] = None,
    config_hash: Optional[str] = None,
    total_speech_duration_sec: Optional[float] = None,
    # AUDQ-02: original sample rate before pipeline resampled to 16kHz
    original_sample_rate: Optional[int] = None,
    # AUDQ-03: raw waveform for fingerprint computation (first 5s SHA-256)
    wav: Optional["np.ndarray"] = None,
    # perf
    skip_sha1: bool = False,
) -> Dict:
    """Build the full v3.0.0 dataset record."""

    timeline = _build_timeline(turns, speaker_map)
    conv_transcript = _build_conversation_transcript(transcript, turns, speaker_map)
    spk_transcripts = _build_speaker_transcripts(conv_transcript)
    annotations = _build_annotations(
        transcript,
        turns,
        language,
        monologue,
        monologues,
        validation,
    )

    # Inject per-speaker language into speaker_meta (non-destructive copy)
    enriched_speaker_meta: Dict[str, Dict] = {}
    for spk_id, meta in speaker_meta.items():
        entry = dict(meta)
        if speaker_lang and spk_id in speaker_lang:
            entry["language"] = speaker_lang[spk_id].to_dict()
        enriched_speaker_meta[spk_id] = entry

    language_meta = _language_aliases(language)
    cfg_dict = cfg.to_dict() if cfg is not None and hasattr(cfg, "to_dict") else {}
    pipeline_mode = str(cfg_dict.get("pipeline_mode") or (processing or {}).get("pipeline_mode") or "offline_standard")
    session_id = session_name or os.path.splitext(os.path.basename(audio_path))[0]
    sample_rate = int(audio_meta.get("processed_sample_rate_hz") or audio_meta.get("sample_rate_hz") or 16000)
    effective_model_versions = model_versions or _collect_model_versions(cfg_dict)
    effective_config_hash = config_hash or (_compute_config_hash(cfg_dict) if cfg_dict else "")
    dominant_language = _bcp47_normalize(language.dominant_language or language.primary_language)
    language_distribution: Dict[str, int] = {}
    for segment in language.language_segments or []:
        code = _bcp47_normalize(segment.language)
        language_distribution[code] = language_distribution.get(code, 0) + 1
    if not language_distribution and dominant_language:
        language_distribution[dominant_language] = len(transcript.segments)
    sorted_turns = sorted(turns, key=lambda t: t.start)

    def find_speaker(seg) -> str:
        forced_speaker = getattr(seg, "ground_truth_speaker", None)
        if forced_speaker:
            return forced_speaker
        mid = (seg.start + seg.end) / 2.0
        for turn in sorted_turns:
            if turn.start <= mid <= turn.end:
                return turn.speaker
        if sorted_turns:
            return min(sorted_turns, key=lambda t: min(abs(t.start - mid), abs(t.end - mid))).speaker
        return "SPEAKER_00"

    record: Dict = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "pipeline_record_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": generated_at or "1970-01-01T00:00:00+00:00",
        "input_mode": input_mode,
        "pipeline_mode": pipeline_mode,
        "session_id": session_id,
        "session_name": session_id,
        "model_versions": effective_model_versions,
        "config_hash": effective_config_hash,
        "num_speakers": int(len(speaker_meta)),
        "total_duration_sec": round(float(audio_meta.get("duration_s") or transcript.duration or 0.0), 3),
        "total_speech_duration_sec": round(float(
            total_speech_duration_sec
            if total_speech_duration_sec is not None
            else audio_meta.get("speech_duration_s") or 0.0
        ), 3),
        "dominant_language": dominant_language,
        "language_distribution": language_distribution,
        # AUDQ-02: preserve original sample rate even after 16kHz resampling.
        # Falls back to audio_meta["sample_rate_hz"] (source rate before pipeline
        # resampled), then to the processed sample_rate.
        "original_sample_rate": int(
            original_sample_rate
            or audio_meta.get("sample_rate_hz")
            or sample_rate
        ),
        # AUDQ-03: dedup fingerprint — SHA-256 of first 5s of waveform
        "audio_fingerprint": (
            _compute_audio_fingerprint(wav, sample_rate)
            if wav is not None and len(wav) > 0
            else None
        ),
        "file": _file_descriptor(audio_path, skip_sha1=skip_sha1),
        "source_files": [
            _file_descriptor(path, skip_sha1=skip_sha1)
            for path in (source_files or ([audio_path] if os.path.isfile(audio_path) else []))
        ],
        "metadata": {
            "audio": audio_meta,
            "language": language_meta,
            "speakers": enriched_speaker_meta,
            "conversation": conversation_meta,
            "interaction": _interaction_aliases(interaction_meta),
        },
        "audio_metadata": audio_meta,
        "transcript": {
            "raw": transcript.text,
            "normalised": normalise_transcript(transcript.text),
            "language": _bcp47_normalize(transcript.language),
            "language_probability": round(float(transcript.language_probability), 4),
            "duration_s": round(transcript.duration, 3),
            "segments": [
                {
                    **s.to_dict(
                        segment_id=f"{session_id}_{idx:04d}",
                        audio_filepath=str(audio_path),
                        speaker_id=find_speaker(s),
                        sample_rate=sample_rate,
                    ),
                    "language": _bcp47_normalize(s.language),
                }
                for idx, s in enumerate(transcript.segments)
            ],
        },
        "timeline": timeline,
        "speaker_transcripts": spk_transcripts,
        "conversation_transcript": conv_transcript,
        "speaker_segmentation": [t.to_dict() for t in turns],
        # per-speaker monologues + legacy single monologue_sample
        "monologues": {
            spk: (m.to_dict() if m else None)
            for spk, m in (monologues or {}).items()
        },
        "monologue_sample": monologue.to_dict() if monologue else None,
        "quality": quality or {},
        "annotations": annotations,
        "quality_targets": quality_targets or {
            "word_accuracy_target": 0.99,
            "speaker_accuracy_target": 0.99,
            "timestamp_accuracy_target": 0.98,
            "code_switch_accuracy_target": 0.99,
            "human_review_required": True,
        },
        "quality_metrics": quality_metrics or _default_quality_metrics(),
        "human_review": human_review or _default_human_review(),
        "code_switch": _enrich_code_switch_from_segments(
            code_switch or _default_code_switch(language_meta),
            transcript,
        ),
        "validation": validation or {"passed": True, "issue_count": 0, "issues": [], "checks": {}},
        "processing": processing or {},
        "artifacts": {},
        "input_alignment": input_alignment or {},
        "mono_mix": mono_mix or {},
        "transcript_candidates": list(transcript_candidates or []),
        "consensus": {},
        "accuracy_gate": _default_accuracy_gate(),
        "routing_decision": routing_decision or _default_routing_decision(),
        "timestamp_method": timestamp_method,
        "timestamp_confidence": round(float(timestamp_confidence), 4) if timestamp_confidence is not None else None,
        "timestamp_refinement": timestamp_refinement or _default_timestamp_refinement(),
        "tts_suitability": tts_suitability or {"eligible": False, "reasons": ["review_not_final"], "confidence": 0.2},
        "dataset_products": list(dataset_products or []),
        "premium_processing": premium_processing or _default_premium_processing(),
        "delivery_status": _default_delivery_status(),
        "review_artifacts": {},
    }
    record["dataset_purpose"] = infer_dataset_purpose(record)

    # speaker_pair mode: add source-file provenance
    if input_mode in ("speaker_pair", "stereo") and speaker_sources:
        record["speaker_sources"] = speaker_sources

    return record
