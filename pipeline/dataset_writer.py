"""dataset_writer.py

Handles all file I/O for the production dataset system.

Output folder structure
-----------------------
<output_root>/
  audio/
    raw/              ← copies / symlinks of original source files
    speaker_separated/<session>/  ← per-speaker PCM-16 WAVs
    mono/<session>/   ← mixed mono WAV
    monologues/<session>/  ← best monologue clip per speaker
  transcripts/<session>/
    raw.txt
    normalised.txt
    combined_conversation.json   ← [{speaker, label, start, end, text}]
    speaker_<SPEAKER_XX>.json
    annotations/
    <session>.json    ← full record (output_formatter schema v3)
  manifests/
    utterances.jsonl  ← one line per turn
    conversations.jsonl ← one line per session
    speakers.jsonl    ← one line per speaker×session
  logs/
    pipeline.log
  schema.json         ← written once on first run
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

from .diarisation import SpeakerTurn
from .exporters import export_phase4_formats
from .monologue_extractor import Monologue
from .products import export_products
from .quality_tier import classify_record
from .review_queue import write_review_queue
from .schema_validator import validate_record


# ── schema declaration ──────────────────────────────────────────────────────

_SCHEMA_VERSION = "3.0.0"

_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "version": _SCHEMA_VERSION,
    "description": "Sonexis conversational audio dataset record",
    "fields": {
        "schema_version": "string",
        "generated_at": "ISO-8601 UTC timestamp",
        "input_mode": "mono | speaker_pair | stereo",
        "session_name": "string",
        "speaker_sources": "object (speaker_pair mode only)",
        "file": {
            "path": "absolute path",
            "name": "filename",
            "size_bytes": "integer | null",
            "sha1": "hex string | null",
        },
        "metadata": {
            "audio": "AudioMetadata",
            "language": "LanguageReport",
            "speakers": "Dict[speaker_id -> SpeakerMetadata]",
            "conversation": "ConversationMetadata",
            "interaction": "InteractionMetadata",
        },
        "transcript": {
            "raw": "string",
            "normalised": "string",
            "language": "ISO-639-1 code",
            "language_probability": "float [0,1]",
            "duration_s": "float",
            "segments": "list of TranscriptSegment",
        },
        "timeline": "list of {speaker, label, start, end}",
        "speaker_transcripts": "Dict[speaker_id -> string]",
        "conversation_transcript": "list of {speaker, label, start, end, text}",
        "speaker_segmentation": "list of SpeakerTurn dicts",
        "monologues": "Dict[speaker_id -> MonologueSample | null]",
        "monologue_sample": "MonologueSample | null (legacy compat)",
        "quality": "QualityReport",
        "validation": "ValidationReport",
        "processing": "model/config/fallback provenance",
        "artifacts": "paths written by DatasetWriter",
    },
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _write_json(path: Path, data: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def _append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    """Write PCM-16 WAV via soundfile; fallback to scipy if unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_SF:
        # Ensure float32 → int16 range for PCM_16
        if wav.dtype in (np.float32, np.float64):
            data = np.clip(wav, -1.0, 1.0)
            data = (data * 32767).astype(np.int16)
        else:
            data = wav.astype(np.int16)
        sf.write(str(path), data, sample_rate, subtype="PCM_16")
    else:
        try:
            from scipy.io import wavfile
            if wav.dtype in (np.float32, np.float64):
                data = np.clip(wav, -1.0, 1.0)
                data = (data * 32767).astype(np.int16)
            else:
                data = wav.astype(np.int16)
            wavfile.write(str(path), sample_rate, data)
        except ImportError:
            logging.getLogger(__name__).warning(
                "soundfile and scipy unavailable — WAV not written to %s", path
            )


# ── DatasetWriter ────────────────────────────────────────────────────────────

class DatasetWriter:
    """Manages all file output for one pipeline run.

    Usage
    -----
    writer = DatasetWriter(output_root="/data/out", output_mode="both")
    writer.write_session(session_name, record, pair_or_audio, turns, monologues)
    """

    def __init__(
        self,
        output_root: str,
        output_mode: str = "both",      # both | speaker_separated | mono
        output_format: str = "json",    # json | jsonl | parquet
        dataset_name: str = "dataset",
    ) -> None:
        self.root = Path(output_root)
        self.output_mode = output_mode
        self.output_format = output_format
        self.dataset_name = dataset_name

        # sub-directories
        self.audio_root = self.root / "audio"
        self.transcripts_root = self.root / "transcripts"
        self.annotations_root = self.root / "annotations"
        self.manifests_root = self.root / "manifests"
        self.logs_root = self.root / "logs"

        self._setup_logging()
        self._write_schema_once()

    # ── setup ──────────────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        self.logs_root.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_root / "pipeline.log"
        handler = logging.FileHandler(str(log_path), encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root_logger = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) and
                   getattr(h, "baseFilename", None) == str(log_path)
                   for h in root_logger.handlers):
            root_logger.addHandler(handler)
        self._log = logging.getLogger(__name__)

    def _write_schema_once(self) -> None:
        schema_path = self.root / "schema.json"
        if not schema_path.exists():
            _write_json(schema_path, _SCHEMA)

    # ── audio writing ──────────────────────────────────────────────────────

    def write_speaker_wavs(
        self,
        session_name: str,
        speaker_wavs: Dict[str, np.ndarray],
        sample_rate: int,
        speaker_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Write per-speaker WAVs. Returns {speaker_id: abs_path}."""
        paths: Dict[str, str] = {}
        if self.output_mode not in ("both", "speaker_separated"):
            return paths
        out_dir = self.audio_root / "model_ready_16k" / session_name
        for spk_id, wav in speaker_wavs.items():
            label = (speaker_map or {}).get(spk_id, spk_id)
            suffix = "spk1" if label == "speaker_1" else ("spk2" if label == "speaker_2" else label.lower().replace(" ", "_"))
            fname = f"{session_name}_{suffix}_16k.wav"
            p = out_dir / fname
            _write_wav(p, wav, sample_rate)
            paths[spk_id] = str(p.resolve())
            self._log.info("Wrote speaker WAV: %s", p)
        return paths

    def write_mono_wav(
        self,
        session_name: str,
        wav: np.ndarray,
        sample_rate: int,
        suffix: str = "mixed",
    ) -> Optional[str]:
        """Write mixed mono WAV. Returns abs path or None if skipped."""
        if self.output_mode not in ("both", "mono"):
            return None
        out_dir = self.audio_root / "model_ready_16k" / session_name
        p = out_dir / f"{session_name}_mono_16k.wav"
        _write_wav(p, wav, sample_rate)
        self._log.info("Wrote mono WAV: %s", p)
        return str(p.resolve())

    def write_monologue_wavs(
        self,
        session_name: str,
        monologues: Dict[str, Optional[Monologue]],
        speaker_wavs: Dict[str, np.ndarray],
        sample_rate: int,
        speaker_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Optional[str]]:
        """Write monologue audio clips. Returns {speaker_id: path | None}."""
        paths: Dict[str, Optional[str]] = {}
        out_dir = self.audio_root / "monologues" / session_name
        for spk_id, mono in monologues.items():
            if mono is None:
                paths[spk_id] = None
                continue
            src_wav = speaker_wavs.get(spk_id)
            if src_wav is None:
                paths[spk_id] = None
                continue
            start_sample = int(mono.start * sample_rate)
            end_sample = int(mono.end * sample_rate)
            clip = src_wav[start_sample:end_sample]
            if clip.size == 0:
                paths[spk_id] = None
                continue
            label = (speaker_map or {}).get(spk_id, spk_id)
            suffix = "spk1" if label == "speaker_1" else ("spk2" if label == "speaker_2" else label.lower().replace(" ", "_"))
            fname = f"{session_name}_{suffix}_monologue.wav"
            p = out_dir / fname
            _write_wav(p, clip, sample_rate)
            paths[spk_id] = str(p.resolve())
            self._log.info("Wrote monologue clip: %s", p)
        return paths

    def copy_raw(self, source_paths: List[str], session_name: str) -> None:
        """Copy original source files into audio/raw/<session>/."""
        out_dir = self.audio_root / "raw" / session_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in source_paths:
            if os.path.isfile(src):
                dst = out_dir / Path(src).name
                if not dst.exists():
                    shutil.copy2(src, dst)

    def write_original_48k(
        self,
        session_name: str,
        original_sources: Dict[str, Any],
        speaker_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Preserve source speaker audio under the canonical original_48k layout."""
        written: Dict[str, str] = {}
        out_dir = self.audio_root / "original_48k" / session_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for spk_id, audio in original_sources.items():
            label = (speaker_map or {}).get(spk_id, spk_id)
            suffix = "spk1" if label == "speaker_1" else ("spk2" if label == "speaker_2" else label.lower().replace(" ", "_"))
            dst = out_dir / f"{session_name}_{suffix}_48k.wav"
            src = getattr(audio, "path", "")
            if os.path.isfile(src) and Path(src).suffix.lower() == ".wav":
                if not dst.exists():
                    shutil.copy2(src, dst)
            else:
                if os.path.isfile(src) and _HAS_SF:
                    data, sr = sf.read(src, dtype="float32", always_2d=False)
                    if getattr(data, "ndim", 1) == 2:
                        data = data.mean(axis=1)
                    _write_wav(dst, data, int(sr))
                else:
                    _write_wav(dst, getattr(audio, "waveform"), int(getattr(audio, "sample_rate", 48_000)))
            written[spk_id] = str(dst.resolve())
        return written

    # ── transcript writing ─────────────────────────────────────────────────

    def write_transcripts(
        self,
        session_name: str,
        raw_text: str,
        normalised_text: str,
        conversation_transcript: List[Dict],
        speaker_transcripts: Optional[Dict[str, str]] = None,
        speaker_map: Optional[Dict[str, str]] = None,
    ) -> None:
        out_dir = self.transcripts_root / session_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # plain text dumps
        (out_dir / "raw.txt").write_text(raw_text, encoding="utf-8")
        (out_dir / "normalised.txt").write_text(normalised_text, encoding="utf-8")

        # combined conversation JSON
        _write_json(out_dir / "combined_conversation.json", conversation_transcript)
        _write_json(self.transcripts_root / f"{session_name}.json", conversation_transcript)

        # per-speaker
        if speaker_transcripts:
            for spk_id, text in speaker_transcripts.items():
                label = (speaker_map or {}).get(spk_id, spk_id)
                fname = f"speaker_{label.lower().replace(' ', '_')}.json"
                _write_json(out_dir / fname, {"speaker": spk_id, "label": label, "text": text})

    # ── annotation writing ─────────────────────────────────────────────────

    def write_annotation(self, session_name: str, record: Dict) -> str:
        """Write full session record. Returns abs path."""
        validate_record(record)
        p = self.annotations_root / f"{session_name}.json"
        _write_json(p, record)
        self._log.info("Wrote annotation: %s", p)
        return str(p.resolve())

    # ── manifest writing ───────────────────────────────────────────────────

    def append_utterances_manifest(
        self,
        session_name: str,
        turns: List[SpeakerTurn],
        speaker_wav_paths: Dict[str, str],
        sample_rate: int,
        speaker_map: Optional[Dict[str, str]] = None,
        conversation_transcript: Optional[List[Dict]] = None,
        fallback_audio_path: str = "",
    ) -> None:
        """Append one line per turn to utterances.jsonl."""
        manifest = self.manifests_root / "utterances.jsonl"
        speaker_manifest = self.manifests_root / "speaker_separated.jsonl"

        for turn in turns:
            label = (speaker_map or {}).get(turn.speaker, turn.speaker)
            text_parts: List[str] = []
            languages: List[str] = []
            quality_scores: List[float] = []
            best_overlap = 0.0
            for seg in conversation_transcript or []:
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", 0.0))
                overlap = min(turn.end, e) - max(turn.start, s)
                if overlap > 0:
                    if seg.get("text"):
                        text_parts.append(str(seg.get("text")).strip())
                    if seg.get("language"):
                        languages.append(str(seg.get("language")))
                    if seg.get("quality_score") is not None:
                        quality_scores.append(float(seg.get("quality_score")))
                if overlap > best_overlap:
                    best_overlap = overlap

            start_ms = int(round(turn.start * 1000))
            end_ms = int(round(turn.end * 1000))
            utterance_id = f"{session_name}_{turn.speaker}_{start_ms:08d}_{end_ms:08d}"

            row = {
                "utterance_id": utterance_id,
                "session": session_name,
                "speaker": turn.speaker,
                "label": label,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "duration_s": round(turn.duration(), 3),
                "confidence": round(turn.confidence, 4),
                "audio_path": speaker_wav_paths.get(turn.speaker, fallback_audio_path),
                "sample_rate": int(sample_rate),
                "text": " ".join(t for t in text_parts if t).strip(),
                "language": languages[0] if languages else None,
                "quality_score": (
                    round(sum(quality_scores) / len(quality_scores), 4)
                    if quality_scores else None
                ),
            }
            _append_jsonl(manifest, row)
            _append_jsonl(speaker_manifest, row)

    def append_conversations_manifest(self, record: Dict) -> None:
        """Append one line per session to conversations.jsonl."""
        manifest = self.manifests_root / "conversations.jsonl"
        mono_manifest = self.manifests_root / "mono_conversation.jsonl"
        conv_meta = record.get("metadata", {}).get("conversation", {})
        interaction = record.get("metadata", {}).get("interaction", {})
        lang = record.get("metadata", {}).get("language", {})
        validation = record.get("validation", {})
        quality = conv_meta.get("quality", {}) or {}
        artifacts = record.get("artifacts", {})
        row = {
            "session": record.get("session_name", ""),
            "input_mode": record.get("input_mode", ""),
            "generated_at": record.get("generated_at", ""),
            "duration_s": conv_meta.get("total_speech_time_s"),
            "conversation_duration_s": interaction.get("conversation_duration_s"),
            "turn_count": conv_meta.get("turn_count"),
            "speaker_count": conv_meta.get("speaker_count"),
            "primary_language": lang.get("primary_language"),
            "dominant_language": lang.get("dominant_language"),
            "multilingual": lang.get("multilingual_flag", False),
            "switching_score": lang.get("switching_score"),
            "engagement_score": interaction.get("engagement_score"),
            "dominant_speaker": interaction.get("dominant_speaker"),
            "alignment_applied": (record.get("input_alignment") or {}).get("applied"),
            "alignment_confidence": (record.get("input_alignment") or {}).get("confidence"),
            "mono_mix_peak_after": (record.get("mono_mix") or {}).get("peak_after"),
            "mean_quality_score": quality.get("mean_quality_score"),
            "validation_passed": validation.get("passed"),
            "validation_issue_count": validation.get("issue_count"),
            "mono_audio_path": artifacts.get("mono_wav", ""),
            "annotation_path": str(
                (self.annotations_root / f"{record.get('session_name','')}.json").resolve()
            ),
        }
        _append_jsonl(manifest, row)
        _append_jsonl(mono_manifest, row)

    def append_speakers_manifest(
        self,
        session_name: str,
        speaker_meta: Dict[str, Dict],
        speaker_map: Optional[Dict[str, str]] = None,
        speaker_lang: Optional[Dict[str, Dict]] = None,
        monologue_paths: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        """Append one line per speaker×session to speakers.jsonl."""
        manifest = self.manifests_root / "speakers.jsonl"
        monologues_manifest = self.manifests_root / "monologues.jsonl"
        for spk_id, meta in speaker_meta.items():
            label = (speaker_map or {}).get(spk_id, spk_id)
            lang_info = (speaker_lang or {}).get(spk_id, {})
            row = {
                "session": session_name,
                "speaker": spk_id,
                "label": label,
                "total_speaking_time_s": meta.get("total_speaking_time_s"),
                "word_count": meta.get("word_count"),
                "wpm": meta.get("wpm"),
                "filler_ratio": meta.get("filler_ratio"),
                "pause_frequency_per_min": meta.get("pause_frequency_per_min"),
                "speaking_style": (meta.get("speaking_style") or {}).get("value"),
                "speaking_style_confidence": (meta.get("speaking_style") or {}).get("confidence"),
                "formality": (meta.get("formality") or {}).get("value"),
                "formality_confidence": (meta.get("formality") or {}).get("confidence"),
                "accent": (meta.get("accent") or {}).get("value"),
                "accent_confidence": (meta.get("accent") or {}).get("confidence"),
                "provided_metadata": meta.get("provided_metadata", {}),
                "turn_count": meta.get("turn_count"),
                "primary_language": lang_info.get("primary_language"),
                "monologue_path": (monologue_paths or {}).get(spk_id),
            }
            _append_jsonl(manifest, row)
            if row.get("monologue_path"):
                _append_jsonl(monologues_manifest, row)

    # ── high-level session write ───────────────────────────────────────────

    def write_session(
        self,
        *,
        session_name: str,
        record: Dict,
        turns: List[SpeakerTurn],
        speaker_wavs: Optional[Dict[str, np.ndarray]] = None,
        mixed_wav: Optional[np.ndarray] = None,
        sample_rate: int = 16_000,
        speaker_map: Optional[Dict[str, str]] = None,
        monologues: Optional[Dict[str, Optional[Monologue]]] = None,
        source_paths: Optional[List[str]] = None,
        original_sources: Optional[Dict[str, Any]] = None,
        speaker_lang: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, str]:
        """Write all outputs for one session.

        Returns dict of written paths keyed by artifact type.
        """
        written: Dict[str, str] = {}

        # 1. Copy raws
        if source_paths:
            self.copy_raw(source_paths, session_name)

        if original_sources:
            original_paths = self.write_original_48k(session_name, original_sources, speaker_map)
            written.update({f"original_48k_{k}": v for k, v in original_paths.items()})

        # 2. Speaker WAVs
        speaker_wav_paths: Dict[str, str] = {}
        if speaker_wavs:
            speaker_wav_paths = self.write_speaker_wavs(
                session_name, speaker_wavs, sample_rate, speaker_map
            )
            written.update({f"speaker_wav_{k}": v for k, v in speaker_wav_paths.items()})

        # 3. Mono WAV
        mono_path = ""
        if mixed_wav is not None:
            mono_path = self.write_mono_wav(session_name, mixed_wav, sample_rate)
            if mono_path:
                written["mono_wav"] = mono_path

        # 4. Monologue clips
        monologue_paths: Dict[str, Optional[str]] = {}
        if monologues and speaker_wavs:
            monologue_paths = self.write_monologue_wavs(
                session_name, monologues, speaker_wavs, sample_rate, speaker_map
            )
            for k, v in monologue_paths.items():
                if v:
                    written[f"monologue_{k}"] = v

        # 5. Transcripts
        transcript = record.get("transcript", {})
        conv_trans = record.get("conversation_transcript", [])
        spk_trans = record.get("speaker_transcripts", {})
        self.write_transcripts(
            session_name,
            raw_text=transcript.get("raw", ""),
            normalised_text=transcript.get("normalised", ""),
            conversation_transcript=conv_trans,
            speaker_transcripts=spk_trans,
            speaker_map=speaker_map,
        )

        # 6. Annotation JSON
        predicted_ann_path = str((self.annotations_root / f"{session_name}.json").resolve())
        written["annotation"] = predicted_ann_path
        classify_record(record)
        review_path = write_review_queue(record, str(self.root))
        if review_path:
            written["review_queue"] = review_path
        record["artifacts"] = dict(written)
        ann_path = self.write_annotation(session_name, record)
        written["annotation"] = ann_path

        # 6b. Product-specific exports derived from the canonical record.
        product_artifacts = export_products(record, str(self.root))
        if product_artifacts:
            written.update(product_artifacts)
            record["artifacts"] = dict(written)
            ann_path = self.write_annotation(session_name, record)
            written["annotation"] = ann_path

        # 6c. Phase 4 multi-format exports from canonical record.
        phase4_artifacts = export_phase4_formats(record, str(self.root))
        if phase4_artifacts:
            written.update(phase4_artifacts)
            record["artifacts"] = dict(written)
            ann_path = self.write_annotation(session_name, record)
            written["annotation"] = ann_path

        # 7. Manifests
        self.append_utterances_manifest(
            session_name,
            turns,
            speaker_wav_paths,
            sample_rate,
            speaker_map=speaker_map,
            conversation_transcript=conv_trans,
            fallback_audio_path=mono_path or "",
        )
        self.append_conversations_manifest(record)
        speaker_meta = record.get("metadata", {}).get("speakers", {})
        self.append_speakers_manifest(
            session_name,
            speaker_meta,
            speaker_map=speaker_map,
            speaker_lang=speaker_lang,
            monologue_paths=monologue_paths,
        )

        self._log.info(
            "Session %s written — %d artifacts", session_name, len(written)
        )
        return written
