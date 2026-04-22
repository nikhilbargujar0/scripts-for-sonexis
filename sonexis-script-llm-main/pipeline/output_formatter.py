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
import os
from typing import Dict, List, Optional

from .diarisation import SpeakerTurn
from .language_detection import LanguageReport
from .monologue_extractor import Monologue
from .transcription import Transcript, normalise_transcript


SCHEMA_VERSION = "3.0.0"


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
        mid = (seg.start + seg.end) / 2.0
        spk = find_speaker(mid)
        result.append({
            "speaker": spk,
            "label": sm.get(spk, spk),
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "language": seg.language,
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
    # perf
    skip_sha1: bool = False,
) -> Dict:
    """Build the full v3.0.0 dataset record."""

    timeline = _build_timeline(turns, speaker_map)
    conv_transcript = _build_conversation_transcript(transcript, turns, speaker_map)
    spk_transcripts = _build_speaker_transcripts(conv_transcript)

    # Inject per-speaker language into speaker_meta (non-destructive copy)
    enriched_speaker_meta: Dict[str, Dict] = {}
    for spk_id, meta in speaker_meta.items():
        entry = dict(meta)
        if speaker_lang and spk_id in speaker_lang:
            entry["language"] = speaker_lang[spk_id].to_dict()
        enriched_speaker_meta[spk_id] = entry

    record: Dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or "1970-01-01T00:00:00+00:00",
        "input_mode": input_mode,
        "session_name": session_name or os.path.splitext(
            os.path.basename(audio_path))[0],
        "file": _file_descriptor(audio_path, skip_sha1=skip_sha1),
        "source_files": [
            _file_descriptor(path, skip_sha1=skip_sha1)
            for path in (source_files or ([audio_path] if os.path.isfile(audio_path) else []))
        ],
        "metadata": {
            "audio": audio_meta,
            "language": language.to_dict(),
            "speakers": enriched_speaker_meta,
            "conversation": conversation_meta,
            "interaction": interaction_meta or {},
        },
        "transcript": {
            "raw": transcript.text,
            "normalised": normalise_transcript(transcript.text),
            "language": transcript.language,
            "language_probability": round(float(transcript.language_probability), 4),
            "duration_s": round(transcript.duration, 3),
            "segments": [s.to_dict() for s in transcript.segments],
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
        "validation": validation or {"passed": True, "issue_count": 0, "issues": [], "checks": {}},
        "processing": processing or {},
        "artifacts": {},
    }

    # speaker_pair mode: add source-file provenance
    if input_mode in ("speaker_pair", "stereo") and speaker_sources:
        record["speaker_sources"] = speaker_sources

    return record
