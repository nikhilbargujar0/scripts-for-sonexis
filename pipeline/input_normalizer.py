"""Normalize messy Drive-style speaker folders into canonical Sonexis input."""
from __future__ import annotations

import json
import re
import shutil
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NORMALIZER_VERSION = "1.0.0"

DEFAULT_LANGUAGE_MAP: Dict[str, str] = {
    "english": "en",
    "hindi": "hi",
    "hinglish": "hinglish",
    "marwadi": "mwr",
    "punjabi": "pa",
}

JUNK_FOLDERS = {
    "output",
    "outputs",
    "qa",
    "review",
    "reviews",
    "transcript",
    "transcripts",
    "manifests",
    "annotations",
    "old",
    "archive",
    "__macosx",
    ".ipynb_checkpoints",
}

IGNORED_FILE_SUFFIXES = {".zip"}
SUPPORTED_AUDIO_SUFFIXES = {".wav"}


def _normalise_name(name: str) -> str:
    stem = Path(str(name)).stem
    return re.sub(r"[\s_-]+", "", stem.lower())


def _is_speaker1_name(name: str) -> bool:
    return _normalise_name(name) in {"speaker1", "speaker01", "spk1", "spk01", "s1"}


def _is_speaker2_name(name: str) -> bool:
    return _normalise_name(name) in {"speaker2", "speaker02", "spk2", "spk02", "s2"}


def _is_junk_dir(path: Path) -> bool:
    return path.name.startswith(".") or path.name.lower() in JUNK_FOLDERS


def _conversation_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name.lower()
    return 10**9, path.name.lower()


def _audio_duration(path: Path) -> float:
    if path.suffix.lower() != ".wav":
        return 0.0
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            return float(frames / rate) if rate else 0.0
    except (wave.Error, OSError, EOFError):
        return 0.0


def _candidate_score(path: Path, speaker: int, conversation_dir: Path) -> Tuple[int, float, int, str]:
    if speaker == 1:
        filename_match = _is_speaker1_name(path.name)
        parent_match = any(_is_speaker1_name(parent.name) for parent in path.parents if conversation_dir in parent.parents or parent == conversation_dir)
    else:
        filename_match = _is_speaker2_name(path.name)
        parent_match = any(_is_speaker2_name(parent.name) for parent in path.parents if conversation_dir in parent.parents or parent == conversation_dir)
    rank = 2 if filename_match else 1 if parent_match else 0
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return rank, _audio_duration(path), size, str(path)


def _iter_audio_files(conversation_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in conversation_dir.rglob("*"):
        if any(_is_junk_dir(parent) for parent in path.relative_to(conversation_dir).parents):
            continue
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix in IGNORED_FILE_SUFFIXES:
            continue
        if suffix in SUPPORTED_AUDIO_SUFFIXES:
            files.append(path)
    return sorted(files)


def _select_speaker_files(conversation_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    audio_files = _iter_audio_files(conversation_dir)
    unsupported_audio = [
        path for path in conversation_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp3", ".flac"}
    ]
    if unsupported_audio and not audio_files:
        return None, None, "unsupported_audio_format"

    speaker1 = [
        path for path in audio_files
        if _is_speaker1_name(path.name) or any(_is_speaker1_name(parent.name) for parent in path.parents if parent != conversation_dir)
    ]
    speaker2 = [
        path for path in audio_files
        if _is_speaker2_name(path.name) or any(_is_speaker2_name(parent.name) for parent in path.parents if parent != conversation_dir)
    ]
    selected1 = max(speaker1, key=lambda p: _candidate_score(p, 1, conversation_dir)) if speaker1 else None
    selected2 = max(speaker2, key=lambda p: _candidate_score(p, 2, conversation_dir)) if speaker2 else None
    if selected1 and selected2 and selected1.resolve() == selected2.resolve():
        return None, None, "ambiguous_speaker_assignment"
    if not selected1 and not selected2:
        return None, None, "missing_speaker_pair"
    if not selected1:
        return None, selected2, "missing_speaker_1"
    if not selected2:
        return selected1, None, "missing_speaker_2"
    return selected1, selected2, None


def _read_source_metadata(conversation_dir: Path) -> Optional[Dict[str, Any]]:
    metadata_path = conversation_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _language_sources(input_root: Path, language_map: Dict[str, str], strict: bool, errors: List[str]) -> List[Tuple[str, str, Path]]:
    canonical_by_name = {name.lower(): (name.lower(), code) for name, code in language_map.items()}
    root_key = input_root.name.lower()
    if root_key in canonical_by_name:
        canonical, code = canonical_by_name[root_key]
        return [(canonical, code, input_root)]

    sources: List[Tuple[str, str, Path]] = []
    for child in sorted(input_root.iterdir() if input_root.is_dir() else []):
        if not child.is_dir() or _is_junk_dir(child):
            continue
        key = child.name.lower()
        if key in canonical_by_name:
            canonical, code = canonical_by_name[key]
            sources.append((canonical, code, child))
        elif strict:
            errors.append(f"unknown_language_folder:{child}")
    return sources


def _conversation_candidates(language_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for child in sorted(language_dir.iterdir(), key=_conversation_sort_key):
        if not child.is_dir() or _is_junk_dir(child):
            continue
        if any(path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES or path.suffix.lower() in {".mp3", ".flac"} for path in child.rglob("*") if path.is_file()):
            candidates.append(child)
    return candidates


def _metadata(
    conversation_id: str,
    language_code: str,
    canonical_language: str,
    source_language_folder: Path,
    source_conversation_folder: Path,
    speaker1: Path,
    speaker2: Path,
    source_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "conversation_id": conversation_id,
        "language": language_code,
        "language_folder": canonical_language,
        "source_language_folder": str(source_language_folder.resolve()),
        "source_conversation_folder": str(source_conversation_folder.resolve()),
        "source_files": {
            "speaker_1": str(speaker1.resolve()),
            "speaker_2": str(speaker2.resolve()),
        },
        "speaker_count": 2,
        "speaker_layout": "speaker_separated",
        "input_normalized": True,
        "normalizer_version": NORMALIZER_VERSION,
        "scenario_id": conversation_id,
        "scenario_name": "unspecified",
        "topic": "unspecified",
        "sub_topic": "unspecified",
        "conversation_style": "natural_conversation",
        "language_mix": language_code,
        "scripted": False,
    }
    if source_metadata is not None:
        payload["source_metadata"] = source_metadata
    return payload


def normalize_messy_input(
    input_root: Path,
    work_root: Path,
    language_map: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    input_root = Path(input_root)
    work_root = Path(work_root)
    lang_map = {str(name).lower(): code for name, code in (language_map or DEFAULT_LANGUAGE_MAP).items()}
    errors: List[str] = []
    report: Dict[str, Any] = {
        "input_root": str(input_root.resolve()),
        "work_root": str(work_root.resolve()),
        "languages": {},
        "total_valid_conversations": 0,
        "total_skipped": 0,
        "errors": errors,
    }

    if not input_root.exists():
        errors.append(f"input_root_not_found:{input_root}")
        work_root.mkdir(parents=True, exist_ok=True)
        _write_json(work_root / "normalization_report.json", report)
        return report

    for canonical_language, language_code, source_language_dir in _language_sources(input_root, lang_map, strict, errors):
        normalised_language_dir = work_root / canonical_language
        language_report: Dict[str, Any] = {
            "language_code": language_code,
            "source_folder": str(source_language_dir.resolve()),
            "normalised_folder": str(normalised_language_dir.resolve()),
            "valid_conversations": 0,
            "skipped": [],
        }
        report["languages"][canonical_language] = language_report
        conversation_index = 1
        for candidate in _conversation_candidates(source_language_dir):
            speaker1, speaker2, reason = _select_speaker_files(candidate)
            if reason or speaker1 is None or speaker2 is None:
                language_report["skipped"].append({"source": str(candidate.resolve()), "reason": reason or "missing_speaker_pair"})
                continue

            conversation_id = f"conversation_{conversation_index:04d}"
            target_dir = normalised_language_dir / conversation_id
            target_spk1 = target_dir / "speaker_1" / "speaker_1.wav"
            target_spk2 = target_dir / "speaker_2" / "speaker_2.wav"
            target_spk1.parent.mkdir(parents=True, exist_ok=True)
            target_spk2.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(speaker1, target_spk1)
            shutil.copy2(speaker2, target_spk2)
            source_metadata = _read_source_metadata(candidate)
            _write_json(
                target_dir / "metadata.json",
                _metadata(
                    conversation_id,
                    language_code,
                    canonical_language,
                    source_language_dir,
                    candidate,
                    speaker1,
                    speaker2,
                    source_metadata,
                ),
            )
            conversation_index += 1
            language_report["valid_conversations"] += 1

        language_report["skipped"].sort(key=lambda row: row["source"])
        report["total_valid_conversations"] += int(language_report["valid_conversations"])
        report["total_skipped"] += len(language_report["skipped"])

    if strict and report["total_valid_conversations"] == 0 and not errors:
        errors.append("no_valid_language_folders_found")
    work_root.mkdir(parents=True, exist_ok=True)
    _write_json(work_root / "normalization_report.json", report)
    return report
