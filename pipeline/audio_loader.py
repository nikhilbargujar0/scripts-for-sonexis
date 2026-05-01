"""audio_loader.py

Two input modes:

  MONO  — single mixed audio file (existing behaviour).
          One file → one pipeline run → one output record.

  SPEAKER_PAIR — two separate per-speaker audio files that belong to
          the same conversation (e.g. Host.wav + Guest.wav from a
          telephony recording system).
          Two files → mixed on-the-fly for ASR/VAD/metadata,
          but speaker identity is ground-truth (no clustering needed).

Auto-detection (--input-mode auto, the default):
  Scans the input tree and decides per session:
  - exactly 2 audio files in same dir  → speaker_pair
  - subfolders each with 1 audio file  → speaker_pair (e.g. Host/ + Guest/)
  - anything else                       → mono (existing path)

Supported folder patterns for speaker_pair:
  Pattern A — flat siblings:
    session_dir/
      speaker1.wav
      speaker2.wav

  Pattern B — labelled subfolders (your zip layout):
    session_dir/
      Host/Host.wav
      Guest/Guest.wav

  Pattern C — multiple sessions:
    root/
      001/
        Host/Host.wav
        Guest/Guest.wav
      002/
        Host/Host.wav
        Guest/Guest.wav
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise ImportError("librosa is required. pip install librosa") from exc

log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
STUDIO_EXTS = {".wav", ".flac"}
TARGET_SR = 16_000  # faster-whisper, webrtcvad and pyannote all expect 16 kHz
STEREO_STEMS = {"stereo", "conversation_stereo", "dual_channel", "two_channel"}


# ── mono audio ────────────────────────────────────────────────────────────

@dataclass
class LoadedAudio:
    """Decoded audio clip: mono float32, normalised to [-1, 1]."""
    path: str
    waveform: np.ndarray
    sample_rate: int
    duration: float
    source_sample_rate: Optional[int] = None
    channels: int = 1
    sample_width_bits: Optional[int] = None
    encoding: Optional[str] = None

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)


# ── speaker-pair audio ────────────────────────────────────────────────────

@dataclass
class SpeakerPairAudio:
    """Two per-speaker recordings from the same conversation session.

    speakers  — ordered dict {label: LoadedAudio}, e.g. {"Host": ..., "Guest": ...}
    mixed     — mono sum of both speakers, used for ASR + audio metadata
    session_name — derived from the containing directory name
    recording_type — "separate" | "stereo" (how the source files were stored)
    """
    session_name: str
    speakers: Dict[str, LoadedAudio]        # label → audio
    mixed: LoadedAudio                      # for ASR + audio metadata
    speaker_map: Dict[str, str] = field(default_factory=dict)
    # speaker_map: {pipeline_label: human_label} e.g. {"SPEAKER_00": "Host"}
    recording_type: str = "separate"        # separate | stereo
    conversation_metadata: Dict = field(default_factory=dict)
    validation_context: Dict = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return self.session_name

    @property
    def duration(self) -> float:
        return self.mixed.duration

    @property
    def path(self) -> str:
        return self.mixed.path


# ── loaders ───────────────────────────────────────────────────────────────

def _audio_probe(path: str) -> Dict[str, Optional[int | str]]:
    """Best-effort source metadata without decoding the whole file."""
    info: Dict[str, Optional[int | str]] = {
        "source_sample_rate": None,
        "channels": None,
        "sample_width_bits": None,
        "encoding": None,
    }
    try:
        import soundfile as sf
        sf_info = sf.info(path)
        info["source_sample_rate"] = int(sf_info.samplerate)
        info["channels"] = int(sf_info.channels)
        info["encoding"] = str(sf_info.subtype or sf_info.format or "")
        subtype = str(sf_info.subtype or "").upper()
        for bits in (8, 16, 24, 32, 64):
            if str(bits) in subtype:
                info["sample_width_bits"] = bits
                break
    except Exception:
        pass
    return info


def _is_probably_stereo(path: str) -> bool:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if stem in STEREO_STEMS or "stereo" in stem:
        return True
    meta = _audio_probe(path)
    channels = meta.get("channels")
    return isinstance(channels, int) and channels >= 2

def _load_with_pydub(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    from pydub import AudioSegment
    seg = AudioSegment.from_file(path)
    seg = seg.set_channels(1).set_frame_rate(target_sr)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    max_val = float(1 << (8 * seg.sample_width - 1))
    return samples / max_val, target_sr


def load_audio(path: str, target_sr: int = TARGET_SR) -> Optional[LoadedAudio]:
    """Decode path to a mono float32 waveform at target_sr. Returns None on failure."""
    if not os.path.isfile(path):
        log.warning("skipping missing file: %s", path)
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        log.warning("skipping unsupported extension %s (%s)", ext, path)
        return None

    source_meta = _audio_probe(path)
    wav: Optional[np.ndarray] = None
    sr: Optional[int] = None

    # Fast path: soundfile (libsndfile) for WAV/FLAC — ~5-10× faster than librosa.
    if ext in (".wav", ".flac"):
        try:
            import soundfile as sf
            data, sr_native = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim == 2:
                source_meta["channels"] = int(data.shape[1])
                data = data.mean(axis=1)
            if sr_native != target_sr:
                data = librosa.resample(data, orig_sr=sr_native, target_sr=target_sr)
                sr_native = target_sr
            wav, sr = data, sr_native
        except Exception as sf_err:
            log.debug("soundfile failed on %s (%s); falling back to librosa", path, sf_err)
            wav = None

    # General path: librosa handles mp3, m4a, ogg + resamples in one call.
    if wav is None:
        try:
            wav, sr = librosa.load(path, sr=target_sr, mono=True)
        except Exception as err:
            log.info("librosa failed on %s (%s); retrying with pydub", path, err)
            try:
                wav, sr = _load_with_pydub(path, target_sr)
            except Exception as err2:
                log.error("unable to decode %s: %s", path, err2)
                return None

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=0)
    return LoadedAudio(
        path=path,
        waveform=wav,
        sample_rate=int(sr),
        duration=float(len(wav) / sr) if sr else 0.0,
        source_sample_rate=int(source_meta["source_sample_rate"] or sr or target_sr),
        channels=int(source_meta["channels"] or 1),
        sample_width_bits=(
            int(source_meta["sample_width_bits"])
            if source_meta.get("sample_width_bits") else None
        ),
        encoding=str(source_meta["encoding"] or ""),
    )


def _mix_two(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Production-safe mono mix.

    Normalise each speaker independently, RMS-match the quieter side, average,
    then peak-limit to avoid clipping. This preserves overlap timing while
    avoiding the old "sum and hope" failure mode.
    """
    n = max(len(a), len(b))
    def prep(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1e-6:
            y = y / peak * 0.95
        return y

    aa = prep(a)
    bb = prep(b)
    rms_a = float(np.sqrt(np.mean(aa ** 2))) if aa.size else 0.0
    rms_b = float(np.sqrt(np.mean(bb ** 2))) if bb.size else 0.0
    if rms_a > 1e-6 and rms_b > 1e-6:
        target = min(max((rms_a + rms_b) / 2.0, 1e-4), 0.35)
        aa *= min(target / rms_a, 2.0)
        bb *= min(target / rms_b, 2.0)

    out = np.zeros(n, dtype=np.float64)
    out[:len(aa)] += aa
    out[:len(bb)] += bb
    out *= 0.5
    peak = float(np.max(np.abs(out)))
    if peak > 0.98:
        out = out / peak * 0.95
    return out.astype(np.float32)


def load_speaker_pair(
    path1: str,
    label1: str,
    path2: str,
    label2: str,
    session_name: str = "session",
    target_sr: int = TARGET_SR,
) -> Optional[SpeakerPairAudio]:
    """Load two speaker files, mix them, return SpeakerPairAudio."""
    a1 = load_audio(path1, target_sr)
    a2 = load_audio(path2, target_sr)
    if a1 is None or a2 is None:
        return None

    mixed_wav = _mix_two(a1.waveform, a2.waveform)
    # Virtual path for the mixed clip: use parent of the first file.
    mixed_path = os.path.join(os.path.dirname(path1), f"{session_name}_mixed.wav")
    mixed = LoadedAudio(
        path=mixed_path,
        waveform=mixed_wav,
        sample_rate=target_sr,
        duration=float(len(mixed_wav) / target_sr),
        source_sample_rate=target_sr,
        channels=1,
        sample_width_bits=16,
        encoding="virtual_mix",
    )
    # Pipeline speaker labels (SPEAKER_00/01) → human labels (Host/Guest).
    speaker_map = {
        "SPEAKER_00": label1,
        "SPEAKER_01": label2,
    }
    return SpeakerPairAudio(
        session_name=session_name,
        speakers={label1: a1, label2: a2},
        mixed=mixed,
        speaker_map=speaker_map,
    )


def load_studio_speaker_folders(
    path1: str,
    label1: str,
    path2: str,
    label2: str,
    *,
    session_name: str,
    metadata: Optional[Dict] = None,
    validation_context: Optional[Dict] = None,
    target_sr: int = TARGET_SR,
) -> Optional[SpeakerPairAudio]:
    pair = load_speaker_pair(path1, label1, path2, label2, session_name=session_name, target_sr=target_sr)
    if pair is None:
        return None
    pair.recording_type = "studio_speaker_folders"
    pair.conversation_metadata = dict(metadata or {})
    pair.validation_context = dict(validation_context or {})
    return pair


def load_stereo_as_pair(
    path: str,
    label_left: str = "Speaker_L",
    label_right: str = "Speaker_R",
    session_name: Optional[str] = None,
    target_sr: int = TARGET_SR,
) -> Optional[SpeakerPairAudio]:
    """Load a stereo WAV and split channels into a SpeakerPairAudio.

    Convention: left channel = speaker 1, right channel = speaker 2.
    Works for telephony stereo recordings where each leg is isolated.
    """
    if not os.path.isfile(path):
        log.warning("stereo file missing: %s", path)
        return None

    source_meta = _audio_probe(path)
    try:
        wav_stereo, sr = librosa.load(path, sr=target_sr, mono=False)
    except Exception as err:
        log.error("cannot load stereo file %s: %s", path, err)
        return None

    wav_stereo = np.asarray(wav_stereo, dtype=np.float32)

    if wav_stereo.ndim == 1:
        # Already mono — duplicate into both channels
        ch_left = wav_stereo
        ch_right = wav_stereo.copy()
        log.warning("Expected stereo but got mono: %s — duplicating channel", path)
    elif wav_stereo.shape[0] >= 2:
        ch_left = wav_stereo[0]
        ch_right = wav_stereo[1]
    else:
        log.error("Unexpected channel count %d in %s", wav_stereo.shape[0], path)
        return None

    sname = session_name or os.path.splitext(os.path.basename(path))[0]

    a_left = LoadedAudio(
        path=path + "#left",
        waveform=ch_left,
        sample_rate=int(sr),
        duration=float(len(ch_left) / sr),
        source_sample_rate=int(source_meta["source_sample_rate"] or sr),
        channels=1,
        sample_width_bits=(
            int(source_meta["sample_width_bits"])
            if source_meta.get("sample_width_bits") else None
        ),
        encoding=str(source_meta["encoding"] or "stereo_channel"),
    )
    a_right = LoadedAudio(
        path=path + "#right",
        waveform=ch_right,
        sample_rate=int(sr),
        duration=float(len(ch_right) / sr),
        source_sample_rate=int(source_meta["source_sample_rate"] or sr),
        channels=1,
        sample_width_bits=(
            int(source_meta["sample_width_bits"])
            if source_meta.get("sample_width_bits") else None
        ),
        encoding=str(source_meta["encoding"] or "stereo_channel"),
    )

    mixed_wav = _mix_two(ch_left, ch_right)
    mixed = LoadedAudio(
        path=path,
        waveform=mixed_wav,
        sample_rate=int(sr),
        duration=float(len(mixed_wav) / sr),
        source_sample_rate=int(source_meta["source_sample_rate"] or sr),
        channels=int(source_meta["channels"] or 2),
        sample_width_bits=(
            int(source_meta["sample_width_bits"])
            if source_meta.get("sample_width_bits") else None
        ),
        encoding=str(source_meta["encoding"] or ""),
    )

    speaker_map = {
        "SPEAKER_00": label_left,
        "SPEAKER_01": label_right,
    }
    return SpeakerPairAudio(
        session_name=sname,
        speakers={label_left: a_left, label_right: a_right},
        mixed=mixed,
        speaker_map=speaker_map,
        recording_type="stereo",
    )


# ── speaker-pair detection ────────────────────────────────────────────────

def _audio_files_in_dir(dirpath: str) -> List[str]:
    """Audio files directly inside dirpath (non-recursive)."""
    out = []
    for name in sorted(os.listdir(dirpath)):
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
            full = os.path.abspath(os.path.join(dirpath, name))
            if os.path.isfile(full):
                out.append(full)
    return out


def _studio_audio_file(dirpath: str, speaker_label: str) -> str:
    files = _audio_files_in_dir(dirpath)
    if not files:
        raise ValueError(f"missing audio file in {speaker_label}")
    preferred = [
        p for p in files
        if os.path.splitext(p)[1].lower() in STUDIO_EXTS
    ]
    if not preferred:
        raise ValueError(f"{speaker_label} must contain WAV or FLAC audio")
    if len(preferred) > 1:
        exact = [p for p in preferred if _stem(p).lower() == speaker_label.lower()]
        if len(exact) == 1:
            return exact[0]
        raise ValueError(f"{speaker_label} must contain exactly one WAV/FLAC audio file")
    return preferred[0]


def validate_studio_conversation_folder(
    conversation_dir: str,
    *,
    allow_missing_metadata: bool = False,
) -> Tuple[str, str, str, str, str, Dict, Dict]:
    """Validate the canonical Sonexis studio folder and return pair info.

    Returns:
        (conversation_id, spk1_path, "speaker_1", spk2_path, "speaker_2",
         metadata, validation_context)
    """
    root = os.path.abspath(conversation_dir)
    if not os.path.isdir(root):
        raise ValueError(f"speaker_folders input must be a conversation directory: {root}")
    conversation_id = os.path.basename(root)
    metadata_path = os.path.join(root, "metadata.json")
    metadata: Dict = {}
    if not os.path.isfile(metadata_path):
        if not allow_missing_metadata:
            raise ValueError("metadata.json is required for speaker_folders input")
    else:
        import json
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json must contain a JSON object")
        if metadata.get("conversation_id") and str(metadata["conversation_id"]) != conversation_id:
            raise ValueError(
                f"metadata conversation_id {metadata['conversation_id']!r} does not match folder {conversation_id!r}"
            )
        for key in ("scenario_id", "scenario_name", "topic", "sub_topic", "conversation_style", "language_mix"):
            if key not in metadata:
                raise ValueError(f"metadata.json missing required field: {key}")
        if metadata.get("scripted") is not False:
            raise ValueError("metadata.json must set scripted: false for unscripted studio conversations")

    spk1_dir = os.path.join(root, "speaker_1")
    spk2_dir = os.path.join(root, "speaker_2")
    if not os.path.isdir(spk1_dir) or not os.path.isdir(spk2_dir):
        raise ValueError("speaker_folders input requires speaker_1/ and speaker_2/ subfolders")
    p1 = _studio_audio_file(spk1_dir, "speaker_1")
    p2 = _studio_audio_file(spk2_dir, "speaker_2")

    audio_checks: Dict[str, Dict] = {}
    warnings: List[str] = []
    for label, path in (("speaker_1", p1), ("speaker_2", p2)):
        ext = os.path.splitext(path)[1].lower()
        probe = _audio_probe(path)
        sr = int(probe.get("source_sample_rate") or 0)
        bit_depth = probe.get("sample_width_bits")
        check = {
            "path": path,
            "format": ext.lstrip("."),
            "source_sample_rate_hz": sr or None,
            "bit_depth": bit_depth,
            "channels": probe.get("channels"),
            "encoding": probe.get("encoding"),
            "format_ok": ext in STUDIO_EXTS,
            "sample_rate_preferred": sr == 48_000,
            "bit_depth_ok": bool(bit_depth is None or int(bit_depth) >= 16),
        }
        audio_checks[label] = check
        if ext == ".flac":
            warnings.append(f"{label}: FLAC input accepted; WAV is preferred")
        if sr and sr != 48_000:
            warnings.append(f"{label}: source sample rate {sr} Hz, 48000 Hz preferred")
        if bit_depth is not None and int(bit_depth) < 16:
            raise ValueError(f"{label}: bit depth {bit_depth} is below required 16-bit")
    validation_context = {
        "metadata_path": metadata_path if os.path.isfile(metadata_path) else None,
        "structure": {
            "conversation_dir": root,
            "speaker_1_dir": spk1_dir,
            "speaker_2_dir": spk2_dir,
        },
        "audio_format": audio_checks,
        "warnings": warnings,
    }
    return conversation_id, p1, "speaker_1", p2, "speaker_2", metadata, validation_context


def detect_studio_conversation_folders(root: str, *, allow_missing_metadata: bool = False) -> List[Tuple[str, str, str, str, str, Dict, Dict]]:
    root = os.path.abspath(root)
    if os.path.isdir(os.path.join(root, "speaker_1")) or os.path.isdir(os.path.join(root, "speaker_2")):
        return [validate_studio_conversation_folder(root, allow_missing_metadata=allow_missing_metadata)]
    out: List[Tuple[str, str, str, str, str, Dict, Dict]] = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        if not name.startswith("conversation_"):
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            out.append(validate_studio_conversation_folder(path, allow_missing_metadata=allow_missing_metadata))
    return out


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def detect_and_group_pairs(
    root: str,
) -> List[Tuple[str, str, str, str, str]]:
    """Scan root and return speaker-pair groups.

    Each tuple: (session_name, path1, label1, path2, label2)

    Handles patterns A, B, C described in module docstring.
    Returns empty list if no valid pairs found.
    """
    root = os.path.abspath(root)

    # Pattern A: root itself has exactly 2 audio files (single session).
    direct = _audio_files_in_dir(root)
    if len(direct) == 2:
        l1, l2 = _stem(direct[0]), _stem(direct[1])
        return [(os.path.basename(root), direct[0], l1, direct[1], l2)]

    pairs: List[Tuple[str, str, str, str, str]] = []

    # Walk one level of subdirectories.
    try:
        subdirs = sorted([
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and not d.startswith(".")
            and d != "__MACOSX"
        ])
    except PermissionError:
        return []

    for session_dir in subdirs:
        session_name = os.path.basename(session_dir)

        # Pattern A within subdir: session_dir has 2 audio files directly.
        direct = _audio_files_in_dir(session_dir)
        if len(direct) == 2:
            l1, l2 = _stem(direct[0]), _stem(direct[1])
            pairs.append((session_name, direct[0], l1, direct[1], l2))
            continue

        # Pattern B: session_dir has subdirs, each with exactly 1 audio file.
        spk_entries: List[Tuple[str, str]] = []  # (label, path)
        try:
            child_dirs = sorted([
                os.path.join(session_dir, d)
                for d in os.listdir(session_dir)
                if os.path.isdir(os.path.join(session_dir, d))
                and not d.startswith(".")
                and d != "__MACOSX"
            ])
        except PermissionError:
            continue
        for child in child_dirs:
            files = _audio_files_in_dir(child)
            if len(files) == 1:
                spk_entries.append((os.path.basename(child), files[0]))
        if len(spk_entries) == 2:
            (l1, p1), (l2, p2) = spk_entries
            pairs.append((session_name, p1, l1, p2, l2))

    return pairs


def detect_stereo_files(root: str) -> List[str]:
    """Return likely stereo session files under root.

    Auto mode only promotes files that are named like stereo recordings or whose
    container metadata reports 2+ channels. Mono-only audio stays on mono path.
    """
    candidates = list(iter_audio_files(root))
    return [p for p in candidates if _is_probably_stereo(p)]


def iter_audio_files(root: str) -> Iterable[str]:
    """Yield absolute paths of supported audio files under root."""
    if os.path.isfile(root):
        if os.path.splitext(root)[1].lower() in SUPPORTED_EXTS:
            yield os.path.abspath(root)
        return
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                yield os.path.abspath(os.path.join(dirpath, name))


def load_batch(root: str, target_sr: int = TARGET_SR) -> List[LoadedAudio]:
    out: List[LoadedAudio] = []
    for path in iter_audio_files(root):
        clip = load_audio(path, target_sr=target_sr)
        if clip is not None:
            out.append(clip)
    return out
