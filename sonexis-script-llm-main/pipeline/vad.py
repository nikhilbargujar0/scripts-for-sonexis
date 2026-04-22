"""vad.py

Voice Activity Detection.

Primary backend: ``webrtcvad`` - extremely fast, CPU-only, no downloads.
Optional backend: ``silero-vad`` via torch.hub - higher recall but needs
torch. The webrtc backend is the default because it satisfies the "fully
offline, CPU-friendly" constraint without any model download.

Both backends emit the same shape of output: a list of ``(start, end)``
timestamps in seconds that describe speech-active regions, after a
small merge + pad pass to avoid over-fragmenting turns.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

Segment = Tuple[float, float]


@dataclass
class VADConfig:
    frame_ms: int = 30            # webrtcvad supports 10 / 20 / 30
    aggressiveness: int = 2       # 0 (lax) .. 3 (strict)
    min_speech_ms: int = 250      # drop micro-blips
    min_silence_ms: int = 300     # merge turns closer than this
    pad_ms: int = 120             # breathing room around each segment


def _to_pcm16(wav: np.ndarray) -> bytes:
    clipped = np.clip(wav, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _merge_segments(
    segments: List[Segment],
    min_silence: float,
    pad: float,
    total_duration: float,
) -> List[Segment]:
    if not segments:
        return []
    segments = sorted(segments)
    merged: List[Segment] = [segments[0]]
    for start, end in segments[1:]:
        prev_s, prev_e = merged[-1]
        if start - prev_e <= min_silence:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))
    padded: List[Segment] = []
    for s, e in merged:
        padded.append((max(0.0, s - pad), min(total_duration, e + pad)))
    return padded


def detect_speech_webrtc(
    wav: np.ndarray,
    sample_rate: int,
    cfg: VADConfig | None = None,
) -> List[Segment]:
    """Run webrtcvad on ``wav`` at ``sample_rate`` (must be 8/16/32/48 kHz)."""
    import webrtcvad

    cfg = cfg or VADConfig()
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError(f"webrtcvad requires 8/16/32/48 kHz, got {sample_rate}")

    vad = webrtcvad.Vad(cfg.aggressiveness)
    pcm = _to_pcm16(wav)
    frame_len = int(sample_rate * cfg.frame_ms / 1000) * 2  # int16 bytes per frame
    frames = [pcm[i : i + frame_len] for i in range(0, len(pcm), frame_len)]

    raw: List[Segment] = []
    open_start = None
    for idx, frame in enumerate(frames):
        if len(frame) < frame_len:
            break
        t = idx * cfg.frame_ms / 1000.0
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech and open_start is None:
            open_start = t
        elif not is_speech and open_start is not None:
            raw.append((open_start, t))
            open_start = None
    if open_start is not None:
        raw.append((open_start, len(wav) / sample_rate))

    min_len = cfg.min_speech_ms / 1000.0
    raw = [(s, e) for (s, e) in raw if (e - s) >= min_len]
    return _merge_segments(
        raw,
        min_silence=cfg.min_silence_ms / 1000.0,
        pad=cfg.pad_ms / 1000.0,
        total_duration=len(wav) / sample_rate,
    )


def detect_speech_silero(
    wav: np.ndarray,
    sample_rate: int,
    cfg: VADConfig | None = None,
) -> List[Segment]:  # pragma: no cover - optional path
    """Optional silero-vad backend. Requires torch + internet on first run."""
    import torch

    cfg = cfg or VADConfig()
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    tensor = torch.from_numpy(wav.astype(np.float32))
    ts = get_speech_timestamps(tensor, model, sampling_rate=sample_rate)
    raw = [(t["start"] / sample_rate, t["end"] / sample_rate) for t in ts]
    return _merge_segments(
        raw,
        min_silence=cfg.min_silence_ms / 1000.0,
        pad=cfg.pad_ms / 1000.0,
        total_duration=len(wav) / sample_rate,
    )


def detect_speech(
    wav: np.ndarray,
    sample_rate: int,
    backend: str = "webrtc",
    cfg: VADConfig | None = None,
) -> List[Segment]:
    """Unified entry point. ``backend`` in {"webrtc", "silero"}."""
    if backend == "silero":
        return detect_speech_silero(wav, sample_rate, cfg)
    return detect_speech_webrtc(wav, sample_rate, cfg)
