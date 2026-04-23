"""diarisation.py

Speaker diarisation.

Primary backend: MFCC + KMeans clustering. It is fully offline and
requires no HuggingFace token, satisfying the "no token" constraint.

Optional backend: pyannote.audio (speaker-diarization-3.1). Higher
accuracy but needs a free HuggingFace token the first time the
weights are downloaded. See ``diarise_pyannote`` at the bottom of
this file.

Primary pipeline:
1. Split VAD-detected speech regions into short fixed-length windows.
2. Extract MFCC + delta statistics per window (mean / std pooling).
3. Estimate the number of speakers (2..max_speakers) via silhouette
   scoring on the pooled embeddings.
4. Cluster windows with KMeans and merge adjacent windows assigned to
   the same speaker into contiguous speaker turns.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)

Segment = Tuple[float, float]


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str
    # Silhouette-based confidence from the clustering step (0.0 = uncertain).
    confidence: float = 0.0

    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
            "duration": round(self.duration(), 3),
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class DiarisationConfig:
    window_s: float = 1.5
    hop_s: float = 0.75
    n_mfcc: int = 20
    max_speakers: int = 4
    min_speakers: int = 1
    random_state: int = 0
    merge_gap_s: float = 0.4
    # Accepted speaker labels in clustering output.
    label_prefix: str = "SPEAKER_"
    # Drop analysis windows shorter than this from a VAD segment.
    min_window_s: float = 0.5
    extras: dict = field(default_factory=dict)


def _iter_windows(
    seg: Segment, window_s: float, hop_s: float, min_window_s: float
) -> List[Segment]:
    start, end = seg
    length = end - start
    if length < min_window_s:
        return [(start, end)]
    windows: List[Segment] = []
    t = start
    while t + window_s <= end + 1e-6:
        windows.append((t, min(end, t + window_s)))
        t += hop_s
    if not windows or windows[-1][1] < end - 0.05:
        windows.append((max(start, end - window_s), end))
    return windows


def _embed_window(
    wav: np.ndarray, sr: int, seg: Segment, n_mfcc: int
) -> np.ndarray:
    import librosa

    s = int(seg[0] * sr)
    e = int(seg[1] * sr)
    chunk = wav[s:e]
    if chunk.size < int(0.1 * sr):  # sub-100ms → pad to avoid librosa errors
        chunk = np.pad(chunk, (0, int(0.1 * sr) - chunk.size))
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    feats = np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1), delta.std(axis=1)]
    )
    # Normalise to keep KMeans well-conditioned.
    norm = np.linalg.norm(feats) + 1e-8
    return (feats / norm).astype(np.float32)


def _estimate_n_speakers(
    embeddings: np.ndarray, min_k: int, max_k: int, random_state: int
) -> Tuple[int, float]:
    """Return (n_speakers, silhouette_confidence)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = embeddings.shape[0]
    if n <= 2 or min_k == max_k:
        return max(min_k, 1), 0.5
    upper = min(max_k, n - 1)
    if upper < 2:
        return 1, 0.5
    best_k, best_score = 1, -1.0
    for k in range(max(min_k, 2), upper + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(embeddings, labels)
        except ValueError:
            continue
        if score > best_score:
            best_score, best_k = score, k
    # 0.1 is a deliberately low bar for noisy conversational data.
    if best_score < 0.10 and min_k <= 1:
        return 1, max(0.0, best_score + 0.5)
    # Map silhouette [-1..1] to confidence [0..1].
    confidence = float(np.clip((best_score + 1.0) / 2.0, 0.0, 1.0))
    return best_k, confidence


def _merge_turns(turns: List[SpeakerTurn], gap_s: float) -> List[SpeakerTurn]:
    """Collapse sliding windows into contiguous, non-overlapping speaker spans.

    Sliding-window clustering produces overlapping windows; we attribute each
    window to its speaker and then split contested regions at the midpoint so
    downstream consumers see exactly one speaker per moment in time.
    Confidence is preserved as the mean of merged window confidences.
    """
    if not turns:
        return []
    turns = sorted(turns, key=lambda t: (t.start, t.end))
    # First pass - resolve overlaps; split contested regions at the midpoint.
    resolved: List[SpeakerTurn] = []
    for t in turns:
        if not resolved:
            resolved.append(SpeakerTurn(t.start, t.end, t.speaker, t.confidence))
            continue
        last = resolved[-1]
        if t.start < last.end:
            if t.speaker == last.speaker:
                avg_conf = (last.confidence + t.confidence) / 2.0
                resolved[-1] = SpeakerTurn(last.start, max(last.end, t.end), last.speaker, avg_conf)
                continue
            boundary = (max(last.start, t.start) + min(last.end, t.end)) / 2.0
            resolved[-1] = SpeakerTurn(last.start, boundary, last.speaker, last.confidence)
            if t.end > boundary:
                resolved.append(SpeakerTurn(boundary, t.end, t.speaker, t.confidence))
        else:
            resolved.append(SpeakerTurn(t.start, t.end, t.speaker, t.confidence))

    # Second pass - merge same-speaker spans separated by small gaps.
    merged: List[SpeakerTurn] = [resolved[0]]
    for t in resolved[1:]:
        last = merged[-1]
        if t.speaker == last.speaker and (t.start - last.end) <= gap_s:
            avg_conf = (last.confidence + t.confidence) / 2.0
            merged[-1] = SpeakerTurn(last.start, max(last.end, t.end), last.speaker, avg_conf)
        else:
            merged.append(t)
    return [t for t in merged if t.end - t.start > 1e-3]


def diarise(
    wav: np.ndarray,
    sample_rate: int,
    speech_segments: List[Segment],
    cfg: DiarisationConfig | None = None,
) -> List[SpeakerTurn]:
    """Assign a speaker label to every chunk of speech.

    Returns a list of ``SpeakerTurn`` objects spanning every speech
    segment. If only one cluster is detected, every turn gets
    ``SPEAKER_00``.
    """
    cfg = cfg or DiarisationConfig()
    if not speech_segments:
        return []

    # Step 1 - window the VAD segments.
    windows: List[Segment] = []
    for seg in speech_segments:
        windows.extend(_iter_windows(seg, cfg.window_s, cfg.hop_s, cfg.min_window_s))

    if not windows:
        return []

    # Step 2 - embeddings.
    embeddings = np.stack(
        [_embed_window(wav, sample_rate, w, cfg.n_mfcc) for w in windows]
    )

    # Step 3 + 4 - cluster.
    cluster_confidence = 0.5
    if embeddings.shape[0] < 2:
        labels = np.zeros(embeddings.shape[0], dtype=int)
    else:
        from sklearn.cluster import KMeans

        n_speakers, cluster_confidence = _estimate_n_speakers(
            embeddings, cfg.min_speakers, cfg.max_speakers, cfg.random_state
        )
        if n_speakers <= 1:
            labels = np.zeros(embeddings.shape[0], dtype=int)
            cluster_confidence = 0.5  # single-speaker is low-confidence diarisation
        else:
            km = KMeans(
                n_clusters=n_speakers,
                random_state=cfg.random_state,
                n_init=10,
            )
            labels = km.fit_predict(embeddings)

    raw_turns = [
        SpeakerTurn(
            start=windows[i][0],
            end=windows[i][1],
            speaker=f"{cfg.label_prefix}{int(labels[i]):02d}",
            confidence=cluster_confidence,
        )
        for i in range(len(windows))
    ]
    return _merge_turns(raw_turns, cfg.merge_gap_s)


def count_speakers(turns: List[SpeakerTurn]) -> int:
    return len({t.speaker for t in turns})


def diarise_from_speaker_vad(
    speaker_vad: "Dict[str, List[Segment]]",
    label_prefix: str = "SPEAKER_",
    merge_gap_s: float = 0.4,
    min_turn_duration_s: float = 0.05,
    preserve_overlaps: bool = True,
) -> "Tuple[List[SpeakerTurn], Dict[str, str]]":
    """Build ground-truth speaker turns from per-speaker VAD segments.

    Used when audio files are already separated by speaker — no clustering
    needed. Confidence is 1.0 on all turns (the separation is exact).

    Args:
        speaker_vad: {human_label: [(start, end), ...]}
                     e.g. {"Host": [(0.0, 2.3), ...], "Guest": [...]}

    Returns:
        (turns, speaker_map) where
        speaker_map = {"SPEAKER_00": "Host", "SPEAKER_01": "Guest"}
    """
    from typing import Dict as _Dict
    turns: List[SpeakerTurn] = []
    speaker_map: _Dict[str, str] = {}

    for i, (human_label, segments) in enumerate(speaker_vad.items()):
        pipeline_label = f"{label_prefix}{i:02d}"
        speaker_map[pipeline_label] = human_label
        for start, end in segments:
            if end - start < 1e-3:
                continue
            turns.append(SpeakerTurn(
                start=start,
                end=end,
                speaker=pipeline_label,
                confidence=1.0,   # ground truth — no clustering uncertainty
            ))

    if preserve_overlaps:
        grouped: Dict[str, List[SpeakerTurn]] = {}
        for turn in turns:
            grouped.setdefault(turn.speaker, []).append(turn)
        merged: List[SpeakerTurn] = []
        for speaker, spk_turns in grouped.items():
            ordered = sorted(spk_turns, key=lambda t: (t.start, t.end))
            current = ordered[0]
            for turn in ordered[1:]:
                if (turn.start - current.end) <= merge_gap_s:
                    avg_conf = (current.confidence + turn.confidence) / 2.0
                    current = SpeakerTurn(
                        current.start,
                        max(current.end, turn.end),
                        speaker,
                        avg_conf,
                    )
                else:
                    if current.duration() >= min_turn_duration_s:
                        merged.append(current)
                    current = turn
            if current.duration() >= min_turn_duration_s:
                merged.append(current)
        merged.sort(key=lambda t: (t.start, t.end, t.speaker))
        return merged, speaker_map

    merged = _merge_turns(sorted(turns, key=lambda t: t.start), merge_gap_s)
    merged = [t for t in merged if t.duration() >= min_turn_duration_s]
    return merged, speaker_map



# ---------------------------------------------------------------------------
#  Optional pyannote.audio backend.
# ---------------------------------------------------------------------------
#
# pyannote.audio is licensed under MIT but its pretrained weights live on
# HuggingFace behind a click-through gate that requires a (free) HF token.
# This module makes pyannote strictly OPT-IN: the default pipeline stays
# 100 % offline via the MFCC + KMeans path above.
#
# To use pyannote:
#   1. Create a free HuggingFace account.
#   2. Accept the terms on https://hf.co/pyannote/speaker-diarization-3.1
#      and https://hf.co/pyannote/segmentation-3.0
#   3. Export the token: ``export HUGGINGFACE_HUB_TOKEN=hf_xxx``
#   4. Run with ``--diarisation-backend pyannote``

def diarise_pyannote(
    wav: np.ndarray,
    sample_rate: int,
    hf_token: "str | None" = None,
    min_speakers: "int | None" = None,
    max_speakers: "int | None" = None,
    merge_gap_s: float = 0.4,
    label_prefix: str = "SPEAKER_",
    local_model_dir: "str | None" = None,
    offline_mode: bool = False,
) -> List[SpeakerTurn]:
    """Run pyannote speaker diarisation and return merged turns.

    Offline mode: set local_model_dir to the directory created by
    download_models.py (models/diarisation/pyannote). No HF token needed.

    Online mode (default): hf_token is required; weights are fetched from HF.

    Raises RuntimeError on any failure so the caller can fall back to KMeans.
    """
    import os

    try:
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError as err:
        raise RuntimeError(
            "pyannote.audio is not installed. pip install pyannote.audio"
        ) from err

    if local_model_dir or offline_mode:
        # Load from local checkpoint — no token or network required.
        model_dir = local_model_dir
        if not model_dir:
            from .offline import default_model_dir, pyannote_local_path
            model_dir = pyannote_local_path(default_model_dir())
        if not os.path.isdir(model_dir):
            raise RuntimeError(
                f"[offline_mode] pyannote model dir not found: {model_dir!r}. "
                "Run: python download_models.py"
            )
        config_yml = os.path.join(model_dir, "config.yaml")
        if not os.path.isfile(config_yml):
            raise RuntimeError(
                f"[offline_mode] config.yaml missing in {model_dir!r}. "
                "Run: python download_models.py"
            )
        try:
            pipeline = PyannotePipeline.from_pretrained(model_dir)
        except Exception as err:
            raise RuntimeError(f"failed to load local pyannote pipeline: {err}") from err
    else:
        token = hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "pyannote requires a HuggingFace token. Set HUGGINGFACE_HUB_TOKEN "
                "or pass --hf-token. Alternatively use --diarisation-backend kmeans."
            )
        try:
            pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
        except Exception as err:
            raise RuntimeError(f"failed to load pyannote pipeline: {err}") from err

    tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    kwargs: dict = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        kwargs["max_speakers"] = int(max_speakers)

    annotation = pipeline(
        {"waveform": tensor, "sample_rate": int(sample_rate)},
        **kwargs,
    )

    # Remap pyannote labels to stable integer indices in order of first appearance.
    # pyannote doesn't emit per-segment silhouette; use 0.8 as a fixed prior
    # (pyannote is generally more accurate than KMeans on clean audio).
    remap: dict = {}
    turns: List[SpeakerTurn] = []
    for segment, _, raw_label in annotation.itertracks(yield_label=True):
        if raw_label not in remap:
            remap[raw_label] = f"{label_prefix}{len(remap):02d}"
        turns.append(SpeakerTurn(
            start=float(segment.start),
            end=float(segment.end),
            speaker=remap[raw_label],
            confidence=0.8,
        ))
    return _merge_turns(turns, merge_gap_s)
