"""transcription.py

Local ASR using faster-whisper.

Offline mode (offline_mode=True in ASRConfig):
  - model_path MUST point to a local CTranslate2 directory.
  - If the directory is missing the pipeline fails immediately with a
    clear OfflineModeError (no silent HF download).

Online mode (default):
  - model_size is used; faster-whisper downloads weights into the HF
    cache on first use (same behaviour as before).

Model directory layout (produced by download_models.py):
    models/whisper/<model-size>/   ← CTranslate2 converted weights
"""
from __future__ import annotations

import logging
import math
import unicodedata as _uc
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def _segment_rms_db(wav: np.ndarray, sr: int, start: float, end: float) -> float:
    """Return RMS in dBFS for the ``[start, end]`` slice of ``wav``."""
    s = max(0, int(start * sr))
    e = min(len(wav), int(end * sr))
    if e <= s:
        return -120.0
    chunk = wav[s:e].astype(np.float64)
    rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
    if rms < 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)


def compute_quality_score(
    *,
    avg_logprob: float,
    compression_ratio: float,
    no_speech_prob: float,
    rms_db: float,
) -> float:
    """Combine per-segment Whisper + acoustic signals into a 0-1 score.

    Components:
      * logprob term   - exp(avg_logprob) maps (-inf..0] -> (0..1]
      * compression    - 1.0 up to 2.4 (whisper's threshold); linearly
                         penalised beyond until it reaches 0 at 3.4
      * speech term    - (1 - no_speech_prob)
      * loudness term  - saturates to 1 around -20 dBFS; drops below
                         -45 dBFS toward 0 (near-silence is unreliable)

    The final score is the product of all four components so a failure on
    any single axis drags the whole score down. Bounded to ``[0, 1]``.
    """
    logprob_term = math.exp(max(avg_logprob, -6.0))                       # (0..1]
    cr = float(compression_ratio)
    if cr <= 2.4:
        compression_term = 1.0
    elif cr >= 3.4:
        compression_term = 0.0
    else:
        compression_term = 1.0 - (cr - 2.4)
    speech_term = max(0.0, 1.0 - float(no_speech_prob))
    loudness_term = float(np.clip((rms_db + 45.0) / 25.0, 0.0, 1.0))      # -45..-20 dB
    score = logprob_term * compression_term * speech_term * loudness_term
    return float(np.clip(score, 0.0, 1.0))

Segment = Tuple[float, float]
_PIPELINE_SAMPLE_RATE = 16_000


def _text_normalized(text: str) -> str:
    """Lowercase and strip punctuation for WER scoring."""
    text = _uc.normalize("NFC", str(text or "").lower())
    kept = []
    for ch in text:
        if ch.isspace():
            kept.append(" ")
            continue
        category = _uc.category(ch)
        if category.startswith(("P", "S")):
            continue
        kept.append(ch)
    return " ".join("".join(kept).split())


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0
    language: str | None = None
    language_confidence: float | None = None
    script: str | None = None
    trailing_punct: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "probability": round(float(self.probability), 4),
            "language": self.language,
            "language_confidence": (
                round(float(self.language_confidence), 4)
                if self.language_confidence is not None
                else None
            ),
            "script": self.script,
            "trailing_punct": self.trailing_punct,
        }


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    language: str
    avg_logprob: float
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    rms_db: float = -120.0
    quality_score: float = 0.0
    words: List[Word] = field(default_factory=list)

    def to_dict(
        self,
        *,
        segment_id: Optional[str] = None,
        audio_filepath: Optional[str] = None,
        speaker_id: Optional[str] = None,
        sample_rate: int = _PIPELINE_SAMPLE_RATE,
    ) -> dict:
        duration = round(max(0.0, self.end - self.start), 3)
        missing = []
        if not getattr(self, "punctuation_applied", False):
            missing.append("punctuation")
        if not hasattr(self, "snr_db"):
            missing.append("snr_db")
        if not hasattr(self, "confidence"):
            missing.append("confidence")
        return {
            "segment_id": segment_id or "",
            "audio_filepath": audio_filepath or "",
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": duration,
            "speaker_id": speaker_id or "",
            "text": self.text,
            "text_normalized": _text_normalized(self.text),
            "language": self.language,
            "sample_rate": int(sample_rate),
            "avg_logprob": round(float(self.avg_logprob), 4),
            "compression_ratio": round(float(self.compression_ratio), 4),
            "no_speech_prob": round(float(self.no_speech_prob), 4),
            "rms_db": round(float(self.rms_db), 2),
            "quality_score": round(float(self.quality_score), 4),
            "words": [w.to_dict() for w in self.words],
            "snr_db": (
                round(float(getattr(self, "snr_db")), 2)
                if getattr(self, "snr_db", None) is not None
                else None
            ),
            "snr_band": getattr(self, "snr_band", None),
            "snr_reason": getattr(self, "snr_reason", None),
            "snr_source": getattr(self, "snr_source", None),
            "snr_quality": getattr(self, "snr_quality", None),
            "clipping_ratio": round(float(getattr(self, "clipping_ratio", 0.0) or 0.0), 4),
            "overlap": bool(getattr(self, "overlap", False)),
            # OVER-01: full overlap metadata fields
            "overlap_type": getattr(self, "overlap_type", None),
            "overlap_speakers": list(getattr(self, "overlap_speakers", []) or []),
            "overlap_start": getattr(self, "overlap_start", None),
            "overlap_end": getattr(self, "overlap_end", None),
            "confidence": getattr(self, "confidence", None),
            "confidence_band": getattr(self, "confidence_band", None),
            "confidence_band_absolute": getattr(self, "confidence_band_absolute", None),
            "confidence_band_relative": getattr(self, "confidence_band_relative", None),
            "confidence_reasons": list(getattr(self, "confidence_reasons", []) or []),
            "confidence_components": dict(getattr(self, "confidence_components", {}) or {}),
            "punctuation_applied": bool(getattr(self, "punctuation_applied", False)),
            "punct_skipped": getattr(self, "punct_skipped", None),
            "matrix_language": getattr(self, "matrix_language", self.language),
            "switch_points": list(getattr(self, "switch_points", []) or []),
            "cs_density": round(float(getattr(self, "cs_density", 0.0) or 0.0), 4),
            "missing_fields": missing,
        }


@dataclass
class Transcript:
    language: str
    language_probability: float
    duration: float
    segments: List[TranscriptSegment]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": round(float(self.language_probability), 4),
            "duration": round(self.duration, 3),
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
        }


@dataclass
class ASRConfig:
    model_size: str = "small"        # chosen by the user
    compute_type: str = "int8"       # CPU-friendly quantisation
    device: str = "cpu"
    beam_size: int = 5
    vad_filter: bool = False         # we run our own VAD upstream
    word_timestamps: bool = True
    language: Optional[str] = None   # None = auto-detect
    # Keep fillers: do NOT set suppress_tokens to aggressive lists and
    # keep condition_on_previous_text=False to avoid hallucinated fixes.
    condition_on_previous_text: bool = False
    # Temperature fallback ladder - re-decodes a segment at higher temperature
    # when compression-ratio or logprob thresholds flag likely repetition.
    # This stops Whisper from getting stuck in repetitive loops without
    # touching natural fillers (which are single tokens, not n-grams).
    # Keep the ladder short - each extra level is another full beam decode.
    temperature: Tuple[float, ...] = (0.0, 0.4, 0.8)
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    initial_prompt: Optional[str] = None
    # Emit a tqdm progress bar during decode.
    show_progress: bool = True
    # Offline mode: load from local path, never download.
    offline_mode: bool = False
    # Path to a CTranslate2 model directory (required when offline_mode=True).
    model_path: Optional[str] = None
    # ── speed knobs ────────────────────────────────────────────────────
    # batched: wrap model in BatchedInferencePipeline (faster-whisper >= 1.1)
    #          ~2-3× faster on CPU; falls back gracefully if unavailable.
    batched: bool = False
    batch_size: int = 16
    # cpu_threads: 0 = all cores; N = limit (useful when running N workers)
    cpu_threads: int = 0
    # num_workers: faster-whisper internal I/O workers for batched decoding
    num_workers: int = 1


class Transcriber:
    """Thin wrapper around faster-whisper that caches the model between calls."""

    def __init__(self, cfg: ASRConfig | None = None):
        self.cfg = cfg or ASRConfig()
        self._model = None
        self._batched_model = None

    def _load(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        # Build keyword args — only pass non-default speed params.
        model_kw: dict = {
            "device": self.cfg.device,
            "compute_type": self.cfg.compute_type,
        }
        if self.cfg.cpu_threads > 0:
            model_kw["cpu_threads"] = self.cfg.cpu_threads
        if self.cfg.num_workers > 1:
            model_kw["num_workers"] = self.cfg.num_workers

        if self.cfg.offline_mode:
            from .offline import require_model_dir
            path = self.cfg.model_path
            if not path:
                raise RuntimeError(
                    "[offline_mode] ASRConfig.model_path must be set. "
                    "Run: python download_models.py"
                )
            require_model_dir(path, label=f"whisper/{self.cfg.model_size}")
            log.info("loading faster-whisper from local path %s", path)
            self._model = WhisperModel(path, local_files_only=True, **model_kw)
        else:
            log.info(
                "loading faster-whisper model=%s compute_type=%s device=%s",
                self.cfg.model_size, self.cfg.compute_type, self.cfg.device,
            )
            self._model = WhisperModel(self.cfg.model_size, **model_kw)

        # Optionally wrap in BatchedInferencePipeline for ~2-3× CPU speedup.
        self._batched_model = None
        if self.cfg.batched:
            try:
                from faster_whisper import BatchedInferencePipeline
                self._batched_model = BatchedInferencePipeline(model=self._model)
                log.info("BatchedInferencePipeline active (batch_size=%d)", self.cfg.batch_size)
            except (ImportError, Exception) as exc:
                log.warning(
                    "BatchedInferencePipeline unavailable (%s); using standard model", exc
                )

        return self._model

    def transcribe(self, wav: np.ndarray, sample_rate: int) -> Transcript:
        model = self._load()
        if sample_rate != 16_000:
            raise ValueError("faster-whisper expects 16 kHz input")

        wav = np.asarray(wav, dtype=np.float32)

        # Use BatchedInferencePipeline when available for faster decoding.
        active = self._batched_model if self._batched_model is not None else model
        extra_kw: dict = {}
        if self._batched_model is not None:
            extra_kw["batch_size"] = self.cfg.batch_size

        seg_iter, info = active.transcribe(
            wav,
            beam_size=self.cfg.beam_size,
            language=self.cfg.language,
            vad_filter=self.cfg.vad_filter,
            word_timestamps=self.cfg.word_timestamps,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            temperature=self.cfg.temperature,
            compression_ratio_threshold=self.cfg.compression_ratio_threshold,
            log_prob_threshold=self.cfg.log_prob_threshold,
            no_speech_threshold=self.cfg.no_speech_threshold,
            initial_prompt=self.cfg.initial_prompt,
            **extra_kw,
        )

        total = float(info.duration or (len(wav) / sample_rate))
        bar = None
        if self.cfg.show_progress:
            try:
                from tqdm import tqdm

                bar = tqdm(
                    total=round(total, 2),
                    desc="ASR",
                    unit="s",
                    leave=False,
                    dynamic_ncols=True,
                )
            except ImportError:  # pragma: no cover
                bar = None

        segments: List[TranscriptSegment] = []
        last_end = 0.0
        for s in seg_iter:
            words: List[Word] = []
            if s.words:
                for w in s.words:
                    words.append(
                        Word(
                            text=w.word,
                            start=float(w.start or s.start),
                            end=float(w.end or s.end),
                            probability=float(w.probability or 0.0),
                        )
                    )

            seg_start = float(s.start)
            seg_end = float(s.end)
            rms_db = _segment_rms_db(wav, sample_rate, seg_start, seg_end)
            avg_logprob = float(getattr(s, "avg_logprob", 0.0))
            compression_ratio = float(getattr(s, "compression_ratio", 0.0))
            no_speech_prob = float(getattr(s, "no_speech_prob", 0.0))
            q = compute_quality_score(
                avg_logprob=avg_logprob,
                compression_ratio=compression_ratio,
                no_speech_prob=no_speech_prob,
                rms_db=rms_db,
            )

            segments.append(
                TranscriptSegment(
                    start=seg_start,
                    end=seg_end,
                    text=s.text.strip(),
                    language=info.language,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio,
                    no_speech_prob=no_speech_prob,
                    rms_db=rms_db,
                    quality_score=q,
                    words=words,
                )
            )

            if bar is not None:
                delta = max(0.0, seg_end - last_end)
                bar.update(delta)
                last_end = seg_end

        if bar is not None:
            # Close out remaining progress so the bar reaches 100 %.
            remainder = max(0.0, total - last_end)
            if remainder > 0:
                bar.update(remainder)
            bar.close()

        return Transcript(
            language=info.language,
            language_probability=float(info.language_probability or 0.0),
            duration=float(info.duration or (len(wav) / sample_rate)),
            segments=segments,
        )


# Filler markers we should NOT strip during normalisation. Keep bilingual.
FILLERS = {
    "um", "uh", "uhh", "hmm", "ah", "ahh", "erm", "er",
    "matlab", "yaani", "bas", "haan", "na", "nahi",
    "kya", "toh", "acha", "abhi",
}


def normalise_transcript(text: str) -> str:
    """Return a normalised transcript with whitespace squashed but fillers kept."""
    import re
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned
