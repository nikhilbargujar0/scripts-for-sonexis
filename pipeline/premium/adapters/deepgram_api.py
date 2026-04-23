"""Deepgram premium ASR adapter using the official Python SDK."""
from __future__ import annotations

import io
import wave
from typing import Any, Dict, List, Optional

import numpy as np

from ...transcription import Transcript, TranscriptSegment, Word
from ..types import TranscriptCandidate
from .base import build_candidate, env_required


def _wav_bytes(wav_data: np.ndarray, sample_rate: int) -> bytes:
    audio = np.asarray(wav_data, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _response_to_dict(response: Any) -> Dict[str, Any]:
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "__dict__"):
        return dict(response.__dict__)
    if isinstance(response, dict):
        return response
    raise TypeError("Unsupported Deepgram response type")


def _words_from_items(items: List[Dict[str, Any]]) -> List[Word]:
    words: List[Word] = []
    for item in items or []:
        token = item.get("punctuated_word") or item.get("word") or item.get("text") or ""
        if not str(token).strip():
            continue
        words.append(
            Word(
                text=str(token).strip(),
                start=float(item.get("start") or 0.0),
                end=float(item.get("end") or item.get("start") or 0.0),
                probability=float(item.get("confidence") or 0.0),
            )
        )
    return words


def _segments_from_alternative(alternative: Dict[str, Any], language: str) -> List[TranscriptSegment]:
    utterances = alternative.get("utterances") or []
    if utterances:
        segments: List[TranscriptSegment] = []
        for item in utterances:
            words = _words_from_items(item.get("words") or [])
            confidence = float(item.get("confidence") or alternative.get("confidence") or 0.0)
            segments.append(
                TranscriptSegment(
                    start=float(item.get("start") or (words[0].start if words else 0.0)),
                    end=float(item.get("end") or (words[-1].end if words else 0.0)),
                    text=str(item.get("transcript") or "").strip(),
                    language=language,
                    avg_logprob=0.0,
                    quality_score=confidence,
                    words=words,
                )
            )
        return segments

    words = _words_from_items(alternative.get("words") or [])
    if words:
        return [
            TranscriptSegment(
                start=words[0].start,
                end=words[-1].end,
                text=str(alternative.get("transcript") or "").strip(),
                language=language,
                avg_logprob=0.0,
                quality_score=float(alternative.get("confidence") or 0.0),
                words=words,
            )
        ]

    return [
        TranscriptSegment(
            start=0.0,
            end=0.0,
            text=str(alternative.get("transcript") or "").strip(),
            language=language,
            avg_logprob=0.0,
            quality_score=float(alternative.get("confidence") or 0.0),
            words=[],
        )
    ]


class DeepgramAPIAdapter:
    engine_name = "deepgram"

    def __init__(
        self,
        *,
        api_key_env: str = "DEEPGRAM_API_KEY",
        model: str = "nova-2",
        smart_format: bool = True,
        punctuate: bool = True,
        detect_language: bool = True,
        diarize: bool = False,
        fasttext_lid=None,
        roman_indic_classifier=None,
    ) -> None:
        self.api_key_env = api_key_env
        self.model = model
        self.smart_format = smart_format
        self.punctuate = punctuate
        self.detect_language_enabled = detect_language
        self.diarize = diarize
        self.fasttext_lid = fasttext_lid
        self.roman_indic_classifier = roman_indic_classifier

    def _client(self):
        api_key = env_required(self.api_key_env, self.engine_name)
        try:
            from deepgram import DeepgramClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Deepgram SDK not installed. Add the deepgram package to dependencies.") from exc
        return DeepgramClient(api_key)

    def transcribe(self, wav: np.ndarray, sample_rate: int) -> TranscriptCandidate:
        client = self._client()
        payload = {"buffer": _wav_bytes(wav, sample_rate)}
        try:
            from deepgram import PrerecordedOptions
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Deepgram SDK not installed. Add the deepgram package to dependencies.") from exc

        options = PrerecordedOptions(
            model=self.model,
            smart_format=self.smart_format,
            punctuate=self.punctuate,
            detect_language=self.detect_language_enabled,
            diarize=self.diarize,
            utterances=True,
            paragraphs=True,
            words=True,
        )
        response = client.listen.rest.v("1").transcribe_file(payload, options)
        data = _response_to_dict(response)
        results = ((data.get("results") or {}).get("channels") or [{}])[0]
        alternative = (results.get("alternatives") or [{}])[0]
        detected_language = (
            alternative.get("detected_language")
            or results.get("detected_language")
            or alternative.get("language")
            or "und"
        )
        segments = _segments_from_alternative(alternative, detected_language)
        duration = max((segment.end for segment in segments), default=0.0)
        transcript = Transcript(
            language=detected_language,
            language_probability=float(alternative.get("confidence") or 0.0),
            duration=duration,
            segments=segments,
        )
        metadata = data.get("metadata") or {}
        return build_candidate(
            engine=self.engine_name,
            provider="deepgram",
            paid_api=True,
            transcript=transcript,
            confidence=float(alternative.get("confidence") or 0.0),
            avg_word_probability=(
                sum(word.probability for segment in segments for word in segment.words)
                / max(sum(len(segment.words) for segment in segments), 1)
                if any(segment.words for segment in segments) else None
            ),
            language_hint=detected_language,
            timing_source="vendor_word_timestamps",
            timestamp_score=1.0 if any(segment.words for segment in segments) else 0.5,
            adapter_metadata={
                "request_id": metadata.get("request_id"),
                "model": self.model,
                "sdk": "official",
            },
            fasttext_lid=self.fasttext_lid,
            roman_indic_classifier=self.roman_indic_classifier,
        )
