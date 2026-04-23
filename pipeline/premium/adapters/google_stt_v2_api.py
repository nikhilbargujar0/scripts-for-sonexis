"""Google Speech-to-Text v2 premium adapter using the official client."""
from __future__ import annotations

import io
import wave
from typing import List, Optional

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


def _seconds(value) -> float:
    if value is None:
        return 0.0
    seconds = getattr(value, "seconds", None)
    nanos = getattr(value, "nanos", None)
    if seconds is not None or nanos is not None:
        return float(seconds or 0.0) + float(nanos or 0.0) / 1_000_000_000.0
    return float(value or 0.0)


class GoogleSTTV2Adapter:
    engine_name = "google_stt_v2"

    def __init__(
        self,
        *,
        credentials_env: str = "GOOGLE_APPLICATION_CREDENTIALS",
        recognizer: str = "_",
        language_codes: Optional[List[str]] = None,
        model: str = "long",
        fasttext_lid=None,
        roman_indic_classifier=None,
    ) -> None:
        self.credentials_env = credentials_env
        self.recognizer = recognizer
        self.language_codes = language_codes or ["en-IN", "hi-IN", "pa-IN"]
        self.model = model
        self.fasttext_lid = fasttext_lid
        self.roman_indic_classifier = roman_indic_classifier

    def _client(self):
        env_required(self.credentials_env, self.engine_name)
        try:
            from google.cloud import speech_v2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Google Speech-to-Text v2 client not installed. Add google-cloud-speech to dependencies."
            ) from exc
        return speech_v2.SpeechClient()

    def transcribe(self, wav: np.ndarray, sample_rate: int) -> TranscriptCandidate:
        env_required(self.credentials_env, self.engine_name)
        try:
            from google.cloud.speech_v2 import SpeechClient
            from google.cloud.speech_v2.types import cloud_speech
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Google Speech-to-Text v2 client not installed. Add google-cloud-speech to dependencies."
            ) from exc

        client = SpeechClient()
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=self.language_codes,
            model=self.model,
            features=cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True,
                enable_word_confidence=True,
            ),
        )
        request = cloud_speech.RecognizeRequest(
            recognizer=self.recognizer,
            config=config,
            content=_wav_bytes(wav, sample_rate),
        )
        response = client.recognize(request=request)

        segments: List[TranscriptSegment] = []
        transcript_parts: List[str] = []
        language_code = self.language_codes[0] if self.language_codes else "und"
        alt_confidences: List[float] = []
        word_confidences: List[float] = []
        for result in response.results:
            if not result.alternatives:
                continue
            alt = result.alternatives[0]
            transcript_parts.append(str(alt.transcript).strip())
            alt_confidences.append(float(getattr(alt, "confidence", 0.0) or 0.0))
            words: List[Word] = []
            for item in getattr(alt, "words", []) or []:
                confidence = float(getattr(item, "confidence", 0.0) or 0.0)
                word_confidences.append(confidence)
                words.append(
                    Word(
                        text=str(getattr(item, "word", "")).strip(),
                        start=_seconds(getattr(item, "start_offset", None)),
                        end=_seconds(getattr(item, "end_offset", None)),
                        probability=confidence,
                    )
                )
            if words:
                start = words[0].start
                end = words[-1].end
            else:
                start = 0.0
                end = 0.0
            language_code = str(getattr(result, "language_code", None) or language_code or "und")
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=str(alt.transcript).strip(),
                    language=language_code,
                    avg_logprob=0.0,
                    quality_score=float(getattr(alt, "confidence", 0.0) or 0.0),
                    words=words,
                )
            )

        transcript = Transcript(
            language=language_code,
            language_probability=(sum(alt_confidences) / len(alt_confidences)) if alt_confidences else 0.0,
            duration=max((segment.end for segment in segments), default=0.0),
            segments=segments,
        )
        metadata = getattr(response, "metadata", None)
        return build_candidate(
            engine=self.engine_name,
            provider="google",
            paid_api=True,
            transcript=transcript,
            confidence=(sum(alt_confidences) / len(alt_confidences)) if alt_confidences else 0.0,
            avg_word_probability=(sum(word_confidences) / len(word_confidences)) if word_confidences else None,
            language_hint=language_code,
            timing_source="vendor_word_timestamps",
            timestamp_score=1.0 if word_confidences else 0.5,
            adapter_metadata={
                "recognizer": self.recognizer,
                "model": self.model,
                "sdk": "official",
                "request_id": getattr(metadata, "request_id", None) if metadata is not None else None,
            },
            fasttext_lid=self.fasttext_lid,
            roman_indic_classifier=self.roman_indic_classifier,
        )
