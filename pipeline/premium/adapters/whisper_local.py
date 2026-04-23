"""Local faster-whisper adapter wrapped in the premium candidate contract."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...transcription import ASRConfig, Transcriber
from ..types import TranscriptCandidate
from .base import build_candidate


class WhisperLocalAdapter:
    engine_name = "whisper_local"

    def __init__(
        self,
        transcriber: Optional[Transcriber] = None,
        *,
        asr_cfg: Optional[ASRConfig] = None,
        fasttext_lid=None,
        roman_indic_classifier=None,
    ) -> None:
        self.transcriber = transcriber
        self.asr_cfg = asr_cfg
        self.fasttext_lid = fasttext_lid
        self.roman_indic_classifier = roman_indic_classifier

    def transcribe(self, wav: np.ndarray, sample_rate: int) -> TranscriptCandidate:
        transcriber = self.transcriber or Transcriber(self.asr_cfg or ASRConfig())
        transcript = transcriber.transcribe(wav, sample_rate)
        return build_candidate(
            engine=self.engine_name,
            provider="local",
            paid_api=False,
            transcript=transcript,
            language_hint=transcript.language,
            timing_source="local_word_timestamps",
            adapter_metadata={
                "backend": "faster-whisper",
                "model_size": getattr(transcriber.cfg, "model_size", None),
            },
            fasttext_lid=self.fasttext_lid,
            roman_indic_classifier=self.roman_indic_classifier,
        )
