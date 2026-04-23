from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.premium.asr_router import PremiumASRRouter
from pipeline.premium.types import TranscriptCandidate
from pipeline.transcription import Transcript, TranscriptSegment, Word


def _candidate(
    engine: str,
    *,
    confidence: float,
    timestamp_confidence: float,
    text: str = "hello namaste",
    code_switch: bool = False,
) -> TranscriptCandidate:
    transcript = Transcript(
        language="en",
        language_probability=confidence,
        duration=2.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text=text,
                language="en",
                avg_logprob=-0.1,
                quality_score=confidence,
                words=[
                    Word("hello", 0.0, 0.8, confidence),
                    Word("namaste", 0.9, 1.8, confidence),
                ],
            )
        ],
    )
    return TranscriptCandidate(
        engine=engine,
        provider=engine,
        paid_api=engine != "whisper_local",
        transcript=transcript,
        confidence=confidence,
        avg_word_confidence=confidence,
        language_hint="hinglish" if code_switch else "en",
        detected_languages=["English", "Hindi"] if code_switch else ["English"],
        code_switch_signals={
            "detected": code_switch,
            "dominant_languages": ["English", "Hindi"] if code_switch else ["English"],
            "switch_count": 4 if code_switch else 0,
            "switch_patterns": ["en->hi"] if code_switch else [],
            "switching_score": 0.6 if code_switch else 0.0,
        },
        timing_source="vendor_word_timestamps" if engine != "whisper_local" else "local_word_timestamps",
        timestamp_confidence=timestamp_confidence,
    )


class _DummyWhisperAdapter:
    def __init__(self, candidate: TranscriptCandidate) -> None:
        self.candidate = candidate

    def transcribe(self, _wav, _sample_rate) -> TranscriptCandidate:
        return self.candidate


class _DummyAdapter:
    def __init__(self, candidate: TranscriptCandidate | None = None, error: Exception | None = None) -> None:
        self.candidate = candidate
        self.error = error

    def transcribe(self, _wav, _sample_rate) -> TranscriptCandidate:
        if self.error is not None:
            raise self.error
        assert self.candidate is not None
        return self.candidate


class PremiumRoutingTests(unittest.TestCase):
    def test_router_keeps_local_only_when_paid_apis_disabled(self) -> None:
        cfg = PipelineConfig(
            pipeline_mode="premium_accuracy",
            allow_paid_apis=False,
            premium={
                "enabled": True,
                "allow_paid_apis": False,
                "preferred_asr_engines": ["whisper_local", "deepgram"],
                "asr_engines": {"deepgram": {"enabled": True, "api_key_env": "DEEPGRAM_API_KEY"}},
            },
        )
        router = PremiumASRRouter(
            cfg=cfg,
            whisper_adapter=_DummyWhisperAdapter(_candidate("whisper_local", confidence=0.92, timestamp_confidence=0.9)),
        )

        result = router.run(np.zeros(32000, dtype=np.float32), 16000, audio_meta={"environment": {"value": "indoor"}})

        self.assertEqual(len(result["candidates"]), 1)
        self.assertFalse(result["routing_decision"].should_escalate)
        self.assertEqual(result["routing_decision"].engines_used, ["whisper_local"])

    def test_router_escalates_when_local_quality_is_low(self) -> None:
        cfg = PipelineConfig(
            pipeline_mode="premium_accuracy",
            allow_paid_apis=True,
            premium={
                "enabled": True,
                "allow_paid_apis": True,
                "preferred_asr_engines": ["whisper_local", "deepgram"],
                "asr_engines": {"deepgram": {"enabled": True, "api_key_env": "DEEPGRAM_API_KEY"}},
            },
        )
        router = PremiumASRRouter(
            cfg=cfg,
            whisper_adapter=_DummyWhisperAdapter(_candidate("whisper_local", confidence=0.32, timestamp_confidence=0.2, code_switch=True)),
        )
        with patch.object(router, "_build_adapter", return_value=_DummyAdapter(_candidate("deepgram", confidence=0.9, timestamp_confidence=0.95, code_switch=True))):
            result = router.run(
                np.zeros(32000, dtype=np.float32),
                16000,
                audio_meta={"environment": {"value": "outdoor"}, "noise_level": {"value": "high"}},
            )

        self.assertEqual([candidate.engine for candidate in result["candidates"]], ["whisper_local", "deepgram"])
        self.assertTrue(result["routing_decision"].should_escalate)
        self.assertIn("deepgram", result["routing_decision"].engines_used)

    def test_router_fails_clearly_when_credentials_are_missing(self) -> None:
        cfg = PipelineConfig(
            pipeline_mode="premium_accuracy",
            allow_paid_apis=True,
            premium={
                "enabled": True,
                "allow_paid_apis": True,
                "preferred_asr_engines": ["whisper_local", "deepgram"],
                "asr_engines": {"deepgram": {"enabled": True, "api_key_env": "DEEPGRAM_API_KEY"}},
            },
        )
        router = PremiumASRRouter(
            cfg=cfg,
            whisper_adapter=_DummyWhisperAdapter(_candidate("whisper_local", confidence=0.31, timestamp_confidence=0.2, code_switch=True)),
        )
        with patch.object(router, "_build_adapter", return_value=_DummyAdapter(error=RuntimeError("deepgram requested but missing credentials in env var DEEPGRAM_API_KEY"))):
            with self.assertRaises(RuntimeError):
                router.run(
                    np.zeros(32000, dtype=np.float32),
                    16000,
                    audio_meta={"environment": {"value": "outdoor"}, "noise_level": {"value": "high"}},
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
