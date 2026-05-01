"""Local-first premium ASR orchestration."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import PipelineConfig
from ..utils.premium_routing import build_routing_context, build_routing_decision
from .adapters.azure_speech_api import AzureSpeechAPIAdapter
from .adapters.deepgram_api import DeepgramAPIAdapter
from .adapters.google_stt_v2_api import GoogleSTTV2Adapter
from .adapters.whisper_local import WhisperLocalAdapter
from .types import RoutingDecision, TranscriptCandidate

log = logging.getLogger(__name__)


def _premium_cfg(cfg: PipelineConfig) -> Dict[str, Any]:
    return dict(getattr(cfg, "premium", {}) or {})


def _engine_cfg(cfg: PipelineConfig, engine_name: str) -> Dict[str, Any]:
    premium = _premium_cfg(cfg)
    engines = dict(premium.get("asr_engines") or {})
    return dict(engines.get(engine_name) or {})


def _allow_paid_apis(cfg: PipelineConfig) -> bool:
    premium = _premium_cfg(cfg)
    raw = premium.get("allow_paid_apis")
    if raw is None:
        raw = getattr(cfg, "allow_paid_apis", False)
    return bool(raw)


def _pipeline_mode(cfg: PipelineConfig) -> str:
    return str(getattr(cfg, "pipeline_mode", "offline_standard") or "offline_standard")


def _preferred_engines(cfg: PipelineConfig) -> List[str]:
    premium = _premium_cfg(cfg)
    values = premium.get("preferred_asr_engines") or getattr(cfg, "preferred_asr_engines", None)
    if values:
        return [str(value) for value in values]
    explicit = getattr(cfg, "premium_engines", None)
    if explicit:
        return [str(value) for value in explicit]
    return ["whisper_local", "deepgram", "google_stt_v2"]


def _engine_enabled(cfg: PipelineConfig, engine_name: str) -> bool:
    if engine_name == "whisper_local":
        return True
    if engine_name in set(getattr(cfg, "premium_engines", []) or []):
        return bool(_allow_paid_apis(cfg))
    config = _engine_cfg(cfg, engine_name)
    if "enabled" in config:
        return bool(config.get("enabled"))
    return False


class PremiumASRRouter:
    """Run local-first ASR and selectively escalate to paid providers."""

    def __init__(
        self,
        *,
        cfg: PipelineConfig,
        whisper_adapter: WhisperLocalAdapter,
        fasttext_lid=None,
        roman_indic_classifier=None,
    ) -> None:
        self.cfg = cfg
        self.whisper_adapter = whisper_adapter
        self.fasttext_lid = fasttext_lid
        self.roman_indic_classifier = roman_indic_classifier

    def _build_adapter(self, engine_name: str):
        config = _engine_cfg(self.cfg, engine_name)
        if engine_name == "deepgram":
            return DeepgramAPIAdapter(
                api_key_env=str(config.get("api_key_env") or "DEEPGRAM_API_KEY"),
                model=str(config.get("model") or "nova-2"),
                fasttext_lid=self.fasttext_lid,
                roman_indic_classifier=self.roman_indic_classifier,
            )
        if engine_name == "google_stt_v2":
            return GoogleSTTV2Adapter(
                credentials_env=str(config.get("credentials_env") or "GOOGLE_APPLICATION_CREDENTIALS"),
                recognizer=str(config.get("recognizer") or "_"),
                language_codes=list(config.get("language_codes") or ["en-IN", "hi-IN", "pa-IN"]),
                model=str(config.get("model") or "long"),
                fasttext_lid=self.fasttext_lid,
                roman_indic_classifier=self.roman_indic_classifier,
            )
        if engine_name == "azure_speech":
            return AzureSpeechAPIAdapter(
                api_key_env=str(config.get("api_key_env") or "AZURE_SPEECH_KEY"),
                region_env=str(config.get("region_env") or "AZURE_SPEECH_REGION"),
                enabled=bool(config.get("enabled")),
            )
        raise ValueError(f"Unsupported premium ASR engine: {engine_name}")

    def run(
        self,
        wav: np.ndarray,
        sample_rate: int,
        *,
        audio_meta: Optional[Dict[str, Any]] = None,
        overlap_duration_s: float = 0.0,
        review_priority: Optional[str] = None,
    ) -> Dict[str, Any]:
        candidates: List[TranscriptCandidate] = []
        attempted_engines: List[str] = []
        skipped_engines: List[str] = []
        engines_used: List[str] = []

        attempted_engines.append("whisper_local")
        local_candidate = self.whisper_adapter.transcribe(wav, sample_rate)
        candidates.append(local_candidate)
        engines_used.append(local_candidate.engine)

        context = build_routing_context(
            pipeline_mode=_pipeline_mode(self.cfg),
            allow_paid_apis=_allow_paid_apis(self.cfg),
            audio_meta=audio_meta,
            transcript=local_candidate.transcript,
            code_switch=local_candidate.code_switch_signals,
            overlap_duration_s=overlap_duration_s,
            review_priority=review_priority,
        )
        decision = build_routing_decision(context, attempted_engines=attempted_engines, skipped_engines=skipped_engines, engines_used=engines_used)
        force_multi_engine = _pipeline_mode(self.cfg) == "premium_accuracy" and _allow_paid_apis(self.cfg)
        if not decision.should_escalate and not force_multi_engine:
            return {
                "candidates": candidates,
                "routing_decision": decision,
                "local_candidate": local_candidate,
            }

        for engine_name in _preferred_engines(self.cfg):
            if engine_name == "whisper_local":
                continue
            if not _engine_enabled(self.cfg, engine_name):
                skipped_engines.append(engine_name)
                continue
            attempted_engines.append(engine_name)
            adapter = self._build_adapter(engine_name)
            try:
                candidate = adapter.transcribe(wav, sample_rate)
            except NotImplementedError as exc:
                log.info("Premium engine %s scaffolded but not active: %s", engine_name, exc)
                skipped_engines.append(engine_name)
                continue
            except RuntimeError as exc:
                message = str(exc)
                if "missing credentials" in message:
                    raise
                log.warning("Premium engine %s failed; keeping local fallback: %s", engine_name, exc)
                skipped_engines.append(engine_name)
                continue
            except Exception as exc:  # pragma: no cover
                log.warning("Premium engine %s failed unexpectedly; keeping local fallback: %s", engine_name, exc)
                skipped_engines.append(engine_name)
                continue
            candidates.append(candidate)
            engines_used.append(candidate.engine)

        decision = build_routing_decision(
            context,
            attempted_engines=attempted_engines,
            skipped_engines=skipped_engines,
            engines_used=engines_used,
        )
        return {
            "candidates": candidates,
            "routing_decision": decision,
            "local_candidate": local_candidate,
        }
