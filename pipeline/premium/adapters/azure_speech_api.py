"""Azure Speech adapter scaffold for future premium provider support."""
from __future__ import annotations

import os
from typing import Optional


class AzureSpeechAPIAdapter:
    engine_name = "azure_speech"

    def __init__(
        self,
        *,
        api_key_env: str = "AZURE_SPEECH_KEY",
        region_env: str = "AZURE_SPEECH_REGION",
        enabled: bool = False,
    ) -> None:
        self.api_key_env = api_key_env
        self.region_env = region_env
        self.enabled = enabled

    def validate_configuration(self) -> dict:
        return {
            "engine": self.engine_name,
            "enabled": bool(self.enabled),
            "api_key_present": bool(os.environ.get(self.api_key_env)),
            "region_present": bool(os.environ.get(self.region_env)),
            "supported": False,
        }

    def transcribe(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Azure Speech scaffold is configured for future use but not implemented in this production pass."
        )
