"""Premium ASR adapters."""

from .deepgram_api import DeepgramAPIAdapter
from .google_stt_v2_api import GoogleSTTV2Adapter
from .whisper_local import WhisperLocalAdapter

__all__ = [
    "DeepgramAPIAdapter",
    "GoogleSTTV2Adapter",
    "WhisperLocalAdapter",
]
