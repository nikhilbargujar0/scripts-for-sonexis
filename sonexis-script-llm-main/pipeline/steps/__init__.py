"""pipeline/steps — modular step implementations for the Sonexis pipeline.

Each sub-module corresponds to one stage of the processing chain:

    audio_loader   — decode and enumerate input sessions
    preprocessing  — normalise, DC-remove, optional denoise
    alignment      — cross-correlation clock alignment for speaker pairs
    transcription  — faster-whisper ASR
    language       — language + code-switching detection
    metadata       — audio, speaker, and conversation metadata
    interaction    — overlap, interruption, and latency metrics
    monologue      — best-segment extractor for per-speaker monologues
    validation     — pre-flight and post-processing quality checks
    output         — assemble and validate the final dataset record
"""
from .alignment import (
    AlignmentError,
    AlignmentResult,
    align_speaker_pair,
)
from .audio_loader import (
    LoadedAudio,
    SpeakerPairAudio,
    detect_and_group_pairs,
    detect_stereo_files,
    iter_audio_files,
    load_audio,
    load_batch,
    load_speaker_pair,
    load_stereo_as_pair,
)
from .interaction import (
    OverlapSegment,
    extract_interaction_metadata,
)
from .language import (
    FastTextLID,
    LanguageReport,
    LanguageSegment,
    detect_language,
    detect_language_per_speaker,
)
from .metadata import (
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)
from .monologue import (
    Monologue,
    MonologueConfig,
    extract_audio_clip,
    extract_monologue,
    extract_monologues_per_speaker,
)
from .output import build_record, validate_record
from .preprocessing import (
    PreprocessConfig,
    preprocess,
    preprocess_speaker_pair,
)
from .transcription import ASRConfig, Transcript, Transcriber
from .validation import build_validation_report, write_validation_report

__all__ = [
    # alignment
    "AlignmentError", "AlignmentResult", "align_speaker_pair",
    # audio loading
    "LoadedAudio", "SpeakerPairAudio",
    "load_audio", "load_speaker_pair", "load_stereo_as_pair",
    "iter_audio_files", "load_batch", "detect_and_group_pairs", "detect_stereo_files",
    # preprocessing
    "PreprocessConfig", "preprocess", "preprocess_speaker_pair",
    # transcription
    "ASRConfig", "Transcriber", "Transcript",
    # language
    "FastTextLID", "LanguageReport", "LanguageSegment",
    "detect_language", "detect_language_per_speaker",
    # metadata
    "extract_audio_metadata", "extract_speaker_metadata", "extract_conversation_metadata",
    # interaction
    "OverlapSegment", "extract_interaction_metadata",
    # monologue
    "Monologue", "MonologueConfig",
    "extract_monologue", "extract_monologues_per_speaker", "extract_audio_clip",
    # validation
    "build_validation_report", "write_validation_report",
    # output
    "build_record", "validate_record",
]
