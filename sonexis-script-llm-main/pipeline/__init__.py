"""Sonexis offline conversational-audio pipeline — v3.0.0."""

from .audio_loader import (
    LoadedAudio,
    SpeakerPairAudio,
    load_audio,
    load_speaker_pair,
    load_stereo_as_pair,
    iter_audio_files,
    load_batch,
    detect_and_group_pairs,
    detect_stereo_files,
)
from .batch_writer import SUPPORTED_FORMATS, BatchWriter
from .config import PipelineConfig
from .dataset_writer import DatasetWriter
from .diarisation import (
    DiarisationConfig,
    SpeakerTurn,
    diarise,
    diarise_pyannote,
    diarise_from_speaker_vad,
)
from .interaction_metadata import (
    OverlapSegment,
    extract_interaction_metadata,
)
from .language_detection import (
    FastTextLID,
    LanguageReport,
    LanguageSegment,
    detect_language,
    detect_language_per_speaker,
)
from .metadata_extraction import (
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)
from .monologue_extractor import (
    Monologue,
    MonologueConfig,
    extract_monologue,
    extract_monologues_per_speaker,
    extract_audio_clip,
)
from .offline import OfflineModeError, require_model_file, require_model_dir
from .output_formatter import build_record
from .preprocessing import PreprocessConfig, preprocess
from .quality_checker import QualityReport, check_mono, check_speaker_pair
from .validation import build_validation_report
from .roman_indic_classifier import (
    ClassifierPrediction,
    RomanIndicClassifier,
)
from .transcription import ASRConfig, Transcriber, Transcript
from .vad import VADConfig, detect_speech

__all__ = [
    # audio loading
    "LoadedAudio", "SpeakerPairAudio",
    "load_audio", "load_speaker_pair", "load_stereo_as_pair",
    "iter_audio_files", "load_batch", "detect_and_group_pairs", "detect_stereo_files",
    # config
    "PipelineConfig",
    # preprocessing
    "PreprocessConfig", "preprocess",
    # VAD
    "VADConfig", "detect_speech",
    # diarisation
    "DiarisationConfig", "SpeakerTurn",
    "diarise", "diarise_pyannote", "diarise_from_speaker_vad",
    # ASR
    "ASRConfig", "Transcriber", "Transcript",
    # language detection
    "FastTextLID", "LanguageReport", "LanguageSegment",
    "detect_language", "detect_language_per_speaker",
    # classifiers
    "RomanIndicClassifier", "ClassifierPrediction",
    # metadata
    "extract_audio_metadata", "extract_speaker_metadata",
    "extract_conversation_metadata",
    # interaction metadata
    "OverlapSegment", "extract_interaction_metadata",
    # quality
    "QualityReport", "check_mono", "check_speaker_pair",
    "build_validation_report",
    # monologue
    "Monologue", "MonologueConfig",
    "extract_monologue", "extract_monologues_per_speaker", "extract_audio_clip",
    # output
    "build_record", "BatchWriter", "SUPPORTED_FORMATS",
    # dataset writer
    "DatasetWriter",
    # offline
    "OfflineModeError", "require_model_file", "require_model_dir",
]
