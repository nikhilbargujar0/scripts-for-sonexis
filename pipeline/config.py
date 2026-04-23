"""config.py

PipelineConfig: single configuration object for the entire pipeline.

Replaces the ad-hoc argparse.Namespace passed around in main.py.
Can be constructed from:
  - kwargs (direct instantiation)
  - a plain dict (e.g. loaded from YAML)
  - argparse.Namespace (from CLI)

Choosing output_mode:
  "both"             → write speaker_separated WAVs + mono mixed (recommended)
  "speaker_separated"→ write only per-speaker WAVs
  "mono"             → write only the mixed mono WAV

Choosing input_type:
  "auto"     → detect from folder structure (default)
  "speaker_pair" → two per-speaker files per session
  "stereo"   → one stereo WAV per session (left=spk1, right=spk2)
  "mono"     → single mixed file (existing mono pipeline)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class PipelineConfig:
    # ── input ─────────────────────────────────────────────────────────
    input_type: str = "auto"          # auto | speaker_pair | stereo | mono

    # ── output ────────────────────────────────────────────────────────
    output_mode: str = "both"         # both | speaker_separated | mono
    output_format: str = "json"       # json | jsonl | parquet (annotation format)
    dataset_name: str = "dataset"

    # ── audio processing ──────────────────────────────────────────────
    sample_rate_target: int = 16_000
    denoise: bool = False

    # ── ASR ───────────────────────────────────────────────────────────
    model_size: str = "small"         # tiny | base | small | medium | large-v3
    compute_type: str = "int8"
    device: str = "cpu"               # cpu | cuda | auto
    language: Optional[str] = None   # None = auto-detect

    # ── VAD ───────────────────────────────────────────────────────────
    vad_backend: str = "webrtc"       # webrtc | silero

    # ── diarisation (mono path only; speaker_pair uses ground-truth) ──
    diarisation_backend: str = "kmeans"   # kmeans | pyannote
    max_speakers: int = 4
    min_speakers: int = 1
    hf_token: Optional[str] = None

    # ── offline mode ──────────────────────────────────────────────────
    offline_mode: bool = True
    model_dir: Optional[str] = None

    # ── language classifier ───────────────────────────────────────────
    classifier: str = "auto"          # auto | on | off
    classifier_cache: Optional[str] = None
    fasttext_model: Optional[str] = None

    # ── quality thresholds ────────────────────────────────────────────
    max_duration_mismatch_s: float = 60.0  # warn above this
    min_audio_duration_s: float = 1.0

    # ── misc ──────────────────────────────────────────────────────────
    fail_fast: bool = False
    verbose: bool = False

    # ── interaction metadata ──────────────────────────────────────────
    interruption_threshold_s: float = 0.5  # overlaps shorter than this = interruption
    alignment_min_confidence: float = 0.35
    pair_merge_gap_s: float = 0.15
    pair_min_turn_duration_s: float = 0.08

    # ── performance / speed ───────────────────────────────────────────
    # num_workers: parallel session workers (>1 → ProcessPoolExecutor)
    # Each worker loads its own Whisper model; best for batches of files.
    num_workers: int = 1
    # ASR beam size: 5 = default quality, 1 = greedy (~2× faster, mild quality drop)
    beam_size: int = 5
    # asr_batched: use BatchedInferencePipeline (faster-whisper >= 1.1, ~2-3× faster)
    asr_batched: bool = False
    asr_batch_size: int = 16
    # cpu_threads: 0 = all cores; set to N to limit Whisper thread usage
    asr_cpu_threads: int = 0
    # skip_sha1: skip SHA1 hash of audio files (saves I/O on large files)
    skip_sha1: bool = False
    # Global deterministic seed for clustering and any future stochastic helpers.
    random_seed: int = 0
    # Stable default keeps JSON records deterministic unless the user overrides it.
    generated_at: str = "1970-01-01T00:00:00+00:00"
    include_runtime_metrics: bool = False
    enable_monologue_extraction: bool = True
    metadata_depth: str = "full"       # basic | full
    metadata_file: Optional[str] = None
    ask_metadata: bool = True
    accent: Optional[str] = None
    dialect: Optional[str] = None
    region: Optional[str] = None
    domain: Optional[str] = None
    gender: Optional[str] = None
    age_band: Optional[str] = None
    recording_context: Optional[str] = None
    consent_status: Optional[str] = None
    user_metadata: Dict = field(default_factory=dict)
    transcript_accuracy_target: float = 0.98
    timestamp_accuracy_target: float = 0.98
    metadata_review_required: bool = True
    pipeline_mode: str = "offline_standard"
    allow_paid_apis: bool = False
    require_human_review: bool = True
    export_products: List[str] = field(default_factory=lambda: ["stt", "diarisation", "evaluation_gold"])
    store_transcript_candidates: bool = True
    store_candidate_segments: bool = True
    store_candidate_words: bool = False
    premium: Dict = field(default_factory=lambda: {
        "enabled": False,
        "allow_paid_apis": False,
        "paid_budget_mode": "smart",
        "preferred_asr_engines": ["whisper_local", "deepgram", "google_stt_v2"],
        "preferred_alignment_engines": ["vendor_word_timestamps", "whisperx"],
        "require_human_review": True,
        "asr_engines": {
            "whisper_local": {"enabled": True},
            "deepgram": {"enabled": False, "api_key_env": "DEEPGRAM_API_KEY", "model": "nova-2"},
            "google_stt_v2": {
                "enabled": False,
                "credentials_env": "GOOGLE_APPLICATION_CREDENTIALS",
                "recognizer": "_",
                "language_codes": ["en-IN", "hi-IN", "pa-IN"],
                "model": "long",
            },
            "azure_speech": {
                "enabled": False,
                "api_key_env": "AZURE_SPEECH_KEY",
                "region_env": "AZURE_SPEECH_REGION",
            },
        },
        "alignment": {
            "whisperx_enabled": False,
            "vendor_word_timestamps_enabled": True,
        },
    })
    word_accuracy_target: float = 0.98
    code_switch_accuracy_target: float = 0.98

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_namespace(cls, ns) -> "PipelineConfig":
        """Build from argparse.Namespace (handles kebab→snake rename)."""
        d = {k.replace("-", "_"): v for k, v in vars(ns).items()}
        # argparse historically exposed --input-mode as input_mode while the
        # config field is input_type. Keep both names working.
        if "input_mode" in d and "input_type" not in d:
            d["input_type"] = d["input_mode"]
        if "interruption_threshold" in d and "interruption_threshold_s" not in d:
            d["interruption_threshold_s"] = d["interruption_threshold"]
        # offline_mode may arrive as string "true"/"false" from CLI
        if "offline_mode" in d and isinstance(d["offline_mode"], str):
            d["offline_mode"] = d["offline_mode"].lower() in ("true", "1", "yes")
        if "allow_paid_apis" in d and isinstance(d["allow_paid_apis"], str):
            d["allow_paid_apis"] = d["allow_paid_apis"].lower() in ("true", "1", "yes")
        if "require_human_review" in d and isinstance(d["require_human_review"], str):
            d["require_human_review"] = d["require_human_review"].lower() in ("true", "1", "yes")
        if "store_transcript_candidates" in d and isinstance(d["store_transcript_candidates"], str):
            d["store_transcript_candidates"] = d["store_transcript_candidates"].lower() in ("true", "1", "yes")
        if "store_candidate_segments" in d and isinstance(d["store_candidate_segments"], str):
            d["store_candidate_segments"] = d["store_candidate_segments"].lower() in ("true", "1", "yes")
        if "store_candidate_words" in d and isinstance(d["store_candidate_words"], str):
            d["store_candidate_words"] = d["store_candidate_words"].lower() in ("true", "1", "yes")
        if "export_products" in d and isinstance(d["export_products"], str):
            d["export_products"] = [item.strip() for item in d["export_products"].split(",") if item.strip()]
        return cls.from_dict(d)
