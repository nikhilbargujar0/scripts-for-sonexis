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

import copy
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


def _detect_device() -> str:
    """Return 'cuda' if a CUDA GPU is visible, else 'cpu'. No crash if torch absent."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class PipelineConfig:
    # ── input ─────────────────────────────────────────────────────────
    input_type: str = "auto"          # auto | speaker_folders | speaker_pair | stereo | mono
    allow_missing_metadata: bool = False

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
    initial_prompt: Optional[str] = None  # Whisper conditioning prompt
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    condition_on_previous_text: bool = False

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
    quality_score_threshold: float = 0.35  # segments below this score → "low quality"
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
    transcript_accuracy_target: float = 0.99
    timestamp_accuracy_target: float = 0.98
    metadata_review_required: bool = True
    pipeline_mode: str = "offline_standard"
    allow_paid_apis: bool = False
    require_human_review: bool = True
    review_threshold: float = 0.99
    speaker_accuracy_target: float = 0.99
    premium_engines: List[str] = field(default_factory=lambda: ["whisper_local", "deepgram", "google_stt_v2", "azure_speech"])
    punctuation_enabled: bool = True
    punctuation_device: Optional[str] = None
    punctuation_model: Optional[str] = None
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
    word_accuracy_target: float = 0.99
    code_switch_accuracy_target: float = 0.99

    def resolve(self) -> "PipelineConfig":
        """Resolve computed defaults in-place and return self.

        - device="auto"  → detected CUDA/CPU
        - compute_type="int8" on CUDA → upgraded to "float16"
        """
        if self.device == "auto":
            self.device = _detect_device()
        if self.device == "cuda" and self.compute_type == "int8":
            self.compute_type = "float16"
        return self

    @classmethod
    def colab_defaults(cls, **kwargs) -> "PipelineConfig":
        """Return a PipelineConfig tuned for Google Colab.

        Differences from stock defaults:
          - offline_mode=False  (Colab has internet; models download from HF)
          - device="auto"       (resolves to cuda if T4/V100/A100 attached)
          - compute_type auto-upgraded to float16 on GPU
          - ask_metadata=False  (stdin is not a tty in notebook cells)

        Any kwarg overrides the Colab default:
            cfg = PipelineConfig.colab_defaults(model_size="medium")
        """
        defaults: Dict = dict(
            offline_mode=False,
            device="auto",
            ask_metadata=False,
        )
        defaults.update(kwargs)
        return cls(**defaults).resolve()

    @classmethod
    def indic_defaults(cls, language: str, **kwargs) -> "PipelineConfig":
        """Config tuned for Indian languages to maximise accuracy.

        Targets 97-99 % WER on Hindi, Hinglish, Marwadi, and Punjabi by:
          - Upgrading to ``large-v3`` (biggest single accuracy gain)
          - Passing the correct Whisper language code (``hinglish`` and
            ``mwr`` both map to ``hi`` since Whisper has no separate codes)
          - Injecting a language-specific initial_prompt that cuts WER on
            code-switched and dialectal audio
          - Raising beam_size to 10 (marginal extra accuracy at cost of speed)
          - Enabling light noise reduction (denoise=True)
          - Enabling WhisperX forced-alignment for sub-word timestamps
          - Setting pipeline_mode="premium_accuracy" for per-speaker
            consensus and timestamp refinement

        Args:
            language: ``'hi'``, ``'hinglish'``, ``'mwr'``, or ``'pa'``.
                      Any BCP-47 code supported by faster-whisper also works
                      (e.g. ``'en'``, ``'ta'``).
            **kwargs: Override any PipelineConfig field, e.g.
                      ``model_size='medium'``, ``beam_size=5``.

        Example::

            cfg = PipelineConfig.indic_defaults("hinglish")
            cfg = PipelineConfig.indic_defaults("pa", model_size="medium")
        """
        # Map logical language names → Whisper BCP-47 codes.
        # faster-whisper does not recognise "hinglish" or "mwr".
        _whisper_lang: Dict[str, str] = {
            "hi": "hi",
            "hinglish": "hi",   # Hindi model; prompt steers code-switching
            "mwr": "hi",        # Marwadi → closest Whisper model is Hindi
            "pa": "pa",
        }
        # Language-specific initial prompts guide Whisper decoding.
        _prompts: Dict[str, str] = {
            "hi": (
                "यह एक हिंदी बातचीत है। "
                "वक्ता शुद्ध हिंदी में बोल रहे हैं।"
            ),
            "hinglish": (
                "यह हिंदी और अंग्रेजी मिश्रित (Hinglish) बातचीत है। "
                "वक्ता हिंदी और अंग्रेजी दोनों में बोलते हैं। "
                "This is a Hindi-English mixed conversation."
            ),
            "mwr": (
                "यह एक राजस्थानी / मारवाड़ी बातचीत है। "
                "वक्ता मारवाड़ी और हिंदी मिश्रित भाषा में बोल सकते हैं।"
            ),
            "pa": (
                "ਇਹ ਇੱਕ ਪੰਜਾਬੀ ਗੱਲਬਾਤ ਹੈ। "
                "ਬੁਲਾਰੇ ਪੰਜਾਬੀ ਅਤੇ ਕਦੇ-ਕਦੇ ਅੰਗਰੇਜ਼ੀ ਵਿੱਚ ਬੋਲ ਸਕਦੇ ਹਨ।"
            ),
        }
        lang_key = language.lower()
        whisper_lang = _whisper_lang.get(lang_key, lang_key)
        prompt = _prompts.get(lang_key)

        defaults: Dict = dict(
            offline_mode=False,
            device="auto",
            model_size="large-v3",
            beam_size=10,
            language=whisper_lang,
            initial_prompt=prompt,
            denoise=True,
            ask_metadata=False,
            input_type="speaker_folders",
            pipeline_mode="premium_accuracy",
        )
        defaults.update(kwargs)
        cfg = cls(**defaults).resolve()

        # Enable WhisperX forced-alignment unless caller passed a custom
        # premium dict.  Deep-copy to avoid mutating the class-level default.
        if "premium" not in kwargs:
            cfg.premium = copy.deepcopy(cfg.premium)
            cfg.premium["enabled"] = True
            cfg.premium.setdefault("alignment", {})["whisperx_enabled"] = True

        return cfg

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
        if "punctuation_enabled" in d and isinstance(d["punctuation_enabled"], str):
            d["punctuation_enabled"] = d["punctuation_enabled"].lower() in ("true", "1", "yes")
        if "store_transcript_candidates" in d and isinstance(d["store_transcript_candidates"], str):
            d["store_transcript_candidates"] = d["store_transcript_candidates"].lower() in ("true", "1", "yes")
        if "store_candidate_segments" in d and isinstance(d["store_candidate_segments"], str):
            d["store_candidate_segments"] = d["store_candidate_segments"].lower() in ("true", "1", "yes")
        if "store_candidate_words" in d and isinstance(d["store_candidate_words"], str):
            d["store_candidate_words"] = d["store_candidate_words"].lower() in ("true", "1", "yes")
        if "export_products" in d and isinstance(d["export_products"], str):
            d["export_products"] = [item.strip() for item in d["export_products"].split(",") if item.strip()]
        if "premium_engines" in d and isinstance(d["premium_engines"], str):
            d["premium_engines"] = [item.strip() for item in d["premium_engines"].split(",") if item.strip()]
        return cls.from_dict(d)
