# Conversational Audio Dataset Hardening Design

## Goal

Upgrade the existing Sonexis Python pipeline into a reliable offline data
product for messy two-speaker conversations without turning it into a large
platform. Keep the existing modular pipeline and harden the product layer:
input resolution, validation, schema, manifests, reproducibility, and docs.

## Public-System Lessons

Whisper and faster-whisper are strong ASR engines, especially with local
CTranslate2 inference, word timestamps, int8 CPU mode, batching, and multilingual
support. They do not produce dataset structure, interaction metrics, or strict
manifests.

Pyannote.audio is strong diarisation infrastructure, but its model access is
often token-gated, it commonly emits diarisation artifacts rather than ML-ready
dataset records, and stereo behavior can erase separated-channel truth unless
the caller handles channels explicitly.

Common Voice, LibriSpeech, Kaldi, and torchaudio workflows are useful references
for simple manifests, deterministic identifiers, and loader APIs. They are weak
models for this product because they mostly assume read or pre-segmented speech
and miss overlap, interruptions, code-switching, and conversation-level
metadata.

## Design

Keep the current `pipeline/` modules. Add targeted hardening:

- CLI compatibility: `python main.py --input ./data --output ./dataset --mode both`.
- Offline-first default: local model loading by default; explicit
  `--allow-model-downloads` for experiments.
- Input handling: support separate speaker files and `stereo.wav`; auto mode
  prefers speaker pairs, then stereo, then mono.
- Validation: add a JSON-safe validation block with audio quality,
  transcript quality, diarisation fallback, no-speech/no-turn checks, and
  speaker-file duration mismatch.
- Schema v3: keep raw transcript untouched, add source provenance,
  validation, processing config, artifacts, language switching score, and
  confidence/method fields for inferred metadata.
- Manifests: strengthen conversation, speaker, and utterance JSONL rows with
  stable IDs, validation state, quality, language, and artifact paths.
- User-provided metadata: dialect, region, gender, age band, recording context,
  and consent status are never inferred. They come from CLI flags, metadata
  JSON, or interactive prompts, and are stored with source/confidence.

## Non-Goals

No UI/backend work. No paid APIs. No new heavyweight model family. No fake
precision for accent, device, or environment fields. No aggressive transcript
cleaning.

## Testing

Compile all Python files, run available unit tests when dependencies are
installed, and add focused tests for stereo detection and validation reporting.
