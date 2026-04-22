# Offline Setup Guide

This pipeline is designed to run without API calls during dataset generation.
Download model files once, then run with `--offline-mode true` (the default).

## 1. Install Open-Source Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For direct requirements install:

```bash
pip install -r pipeline/requirements.txt
```

Install `ffmpeg` if you need robust `mp3` or `m4a` decoding.

## 2. Download Local Models

```bash
python download_models.py --model-dir ./models --whisper-size small
```

This creates:

```text
models/
  whisper/small/
  fasttext/lid.176.ftz
```

Optional pyannote diarisation:

```bash
python download_models.py --model-dir ./models --with-pyannote --hf-token hf_xxx
```

Pyannote weights require accepting the relevant Hugging Face model terms once.
Runtime processing still remains local after the model files exist.

## 3. Run Dataset Generation

```bash
python main.py --input ./data --output ./dataset --mode both --model-dir ./models
```

`--mode both` writes speaker-separated audio and mixed mono audio.

If dialect, region, gender, or consent fields matter, provide them explicitly:

```bash
python main.py --input ./data --output ./dataset --mode both \
  --metadata-file ./metadata.json
```

When run interactively, the CLI asks for missing speaker metadata. Press Enter
to store `unknown`; the pipeline does not infer these user-side fields.

## 4. Expected Dataset Layout

```text
dataset/
  audio/
    raw/
    speaker_separated/
    mono/
    monologues/
  transcripts/
  annotations/
  manifests/
    utterances.jsonl
    conversations.jsonl
    speakers.jsonl
  logs/
    pipeline.log
  schema.json
```

## 5. Reproducibility Defaults

- Runtime model downloads are disabled by default.
- `--random-seed 0` controls deterministic clustering.
- `--generated-at 1970-01-01T00:00:00+00:00` is the default stable record
  timestamp; pass another fixed value if you need release-specific metadata.
- Wall-clock `processing_time_s` is omitted unless `--include-runtime-metrics`
  is set.
- KMeans diarisation uses fixed `random_state`.
- Validation records fallback usage, audio quality warnings, empty transcript
  risk, and speaker-file duration mismatch.

Use `--allow-model-downloads` only for local experimentation, not production
dataset builds.
