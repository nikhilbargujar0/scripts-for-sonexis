# Sonexis-style Offline Conversational-Audio Pipeline

A production-grade, **fully offline** Python pipeline that turns raw
conversational audio (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`) into a
structured JSON dataset. Designed for real-world Indian conversational
data (English, Hindi, Hinglish, Punjabi, Marwadi) - disfluencies,
code-switching and interruptions are preserved, not scrubbed.

> No paid APIs. No cloud calls. No tokens. Only open-source models run
> locally on CPU (GPU optional).

---

## What you get per audio file

A single JSON document with:

- **`transcript`** - raw (fillers kept) + normalised versions, plus
  word-level timestamps. Each segment also carries `avg_logprob`,
  `compression_ratio`, `no_speech_prob`, `rms_db` and a combined
  `quality_score` in `[0, 1]` so you can filter low-confidence spans
  without re-decoding.
- **`speaker_segmentation`** - turn-by-turn speaker labels
  (`SPEAKER_00`, `SPEAKER_01`, ...).
- **`metadata.audio`** - duration, RMS (dB), estimated SNR, RT60-proxy
  reverb, spectral centroid, environment class, device estimate.
- **`metadata.speakers`** - per-speaker WPM, pause rate, filler ratio,
  speaking style (fluent / deliberate / rapid / hesitant /
  conversational), coarse accent proxy.
- **`metadata.language`** - primary language, confidence, detected
  scripts, code-switching flag, per-segment language.
- **`metadata.conversation`** - turn count, speaker count, average turn
  length, TF-IDF topic keywords, rule-based intents, and a `quality`
  block summarising mean / median / duration-weighted segment quality
  scores plus low/high-confidence segment ratios - ready for dataset
  filtering.
- **`monologue_sample`** - the longest uninterrupted 10-30 s single
  speaker span with timestamps and transcript.

Full example at the bottom of this file.

---

## Pipeline architecture

```
audio_loader  ->  preprocessing  ->  vad  ->  diarisation
                                              |
                                              v
                       transcription  ->  language_detection
                                              |
                                              v
                   metadata_extraction  ->  monologue_extractor
                                              |
                                              v
                                   output_formatter -> JSON
```

Every stage lives in its own module under `pipeline/`, matching the
mandated layout:

```
pipeline/
├── __init__.py
├── audio_loader.py
├── preprocessing.py
├── vad.py
├── diarisation.py          # kmeans + optional pyannote backend
├── transcription.py
├── language_detection.py
├── roman_indic_classifier.py
├── metadata_extraction.py
├── monologue_extractor.py
├── output_formatter.py
├── batch_writer.py         # json / jsonl / parquet
└── main.py
```

### Key design decisions

| Concern                 | Choice                                  | Why |
| ----------------------- | --------------------------------------- | --- |
| ASR                     | `faster-whisper` (`small`, `int8` CPU)  | Fully local, multilingual, word timestamps. |
| VAD                     | `webrtcvad-wheels` (silero optional)    | Pure-CPU, no model download, deterministic. |
| Diarisation             | **MFCC + KMeans with silhouette `k` estimation** (default) or pyannote.audio (optional, token-gated) | No HuggingFace token required by default. |
| Language ID             | Trained char-ngram classifier (sklearn) + script detection + romanised-Indic lexicon + optional `fasttext lid.176` refinement | Classifier handles Hinglish / Punjabi / Marwadi that single-source approaches miss. |
| Topic modelling         | TF-IDF top-k (scikit-learn)             | Fast, interpretable, no LLM. |
| Intent                  | Rule-based regex patterns (bilingual)   | Deterministic, auditable, no LLM. |
| Monologue selection     | Longest single-speaker span in 10-30 s with gap-merge | Handles imperfect diarisation. |
| Output formats          | `json` per file / `jsonl` / `parquet`   | Plug directly into ML pipelines. |

### What we intentionally DO NOT do

- No aggressive denoising, no profanity filter, no filler stripping.
  Real conversational datasets keep "uh", "matlab", "haan", "yaar".
- No API calls. Not to OpenAI, Google, Azure, AssemblyAI or any paid
  service.
- No mandatory HuggingFace token. `pyannote.audio` is deliberately
  not used.

---

## Installation

Tested on Python 3.10+.

```bash
# 1. Create a virtualenv (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install the package - installs the sonexis-pipeline CLI
pip install -e .
# ...or just the requirements without the CLI shim:
# pip install -r pipeline/requirements.txt

# 3. Optional extras
pip install -e .[parquet]    # pandas + pyarrow (enabled by default in requirements.txt)
pip install -e .[pyannote]   # pyannote.audio + torch (token-gated)

# 4. (Optional) Install ffmpeg for mp3/m4a fallback decoding
#    Ubuntu:   sudo apt-get install -y ffmpeg
#    macOS:    brew install ffmpeg
```

Download local models before the first production run:

```bash
python download_models.py --model-dir ./models --whisper-size small
```

Three equivalent ways to run the pipeline:

```bash
sonexis-pipeline --input ./audio --output ./dataset      # after pip install -e .
python -m pipeline.main --input ./audio --output ./dataset
python main.py --input ./audio --output ./dataset --mode both
```

### Optional: fasttext language-ID model

For improved language detection on clean single-script inputs, download
the public lid.176 model once:

```bash
mkdir -p ~/.cache/sonexis
curl -L -o ~/.cache/sonexis/lid.176.ftz \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

Then point the CLI at it (or rely on the default path):

```bash
python -m pipeline.main --input ./audio --output ./dataset \
    --fasttext-model ~/.cache/sonexis/lid.176.ftz
```

The pipeline works without fasttext - it will just use the built-in
script + romanised-Indic heuristic.

---

## CLI usage

```bash
python main.py --input ./data --output ./dataset --mode both
```

Flags:

| Flag                    | Default   | Description |
| ----------------------- | --------- | ----------- |
| `--input`               | *required*| File or folder. Recurses into subfolders. |
| `--output`              | *required*| Output directory. |
| `--input-mode`          | `auto`    | `auto` / `speaker_pair` / `stereo` / `mono`. Auto detects two-speaker folders first, then stereo files. |
| `--mode`                | `both`    | Alias for output mode: `both` / `speaker_separated` / `mono`. |
| `--output-mode`         | `both`    | Same as `--mode`; retained for compatibility. |
| `--output-format`       | `json`    | `json` (per-file) / `jsonl` / `parquet` |
| `--dataset-name`        | `dataset` | Base filename for `jsonl` / `parquet` output. |
| `--model-size`          | `small`   | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `--compute-type`        | `int8`    | faster-whisper compute type. `int8` is best on CPU. |
| `--device`              | `cpu`     | `cpu` / `cuda` / `auto` |
| `--language`            | auto      | Force a language code, e.g. `hi`. |
| `--vad-backend`         | `webrtc`  | `webrtc` or `silero` |
| `--diarisation-backend` | `kmeans`  | `kmeans` (no token) or `pyannote` (HF token required) |
| `--hf-token`            | env       | HuggingFace token for pyannote (or set `HUGGINGFACE_HUB_TOKEN`). |
| `--max-speakers`        | `4`       | Upper bound for `k` in KMeans. |
| `--min-speakers`        | `1`       | Lower bound. Set to `2` to force diarisation. |
| `--denoise`             | off       | Enable mild stationary noise reduction. |
| `--fasttext-model`      | env or cache path | Path to `lid.176.ftz`. |
| `--classifier`          | `auto`    | `auto` / `on` / `off` — trained romanised-Indic classifier. |
| `--classifier-cache`    | cache path| Path to the classifier pickle cache. |
| `--offline-mode`        | `true`    | Load models from local `--model-dir`; no runtime downloads. |
| `--allow-model-downloads` | off      | Allow faster-whisper/HF downloads at runtime. Not recommended for production datasets. |
| `--model-dir`           | `./models` or `$SONEXIS_MODEL_DIR` | Local model root. |
| `--random-seed`         | `0`       | Fixed seed for deterministic clustering. |
| `--generated-at`        | `1970-01-01T00:00:00+00:00` | Stable timestamp stored in records for deterministic JSON. |
| `--include-runtime-metrics` | off    | Include non-deterministic wall-clock `processing_time_s` in records. |
| `--metadata-file`       | none      | JSON metadata file. Supports global `*` and speaker-label keys. |
| `--ask-metadata` / `--no-ask-metadata` | ask when TTY | Ask for missing user-provided metadata; disabled automatically in non-interactive runs. |
| `--dialect`, `--region`, `--gender` | none | User-provided global metadata. Stored with `source=cli`; never inferred. |
| `--age-band`, `--recording-context`, `--consent-status` | none | Additional user-provided global metadata. Stored with `source=cli`; never inferred. |
| `--verbose`             | off       | Debug-level logging. |
| `--fail-fast`           | off       | Abort the batch on the first failure. |

### User-provided metadata

Dialect, region, gender, age band, recording context, and consent status are
never inferred. The CLI asks for missing values only when stdin is interactive.
Press Enter to store `unknown`; non-interactive runs must use flags or a JSON
file if those fields are required.

Global metadata file:

```json
{
  "*": {
    "region": "Rajasthan",
    "dialect": "Marwadi",
    "recording_context": "customer_support",
    "consent_status": "consented"
  }
}
```

Per-speaker metadata file:

```json
{
  "speaker_1": { "gender": "female", "region": "Delhi" },
  "speaker_2": { "gender": "male", "region": "Punjab", "dialect": "Punjabi" }
}
```

Stored shape:

```json
{
  "provided_metadata": {
    "region": {
      "value": "Rajasthan",
      "confidence": 1.0,
      "source": "interactive_prompt"
    }
  }
}
```

### Supported input layouts

Separate speaker files:

```text
data/
  conversation_0001/
    speaker_1.wav
    speaker_2.wav
```

Stereo conversation file:

```text
data/
  conversation_0001/
    stereo.wav
```

Supported source containers: `wav`, `flac`, `mp3`, `m4a`, `ogg`. The pipeline
records source sample rate/channel metadata and processes internally at 16 kHz,
which is what faster-whisper, VAD and diarisation expect.

### Output formats

Default: one `<name>.json` per input file under `--output`.

Single-dataset modes:

```bash
# One JSONL file (streaming friendly):
sonexis-pipeline --input ./audio --output ./ds --output-format jsonl --dataset-name convs
# -> ./ds/convs.jsonl  (one JSON record per line)

# One Parquet file (columnar, pandas/DuckDB friendly):
sonexis-pipeline --input ./audio --output ./ds --output-format parquet --dataset-name convs
# -> ./ds/convs.parquet
```

The parquet schema flattens the nested JSON: scalar columns like
`duration_s`, `primary_language`, `mean_quality_score`, ... and
JSON-encoded string columns for the nested structures
(`transcript_segments_json`, `speaker_segmentation_json`,
`speakers_json`, `topic_keywords`, `intents`, `scripts`). That way you
can filter conversations quickly (`df.query('mean_quality_score > 0.7')`)
without losing any detail.

### Optional: pyannote diarisation

Higher accuracy diarisation via [pyannote.audio](https://github.com/pyannote/pyannote-audio):

```bash
pip install -e .[pyannote]
# Accept the model terms once at https://hf.co/pyannote/speaker-diarization-3.1
export HUGGINGFACE_HUB_TOKEN=hf_xxx
sonexis-pipeline --input ./audio --output ./ds --diarisation-backend pyannote
```

The default `kmeans` backend still runs without any token.

### Trained romanised-Indic classifier

A small character-ngram LogisticRegression model (trained on the
bundled corpus in ``roman_indic_classifier.py``) kicks in automatically
for Latin-script text. It catches common Hinglish / Punjabi patterns
that plain fasttext lid.176 misses. The pickle is cached under
``~/.cache/sonexis/roman_indic.pkl`` after the first run. Disable it
with ``--classifier off``.

### Batch processing

Point `--input` at a folder:

```bash
python -m pipeline.main --input ./recordings --output ./dataset --verbose
```

Each audio file produces `<name>.json` in the output folder.

### Programmatic usage

```python
from pipeline import load_audio, preprocess, detect_speech, diarise, \
    Transcriber, detect_language, extract_audio_metadata, \
    extract_speaker_metadata, extract_conversation_metadata, \
    extract_monologue, build_record
from pipeline.transcription import FILLERS

clip = load_audio("call.wav")
wav  = preprocess(clip.waveform, clip.sample_rate)
speech = detect_speech(wav, clip.sample_rate)
turns = diarise(wav, clip.sample_rate, speech)
transcript = Transcriber().transcribe(wav, clip.sample_rate)
lang = detect_language(transcript.text,
                       [s.text for s in transcript.segments])

record = build_record(
    audio_path=clip.path,
    transcript=transcript,
    turns=turns,
    language=lang,
    audio_meta=extract_audio_metadata(wav, clip.sample_rate, speech),
    speaker_meta=extract_speaker_metadata(
        transcript, turns, lang.primary_language, lang.scripts, FILLERS),
    conversation_meta=extract_conversation_metadata(transcript, turns),
    monologue=extract_monologue(transcript, turns),
)
```

---

## Example output (truncated)

```json
{
  "schema_version": "3.0.0",
  "generated_at": "1970-01-01T00:00:00+00:00",
  "input_mode": "speaker_pair",
  "session_name": "conversation_0001",
  "file": {
    "name": "conversation_0001_mixed.wav",
    "note": "virtual_mixed_audio"
  },
  "metadata": {
    "audio": {
      "duration_s": 60.0,
      "sample_rate": 16000,
      "rms_db": -24.7,
      "snr_db_estimate": 14.3,
      "rt60_s_estimate": 0.28,
      "spectral_centroid_khz": 2.1,
      "environment": {
        "value": "quiet_indoor",
        "confidence": 0.8,
        "method": "snr_rt60_heuristic"
      },
      "noise_level": {
        "value": "moderate",
        "confidence": 0.7,
        "method": "snr_heuristic"
      },
      "device_estimate": {
        "value": "phone_mic",
        "confidence": 0.65,
        "method": "sample_rate_spectral_centroid_heuristic"
      }
    },
    "language": {
      "primary_language": "hi-Latn",
      "confidence": 0.81,
      "dominant_language": "hi-Latn",
      "code_switching": true,
      "switching_frequency": 3.2,
      "switching_score": 0.32,
      "scripts": ["Latin"],
      "language_segments": [
        { "start": 0.0, "end": 4.1, "language": "hi-Latn", "confidence": 0.74 }
      ],
      "method": "heuristic"
    },
    "speakers": {
      "SPEAKER_00": {
        "total_speaking_time_s": 32.4,
        "word_count": 102,
        "wpm": 188.9,
        "pause_rate": 0.12,
        "filler_ratio": 0.04,
        "speaking_style": {
          "value": "rapid",
          "confidence": 0.8,
          "method": "wpm_pause_filler_heuristic"
        },
        "formality": {
          "value": "unknown",
          "confidence": 0.3,
          "method": "filler_pause_heuristic"
        },
        "accent": {
          "value": "indian_subcontinent",
          "confidence": 0.65,
          "method": "language_script_proxy"
        },
        "turn_count": 6
      },
      "SPEAKER_01": { "...": "..." }
    },
    "conversation": {
      "turn_count": 12,
      "speaker_count": 2,
      "avg_turn_length_s": 3.8,
      "total_speech_time_s": 46.1,
      "topic_keywords": ["refund", "order id", "delivery", "support"],
      "intents": ["support_inquiry", "complaint", "question"]
    }
  },
  "transcript": {
    "raw": "haan toh bhai, mera order abhi tak nahi aaya, uh kya issue hai?",
    "normalised": "haan toh bhai, mera order abhi tak nahi aaya, uh kya issue hai?",
    "language": "hi",
    "language_probability": 0.82,
    "duration_s": 60.0,
    "segments": [ { "start": 0.0, "end": 4.1, "text": "...", "words": [ ... ] } ]
  },
  "speaker_segmentation": [
    { "start": 0.0, "end": 4.1, "speaker": "SPEAKER_00", "duration": 4.1 },
    { "start": 4.1, "end": 7.6, "speaker": "SPEAKER_01", "duration": 3.5 }
  ],
  "monologue_sample": {
    "speaker": "SPEAKER_00",
    "start": 12.4,
    "end": 26.1,
    "duration_s": 13.7,
    "transcript": "haan toh bhai matlab pichhle hafte...",
    "in_range": true
  },
  "validation": {
    "passed": true,
    "issue_count": 0,
    "issues": [],
    "checks": {
      "audio_quality": { "passed": true },
      "diarisation": {
        "requested_backend": "speaker_vad",
        "effective_backend": "speaker_vad"
      }
    }
  },
  "processing": {
    "offline_mode": true,
    "random_seed": 0,
    "asr": { "backend": "faster-whisper", "model_size": "small" }
  },
  "artifacts": {
    "mono_wav": "/abs/dataset/audio/mono/conversation_0001/mixed.wav",
    "annotation": "/abs/dataset/annotations/conversation_0001.json"
  }
}
```

---

## Testing

```bash
# 47-test pytest suite covering every module (no ASR weights required)
python -m pytest pipeline/tests/ -q
```

---

## Performance notes

- Designed for CPU with 8-16 GB RAM. `faster-whisper small` with
  `int8` quantisation uses roughly 1.5 GB peak.
- Real-time factor on a typical 4-core laptop CPU is around 0.4-0.8x
  (i.e. ~1 minute of audio processed in 25-50 s).
- For longer recordings, increase `--min-speakers` only if you know
  there are multiple speakers - the auto `k` estimator already
  silhouettes its way to 1 for monologues.

---

## Troubleshooting

- **`mp3` files fail to load** - install `ffmpeg`.
- **Missing Whisper model in offline mode** - run `python download_models.py
  --model-dir ./models --whisper-size small`, or pass `--allow-model-downloads`
  explicitly for non-production experimentation.
- **Diarisation splits a single speaker into multiple clusters** -
  this is usually a symptom of noisy segments; try `--denoise` or
  force `--max-speakers 1` if you already know the recording is a
  monologue.

---

## License

Pipeline code: your choice. Third-party models retain their original
licenses (Whisper - MIT, fasttext lid.176 - CC-BY-SA 3.0).
