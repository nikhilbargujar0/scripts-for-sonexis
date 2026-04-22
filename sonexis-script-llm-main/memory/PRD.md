# PRD - Sonexis-style Offline Conversational-Audio Pipeline

## Original problem statement

Build a production-grade Python pipeline that processes raw conversational
audio into a structured JSON dataset. Must run fully offline - no paid APIs,
no token-based services, no external calls. Only open-source, locally-runnable
models. Targets Indian-English / Hindi / Hinglish / Punjabi / Marwadi
conversational data, preserving disfluencies, interruptions and code-switching.

## User personas

- **ML engineer** running batch jobs on a folder of raw recordings and feeding
  the JSON into downstream training / evaluation pipelines.
- **Dataset curator** inspecting per-file metadata (language, speakers,
  intents, monologue samples) to triage and tag content.

## User choices (from ask_human)

- Delivery: pure Python CLI package (no web UI, no FastAPI)
- Diarisation: skip `pyannote.audio`, use MFCC + KMeans fallback only
- ASR model: `faster-whisper small` (int8 on CPU)
- Testing: full-pipeline E2E with user-supplied audio
- Audio asset: `Guest.wav` (30:08 min, 48 kHz mono Hindi conversation)

## Architecture

```
audio_loader -> preprocessing -> vad -> diarisation -> transcription
                                             |
                        language_detection <-+
                                             |
                        metadata_extraction <-
                                             |
                        monologue_extractor <-
                                             |
                                 output_formatter -> JSON
```

All modules under `/app/pipeline/`. CLI entry: `python -m pipeline.main`.

## Tech stack (100% local)

| Stage | Library |
| ----- | ------- |
| Audio decode | librosa + soundfile (pydub/ffmpeg fallback) |
| VAD | webrtcvad-wheels (silero-vad optional) |
| Diarisation | scikit-learn KMeans on MFCC+delta embeddings |
| ASR | faster-whisper `small`, int8, CPU |
| Language ID | script heuristic + romanised-Indic lexicon + optional fasttext `lid.176` |
| Topic | scikit-learn TF-IDF top-k |
| Intent | regex rules (bilingual EN+HI) |

## Core requirements (static)

1. Fully offline, deterministic, CPU-friendly (8-16 GB RAM).
2. Modular: one file per concern under the mandated module names.
3. Preserve disfluencies / fillers; never over-clean the transcript.
4. Batch process a folder of .wav/.mp3/.flac/.m4a/.ogg files.
5. JSON output per file with transcript, speaker segmentation, multi-tier
   metadata, and a 10-30 s monologue sample.

## What's implemented (2026-02)

Date: 2026-02-14

- [x] `audio_loader.py` - librosa decode, pydub fallback, folder iterator
- [x] `preprocessing.py` - DC remove, peak normalise, optional mild denoise
- [x] `vad.py` - webrtcvad primary with silero-vad optional, merge+pad
- [x] `diarisation.py` - MFCC+delta windows, KMeans with silhouette k-estimation,
  non-overlapping turn resolution
- [x] `transcription.py` - faster-whisper (small/int8/CPU), word timestamps,
  temperature fallback ladder (0.0, 0.4, 0.8) to reduce repetition
  hallucinations without stripping fillers, tqdm progress bar,
  per-segment `compression_ratio` / `no_speech_prob` / `rms_db` /
  `quality_score`
- [x] `language_detection.py` - Unicode-script detection (Devanagari,
  Gurmukhi, Latin, Bengali, Tamil, Telugu), romanised-Hindi / Punjabi /
  Marwadi lexicons (with ambiguous-English tokens pruned), optional
  fasttext lid.176 refinement, per-segment code-switching report
- [x] `metadata_extraction.py` -
  - Audio: duration, RMS dB, estimated SNR dB, RT60 proxy (fixed to
    handle long files), spectral centroid, environment + device classes
  - Speaker: WPM, pause rate, filler ratio, speaking style, accent proxy,
    turn count
  - Conversation: turn count, speaker count, avg turn length, TF-IDF
    keywords, rule-based multi-label intent, aggregated quality summary
    (mean / median / duration-weighted / low-quality ratio / high-quality
    ratio / segment count)
- [x] `monologue_extractor.py` - 10-30 s longest coherent single-speaker span
- [x] `output_formatter.py` - stable schema 1.0.0 JSON emitter with sha1
- [x] `main.py` - argparse CLI, batch loop, per-file JSON
- [x] `tests/` - 47-test pytest suite covering every module (no Whisper
  weights required for the test run)
- [x] `requirements.txt` + detailed `README.md` with setup, CLI flags, example
  output, performance notes

## Testing evidence

- **Self-test**: synthetic 2-speaker 11 s file -> all modules execute,
  diarisation correctly returns non-overlapping speaker turns.
- **Real audio E2E**: Guest.wav 30:08 min Hindi ->
  - Whisper language = `hi` (0.82 conf)
  - 2 speakers detected (760 s / 768 s speech time, balanced)
  - 788 turns, avg 1.94 s, 153 transcript segments
  - Monologue 29.4 s in-range
  - Language report flagged code-switching (Devanagari + Latin)
  - Processing time: 1768 s on CPU (0.98x real-time)

## Prioritised backlog (not done)

P1:
- ~~Unit tests for each module~~ **DONE** (61 tests, all passing)
- ~~Progress bar for ASR~~ **DONE** (tqdm on `seg_iter`)

P2:
- ~~Optional pyannote path when user supplies an HF token~~ **DONE**
  (`--diarisation-backend pyannote`, graceful fallback to kmeans)
- ~~Trained romanised-Indic language classifier~~ **DONE**
  (`roman_indic_classifier.py`: TF-IDF char-ngrams + LogisticRegression,
  bundled corpus, pickle cache, integrated into `detect_language`)
- ~~Parquet / JSONL batch output format~~ **DONE**
  (`batch_writer.py` with `--output-format {json,jsonl,parquet}`)
- ~~Setuptools entry point (`sonexis-pipeline` CLI shim)~~ **DONE**
  (`pyproject.toml`, `pip install -e .` installs the CLI)

Future nice-to-haves:
- Accent identification via a proper embedding model (currently a coarse
  proxy label only).
- Per-speaker embeddings persisted to parquet for re-identification
  across recordings.
- Streaming mode that writes JSONL incrementally as each file finishes
  (current JSONL impl already streams per-record; can be generalised).
