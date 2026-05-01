# Architecture

- `main.py` is a thin root CLI shim.
- `scripts/main.py` owns argument parsing and `PipelineConfig` assembly.
- `pipeline/runner.py` orchestrates input resolution, model loading, record processing, validation, batch writing, and downloadable zip creation.
- `pipeline/steps/` holds step-level audio, diarisation, transcription, metadata, validation, and alignment functions.
- `pipeline/processors/` contains mono and speaker-pair record pipelines.
- `pipeline/premium/` routes premium accuracy behavior across ASR, alignment, quality, consensus, and review.
- `pipeline/exporters/` produces downstream dataset formats.
- `schema/dataset_schema.json` is the record contract, covered by schema-lockdown tests.

