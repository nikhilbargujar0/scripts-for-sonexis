# Structure

- Entrypoints: `main.py`, `scripts/main.py`, `pipeline/main.py`.
- Configuration: `pipeline/config.py`.
- Orchestration: `pipeline/runner.py`.
- Persistence: `pipeline/dataset_writer.py`, `pipeline/batch_writer.py`, `pipeline/exporters/`.
- Audio IO and transforms: `pipeline/audio_loader.py`, `pipeline/steps/audio_processing.py`, `pipeline/preprocessing.py`.
- ASR and language: `pipeline/transcription.py`, `pipeline/language_detection.py`, `pipeline/code_switch.py`, `pipeline/roman_indic_classifier.py`.
- Quality and validation: `pipeline/quality_checker.py`, `pipeline/quality_tier.py`, `pipeline/validation.py`, `pipeline/schema_validator.py`, `pipeline/steps/validation.py`.
- Tests are unittest-style and runnable with `python3 -m unittest discover -s tests -q`.

