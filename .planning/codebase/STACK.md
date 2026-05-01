# Stack

- Python package for offline conversational-audio dataset generation.
- Minimum Python version: 3.10 (`pyproject.toml`).
- Packaging: setuptools with console script `sonexis-pipeline = scripts.main:main`.
- Core audio and ML dependencies: faster-whisper, librosa, soundfile, pydub, noisereduce, webrtcvad-wheels, scikit-learn, numpy, scipy, fasttext-wheel.
- Schema enforcement: jsonschema.
- Optional exports: pandas and pyarrow for parquet.
- Optional diarisation: pyannote.audio and torch.

