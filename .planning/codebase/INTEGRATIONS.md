# Integrations

- Local Whisper/faster-whisper models are loaded from configured model directories.
- FastText language ID can be loaded locally through the model setup flow.
- Premium ASR adapters exist for Deepgram and Google STT v2; Azure adapter is present but intentionally not implemented.
- Hugging Face is used by `download_models.py` for model acquisition, not by the default offline processing path.
- Phase 4 exports target common speech dataset consumers: Hugging Face dataset JSONL, CSV/TSV manifests, and Kaldi artifacts.

