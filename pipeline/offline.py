"""offline.py

Guards for --offline_mode.

In offline mode every model must already exist on disk. Any module
that tries to load a model calls require_model_file() / require_model_dir()
at startup. If the path is missing we raise OfflineModeError immediately
so the user gets a clear message rather than a silent download or a
cryptic import error halfway through a batch.

Directory layout expected under --model-dir (default: ./models):

    models/
      whisper/<model-size>/            # faster-whisper CTranslate2 format
      diarisation/pyannote/            # optional pyannote checkpoint
      fasttext/lid.176.ftz             # lid.176 fasttext model

Run  python download_models.py  to populate this tree.
"""
from __future__ import annotations

import os


class OfflineModeError(RuntimeError):
    """Raised when a required model file/dir is missing in offline mode."""


def require_model_file(path: str, label: str, download_hint: str = "") -> None:
    """Raise OfflineModeError if *path* does not point to an existing file."""
    if not os.path.isfile(path):
        hint = f"\n  Run: python download_models.py  or  {download_hint}" if download_hint else \
               "\n  Run: python download_models.py"
        raise OfflineModeError(
            f"[offline_mode] Missing model file for '{label}': {path!r}{hint}"
        )


def require_model_dir(path: str, label: str, download_hint: str = "") -> None:
    """Raise OfflineModeError if *path* does not point to an existing directory."""
    if not os.path.isdir(path):
        hint = f"\n  Run: python download_models.py  or  {download_hint}" if download_hint else \
               "\n  Run: python download_models.py"
        raise OfflineModeError(
            f"[offline_mode] Missing model directory for '{label}': {path!r}{hint}"
        )


def default_model_dir() -> str:
    """Return the default models root, respecting SONEXIS_MODEL_DIR env var."""
    return os.environ.get("SONEXIS_MODEL_DIR", os.path.join(os.getcwd(), "models"))


def whisper_local_path(model_dir: str, model_size: str) -> str:
    if os.path.basename(os.path.normpath(model_dir)) == model_size:
        return model_dir
    if os.path.basename(os.path.normpath(model_dir)) == "whisper":
        return os.path.join(model_dir, model_size)
    return os.path.join(model_dir, "whisper", model_size)


def fasttext_local_path(model_dir: str) -> str:
    return os.path.join(model_dir, "fasttext", "lid.176.ftz")


def pyannote_local_path(model_dir: str) -> str:
    return os.path.join(model_dir, "diarisation", "pyannote")
