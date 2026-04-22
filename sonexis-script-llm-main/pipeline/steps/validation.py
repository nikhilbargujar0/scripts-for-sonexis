"""pipeline/steps/validation.py

Re-exports from pipeline.validation plus ``write_validation_report``.

``write_validation_report`` serialises the validation dict to
``<output_dir>/validation_report.json`` so that downstream curators can
audit pipeline runs without opening individual dataset records.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

from pipeline.validation import build_validation_report  # noqa: F401

log = logging.getLogger(__name__)

__all__ = [
    "build_validation_report",
    "write_validation_report",
]


def write_validation_report(
    report: Dict,
    output_dir: str,
    session_name: str = "session",
    filename: Optional[str] = None,
) -> str:
    """Persist a validation report dict to *output_dir*.

    The file is written as pretty-printed JSON.  The default filename is
    ``<session_name>_validation_report.json``.

    Parameters
    ----------
    report :
        Dict returned by :func:`build_validation_report`.
    output_dir :
        Directory where the file will be written.  Created if absent.
    session_name :
        Used to construct the default filename.
    filename :
        Override the filename (must end with ``.json``).

    Returns
    -------
    str
        Absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = filename or f"{session_name}_validation_report.json"
    path = os.path.join(output_dir, fname)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    log.debug("validation report written: %s", path)
    return os.path.abspath(path)
