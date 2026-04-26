"""CSV/TSV flat segment manifests."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

from .common import segment_rows, session_id


FIELDS = [
    "utt_id",
    "recording_id",
    "segment_id",
    "audio_filepath",
    "start",
    "end",
    "duration",
    "speaker_id",
    "language",
    "text",
    "text_normalized",
    "sample_rate",
    "snr_db",
    "snr_band",
    "confidence",
    "confidence_band",
    "quality_tier",
    "matrix_language",
    "cs_density",
    "flagged_for_review",
]


def _write(path: Path, rows, delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_csv_manifest(record: Dict, output_root: str) -> Dict[str, str]:
    rows = segment_rows(record)
    base = Path(output_root) / "exports" / "csv"
    sid = session_id(record)
    csv_path = base / f"{sid}.csv"
    tsv_path = base / f"{sid}.tsv"
    _write(csv_path, rows, ",")
    _write(tsv_path, rows, "\t")
    return {"csv_manifest": str(csv_path.resolve()), "tsv_manifest": str(tsv_path.resolve())}
