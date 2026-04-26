"""Multi-format exporters for canonical Sonexis records."""
from __future__ import annotations

from typing import Dict

from .csv_manifest import export_csv_manifest
from .hf_dataset import export_hf_dataset
from .kaldi import export_kaldi_bundle


def export_phase4_formats(record: Dict, output_root: str) -> Dict[str, str]:
    written: Dict[str, str] = {}
    for exporter in (export_csv_manifest, export_hf_dataset, export_kaldi_bundle):
        written.update(exporter(record, output_root))
    return written
