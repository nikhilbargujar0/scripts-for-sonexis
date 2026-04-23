"""Purpose-specific dataset product exporters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

from .diarisation_product import build_diarisation_product
from .evaluation_gold_product import build_evaluation_gold_product
from .stt_product import build_stt_product
from .tts_export_product import build_tts_export_product


PRODUCT_BUILDERS: Dict[str, Callable[[Dict], Dict]] = {
    "stt": build_stt_product,
    "tts_export": build_tts_export_product,
    "diarisation": build_diarisation_product,
    "evaluation_gold": build_evaluation_gold_product,
}


def export_products(record: Dict, output_root: str) -> Dict[str, str]:
    written: Dict[str, str] = {}
    for product in record.get("dataset_products", []) or []:
        builder = PRODUCT_BUILDERS.get(product)
        if builder is None:
            continue
        payload = builder(record)
        path = Path(output_root) / "products" / product / f"{record.get('session_name', 'session')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written[f"product_{product}"] = str(path.resolve())
    return written
