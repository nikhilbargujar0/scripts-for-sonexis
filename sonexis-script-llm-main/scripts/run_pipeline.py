"""scripts/run_pipeline.py

CLI entry point using the refactored runner + config pattern.

This script is the recommended way to invoke the pipeline from the command
line.  It is equivalent to the legacy ``python -m pipeline.main`` path but
uses the new ``load_config`` / ``run_pipeline`` API so that programmatic
users and the CLI share exactly the same code path.

Usage
-----
::

    python scripts/run_pipeline.py --input ./audio --output ./dataset
    python scripts/run_pipeline.py --input ./audio --output ./dataset \\
        --model-dir ./models --model-size small

    # Offline mode with explicit model directory:
    python scripts/run_pipeline.py \\
        --input ./audio --output ./dataset \\
        --model-dir ./models --offline-mode true

    # Fast batch run on an 8-core machine:
    python scripts/run_pipeline.py \\
        --input ./audio --output ./dataset \\
        --beam-size 1 --asr-batched --num-workers 4 --skip-sha1

    # Load settings from a YAML file:
    python scripts/run_pipeline.py --config pipeline_config.yaml

The ``--config`` flag accepts a YAML file whose keys map directly to
:class:`pipeline.config.PipelineConfig` fields.  CLI flags always override
YAML values when both are present.

YAML example (``pipeline_config.yaml``)::

    input_dir: ./audio
    output_dir: ./dataset
    model_size: small
    offline_mode: true
    model_dir: ./models
    num_workers: 4
    beam_size: 1
"""
from __future__ import annotations

import sys
import os

# Allow running from both the project root and the scripts/ directory.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_scripts_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pipeline.config import load_config  # noqa: E402
from pipeline.runner import run_pipeline  # noqa: E402


def main(argv=None) -> int:
    config = load_config(argv=argv)
    return run_pipeline(config)


if __name__ == "__main__":
    sys.exit(main())
