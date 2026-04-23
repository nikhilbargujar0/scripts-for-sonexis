"""Compatibility CLI wrapper.

Heavy processing lives in ``pipeline.processors`` and orchestration lives in
``pipeline.runner``. This module stays only to preserve ``python -m pipeline.main``.
"""
from __future__ import annotations

import sys

from scripts.main import main as run


def main(argv=None) -> int:
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
