#!/usr/bin/env python3
"""CLI shim for the Sonexis conversational-audio dataset pipeline."""
from pipeline.main import run


if __name__ == "__main__":
    raise SystemExit(run())
