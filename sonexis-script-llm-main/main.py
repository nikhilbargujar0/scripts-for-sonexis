#!/usr/bin/env python3
"""CLI shim for the Sonexis conversational-audio dataset pipeline.

Delegates to the production entry points:

    load_config() — parses CLI arguments and optional YAML config file
    run_pipeline() — runs the pipeline with a populated PipelineConfig

Legacy path (pipeline.main.run) is still available for backwards
compatibility but this shim now uses the recommended API.
"""
from pipeline.config import load_config
from pipeline.runner import run_pipeline


def run(argv=None):
    config = load_config(argv=argv)
    return run_pipeline(config)


if __name__ == "__main__":
    import sys
    raise SystemExit(run())
