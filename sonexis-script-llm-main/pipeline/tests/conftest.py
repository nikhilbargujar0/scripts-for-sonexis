"""Shared fixtures for the pipeline test suite."""
from __future__ import annotations

import math

import numpy as np
import pytest


SR = 16_000


def _voiced(freq: float, dur: float, sr: int = SR, noise: float = 0.0) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = 0.5 * np.sin(2 * math.pi * freq * t)
    sig += 0.3 * np.sin(2 * math.pi * (freq * 2.2) * t)
    sig += 0.2 * np.sin(2 * math.pi * (freq * 3.7) * t)
    env = 0.5 + 0.5 * np.sin(2 * math.pi * 4.0 * t)
    sig = sig * env
    if noise > 0:
        rng = np.random.default_rng(0)
        sig = sig + noise * rng.standard_normal(sig.size)
    return sig.astype(np.float32)


def _silence(dur: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * dur), dtype=np.float32)


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def two_speaker_wav() -> np.ndarray:
    """11 s fake 2-speaker conversation at 16 kHz."""
    parts = [
        _voiced(120, 1.8),    # Speaker A
        _silence(0.3),
        _voiced(220, 2.2),    # Speaker B
        _silence(0.3),
        _voiced(115, 1.4),    # A
        _silence(0.4),
        _voiced(210, 1.6),    # B
        _silence(0.2),
    ]
    wav = np.concatenate(parts)
    # Small noise floor.
    rng = np.random.default_rng(0)
    return (wav + 0.005 * rng.standard_normal(wav.size).astype(np.float32)).astype(
        np.float32
    )


@pytest.fixture
def silent_wav() -> np.ndarray:
    return _silence(3.0)


@pytest.fixture
def noisy_wav() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(int(SR * 2.0))).astype(np.float32)
