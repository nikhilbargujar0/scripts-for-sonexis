"""Tests for diarisation pyannote path (token not required - we only
verify the error-handling contract; the primary KMeans path stays the
main behavioural test suite)."""
import pytest

from pipeline.diarisation import diarise_pyannote


def test_pyannote_raises_without_token(monkeypatch, two_speaker_wav, sr):
    # Ensure no HF token is present in the environment for this test.
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        # pyannote not installed -> first branch of RuntimeError.
        with pytest.raises(RuntimeError) as excinfo:
            diarise_pyannote(two_speaker_wav, sr)
        assert "pyannote.audio is not installed" in str(excinfo.value)
        return

    # pyannote installed but no token -> second branch.
    with pytest.raises(RuntimeError) as excinfo:
        diarise_pyannote(two_speaker_wav, sr, hf_token=None)
    assert "HuggingFace access token" in str(excinfo.value)
