"""Tests for language_detection."""
from pipeline.language_detection import (
    FastTextLID,
    detect_language,
)


def test_detects_pure_english():
    r = detect_language("Hello there, how are you doing today?", [])
    assert r.primary_language == "en"
    assert r.scripts == ["Latin"]
    assert r.code_switching is False


def test_detects_devanagari_hindi():
    text = "नमस्ते, आप कैसे हैं? मैं ठीक हूँ।"
    r = detect_language(text, [])
    assert r.primary_language == "hi"
    assert "Devanagari" in r.scripts


def test_detects_romanised_hinglish():
    text = "haan bhai main theek hoon, tum batao kya chal raha hai"
    r = detect_language(text, segments_text=[text])
    assert r.primary_language in {"hi-Latn", "hi"}
    assert r.scripts == ["Latin"]


def test_detects_code_switching_via_scripts():
    # Mixed script -> code switching.
    r = detect_language("Hello नमस्ते everyone", ["Hello", "नमस्ते everyone"])
    assert r.code_switching is True


def test_detects_code_switching_via_per_segment():
    # Latin-only but the per-segment probe splits en vs hi-Latn.
    segments = [
        "Welcome to the show, how are you today",
        "haan yaar main ekdam theek hoon aur tum batao",
    ]
    r = detect_language(" ".join(segments), segments)
    assert r.code_switching is True


def test_detects_gurmukhi():
    r = detect_language("ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀ ਕਿਵੇਂ ਹੋ?", [])
    assert r.primary_language == "pa"
    assert "Gurmukhi" in r.scripts


def test_fasttext_gracefully_unavailable(tmp_path):
    lid = FastTextLID(path=str(tmp_path / "missing.ftz"))
    assert lid.available() is False
    assert lid.predict("hello there") == ("und", 0.0)


def test_empty_text_returns_default():
    r = detect_language("", [])
    assert r.primary_language in {"en", "und"}
