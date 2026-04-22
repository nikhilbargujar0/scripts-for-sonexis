"""Tests for metadata_extraction."""
import numpy as np

from pipeline.diarisation import SpeakerTurn
from pipeline.metadata_extraction import (
    extract_audio_metadata,
    extract_conversation_metadata,
    extract_speaker_metadata,
)
from pipeline.transcription import (
    FILLERS,
    Transcript,
    TranscriptSegment,
    Word,
    compute_quality_score,
)


def _make_transcript(segs_spec):
    """segs_spec: list of (start, end, text, [(wstart,wend,word), ...])"""
    segments = []
    for (s, e, txt, words_spec) in segs_spec:
        words = [Word(text=w, start=ws, end=we, probability=0.9)
                 for (ws, we, w) in words_spec]
        q = compute_quality_score(
            avg_logprob=-0.2, compression_ratio=1.5,
            no_speech_prob=0.1, rms_db=-22.0,
        )
        segments.append(TranscriptSegment(
            start=s, end=e, text=txt, language="en",
            avg_logprob=-0.2, compression_ratio=1.5,
            no_speech_prob=0.1, rms_db=-22.0, quality_score=q,
            words=words,
        ))
    total = segments[-1].end if segments else 0.0
    return Transcript(language="en", language_probability=0.9,
                      duration=total, segments=segments)


def test_audio_metadata_basic(two_speaker_wav, sr):
    # Fake VAD segments covering roughly the first and last halves.
    dur = len(two_speaker_wav) / sr
    speech = [(0.0, dur * 0.45), (dur * 0.55, dur)]
    meta = extract_audio_metadata(two_speaker_wav, sr, speech)

    assert meta["sample_rate"] == sr
    assert abs(meta["duration_s"] - dur) < 0.02
    assert -60.0 < meta["rms_db"] < 0.0
    assert 0.0 <= meta["rt60_s_estimate"] <= 3.0
    assert meta["environment"]["value"] in {
        "studio", "quiet_indoor", "reverberant_indoor", "noisy", "unknown",
    }
    assert meta["noise_level"]["value"] in {"low", "moderate", "high", "very_high", "unknown"}
    assert meta["device_estimate"]["value"] in {
        "telephony_narrowband", "telephony_or_lapel", "phone_mic",
        "laptop_or_headset", "studio_mic",
    }


def test_audio_metadata_silence(silent_wav, sr):
    meta = extract_audio_metadata(silent_wav, sr, [])
    assert meta["duration_s"] == 3.0
    # Silent input should collapse to -120 dB floor.
    assert meta["rms_db"] <= -60.0


def test_speaker_metadata_counts_words():
    words = [(0.0, 0.3, "hello"), (0.35, 0.7, "world"), (0.75, 1.0, "today")]
    transcript = _make_transcript([(0.0, 1.0, "hello world today", words)])
    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]
    meta = extract_speaker_metadata(
        transcript, turns, language="en", scripts=["Latin"],
        filler_lexicon=FILLERS,
    )
    assert "SPEAKER_00" in meta
    assert meta["SPEAKER_00"]["word_count"] == 3
    assert meta["SPEAKER_00"]["wpm"] == round(3 / 1.0 * 60.0, 2)
    assert meta["SPEAKER_00"]["turn_count"] == 1


def test_speaker_metadata_filler_detection():
    words = [(0.0, 0.3, "um"), (0.4, 0.7, "matlab"), (0.8, 1.0, "uh"),
             (1.1, 1.3, "hello"), (1.4, 1.6, "world")]
    transcript = _make_transcript([(0.0, 1.6, "um matlab uh hello world", words)])
    turns = [SpeakerTurn(0.0, 1.6, "SPEAKER_00")]
    meta = extract_speaker_metadata(
        transcript, turns, language="hi-Latn", scripts=["Latin"],
        filler_lexicon=FILLERS,
    )
    assert meta["SPEAKER_00"]["filler_ratio"] > 0.5
    assert meta["SPEAKER_00"]["speaking_style"]["value"] == "hesitant"
    assert meta["SPEAKER_00"]["accent"]["value"] == "indian_english_or_romanised_indic"


def test_conversation_metadata_summarises_quality():
    words = [(0.0, 0.3, "this"), (0.4, 0.6, "is"), (0.7, 1.0, "great")]
    transcript = _make_transcript([(0.0, 1.0, "this is great please help", words)])
    turns = [
        SpeakerTurn(0.0, 0.5, "SPEAKER_00"),
        SpeakerTurn(0.5, 1.0, "SPEAKER_01"),
    ]
    meta = extract_conversation_metadata(transcript, turns)
    assert meta["turn_count"] == 2
    assert meta["speaker_count"] == 2
    assert "quality" in meta
    q = meta["quality"]
    for key in ("mean_quality_score", "duration_weighted_quality_score",
                "low_quality_segment_ratio", "high_quality_segment_ratio",
                "segment_count"):
        assert key in q


def test_intent_detection_picks_request_and_question():
    words = [(0.0, 0.2, "please"), (0.3, 0.5, "help"),
             (0.6, 0.8, "what"), (0.9, 1.0, "is")]
    transcript = _make_transcript([
        (0.0, 1.0, "please help, what is going on?", words),
    ])
    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]
    meta = extract_conversation_metadata(transcript, turns)
    assert "request" in meta["intents"]
    assert "question" in meta["intents"]


def test_topic_keywords_are_non_stopwords():
    words = [(0.0, 0.2, "refund"), (0.3, 0.5, "order"),
             (0.6, 0.8, "delivery"), (0.9, 1.0, "support")]
    transcript = _make_transcript([
        (0.0, 1.0, "refund order delivery support help", words),
    ])
    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]
    meta = extract_conversation_metadata(transcript, turns)
    # At least one of our domain nouns should bubble up.
    assert any(k in meta["topic_keywords"]
               for k in ("refund", "order", "delivery", "support"))
