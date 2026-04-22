"""Tests for monologue_extractor."""
from pipeline.diarisation import SpeakerTurn
from pipeline.monologue_extractor import (
    Monologue,
    MonologueConfig,
    extract_monologue,
)
from pipeline.transcription import (
    Transcript,
    TranscriptSegment,
    Word,
)


def _make_transcript_with_dense_words(speaker_turns):
    """Build one contiguous segment + dense word timestamps covering each turn."""
    all_words = []
    for (t_start, t_end, text) in speaker_turns:
        n = max(1, int((t_end - t_start) * 3))  # 3 wps fake rate
        dur = (t_end - t_start) / n
        for i in range(n):
            ws = t_start + i * dur
            we = ws + dur * 0.8
            all_words.append(Word(text=f"w{i}", start=ws, end=we, probability=0.9))
    segs = [TranscriptSegment(
        start=speaker_turns[0][0],
        end=speaker_turns[-1][1],
        text=" ".join(t[2] for t in speaker_turns),
        language="en",
        avg_logprob=-0.2, compression_ratio=1.5, no_speech_prob=0.1,
        rms_db=-22.0, quality_score=0.8, words=all_words,
    )]
    return Transcript(language="en", language_probability=0.9,
                      duration=speaker_turns[-1][1], segments=segs)


def test_monologue_picks_longest_in_range():
    turns = [
        SpeakerTurn(0.0, 3.0, "SPEAKER_00"),   # too short
        SpeakerTurn(3.0, 25.0, "SPEAKER_01"),  # 22s - sweet spot
        SpeakerTurn(25.0, 28.0, "SPEAKER_00"), # short
    ]
    transcript = _make_transcript_with_dense_words(
        [(0.0, 3.0, "hi"), (3.0, 25.0, "long speech"), (25.0, 28.0, "bye")]
    )
    mono = extract_monologue(transcript, turns)
    assert mono is not None
    assert mono.speaker == "SPEAKER_01"
    assert 10.0 <= mono.duration() <= 30.0
    assert mono.in_range is True


def test_monologue_trims_over_long_span():
    turns = [SpeakerTurn(0.0, 60.0, "SPEAKER_00")]
    transcript = _make_transcript_with_dense_words(
        [(0.0, 60.0, "super long monologue that is way over thirty seconds")]
    )
    mono = extract_monologue(transcript, turns, MonologueConfig())
    assert mono is not None
    assert mono.duration() <= 30.0 + 0.5
    assert mono.speaker == "SPEAKER_00"


def test_monologue_returns_none_for_empty_transcript():
    empty_transcript = Transcript(language="en", language_probability=0.0,
                                  duration=0.0, segments=[])
    assert extract_monologue(empty_transcript, []) is None


def test_monologue_tolerates_missing_turns():
    transcript = _make_transcript_with_dense_words([(0.0, 15.0, "solo speech")])
    mono = extract_monologue(transcript, [])
    assert mono is not None
    assert mono.speaker == "SPEAKER_00"
    assert mono.in_range is True
