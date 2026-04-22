"""Tests for output_formatter."""
import json
import os

from pipeline.diarisation import SpeakerTurn
from pipeline.language_detection import LanguageReport
from pipeline.monologue_extractor import Monologue
from pipeline.output_formatter import SCHEMA_VERSION, build_record
from pipeline.transcription import Transcript, TranscriptSegment, Word


def _dummy_transcript():
    seg = TranscriptSegment(
        start=0.0, end=1.0, text="hello", language="en",
        avg_logprob=-0.3, compression_ratio=1.5, no_speech_prob=0.1,
        rms_db=-20.0, quality_score=0.82,
        words=[Word(text="hello", start=0.0, end=1.0, probability=0.9)],
    )
    return Transcript(language="en", language_probability=0.9, duration=1.0,
                      segments=[seg])


def test_build_record_has_all_top_level_keys(tmp_path):
    path = tmp_path / "x.wav"
    path.write_bytes(b"RIFF----WAVEfmt ")
    transcript = _dummy_transcript()
    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]
    lang = LanguageReport(primary_language="en", confidence=0.9,
                          code_switching=False, scripts=["Latin"],
                          per_segment=[], method="heuristic")
    mono = Monologue(speaker="SPEAKER_00", start=0.0, end=1.0,
                     transcript="hello", words=[], in_range=False)
    rec = build_record(
        audio_path=str(path), transcript=transcript, turns=turns,
        language=lang,
        audio_meta={"duration_s": 1.0, "sample_rate": 16000},
        speaker_meta={"SPEAKER_00": {"wpm": 60.0}},
        conversation_meta={"turn_count": 1},
        monologue=mono,
    )
    assert rec["schema_version"] == SCHEMA_VERSION
    for key in ("file", "metadata", "transcript", "speaker_segmentation",
                "monologue_sample", "generated_at"):
        assert key in rec
    assert rec["file"]["name"] == "x.wav"
    assert rec["file"]["size_bytes"] == os.path.getsize(path)
    assert rec["transcript"]["raw"] == "hello"
    assert rec["transcript"]["segments"][0]["quality_score"] == 0.82


def test_build_record_is_json_serialisable(tmp_path):
    path = tmp_path / "y.wav"
    path.write_bytes(b"RIFF----WAVEfmt ")
    transcript = _dummy_transcript()
    lang = LanguageReport(primary_language="hi", confidence=0.8,
                          code_switching=True, scripts=["Devanagari", "Latin"],
                          per_segment=[], method="heuristic")
    rec = build_record(
        audio_path=str(path), transcript=transcript, turns=[], language=lang,
        audio_meta={}, speaker_meta={}, conversation_meta={}, monologue=None,
    )
    # Round-trip through json to make sure everything is serialisable.
    s = json.dumps(rec, ensure_ascii=False)
    parsed = json.loads(s)
    assert parsed["metadata"]["language"]["code_switching"] is True
    assert parsed["monologue_sample"] is None
