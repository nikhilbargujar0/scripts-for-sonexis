"""Tests for user-provided metadata handling."""
from pipeline.main import _apply_user_metadata


def test_apply_user_metadata_prefers_label_over_global():
    speaker_meta = {
        "SPEAKER_00": {"wpm": 100.0},
        "SPEAKER_01": {"wpm": 120.0},
    }
    user_metadata = {
        "*": {
            "region": {"value": "unknown", "confidence": 0.0, "source": "interactive_prompt"},
        },
        "Host": {
            "region": {"value": "Delhi", "confidence": 1.0, "source": "metadata_file"},
            "gender": {"value": "female", "confidence": 1.0, "source": "metadata_file"},
        },
    }
    enriched = _apply_user_metadata(
        speaker_meta,
        user_metadata,
        speaker_map={"SPEAKER_00": "Host", "SPEAKER_01": "Guest"},
    )
    assert enriched["SPEAKER_00"]["provided_metadata"]["region"]["value"] == "Delhi"
    assert enriched["SPEAKER_00"]["provided_metadata"]["gender"]["value"] == "female"
    assert enriched["SPEAKER_01"]["provided_metadata"]["region"]["value"] == "unknown"
