"""Conservative dataset-purpose tagging helpers."""
from __future__ import annotations

from typing import Dict, List


def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def infer_dataset_purpose(record: Dict) -> Dict:
    """Infer downstream dataset suitability without overclaiming.

    Conversational Sonexis data is usually useful for ASR/STT, diarisation,
    and interaction modelling. TTS remains opt-in only and stays excluded
    unless the record clearly advertises that readiness elsewhere.
    """
    transcript = record.get("transcript") or {}
    turns = record.get("speaker_segmentation") or []
    metadata = record.get("metadata") or {}
    interaction = metadata.get("interaction") or {}
    audio = metadata.get("audio") or {}
    validation = record.get("validation") or {}
    input_mode = str(record.get("input_mode") or "mono")

    transcript_available = bool(str(transcript.get("raw") or "").strip())
    timestamps_available = bool(transcript.get("segments"))
    turns_available = bool(turns)
    speaker_count = len(metadata.get("speakers") or {})
    overlap_duration = float(
        interaction.get("overlap_duration")
        or interaction.get("overlap_duration_s")
        or 0.0
    )

    primary: List[str] = []
    secondary: List[str] = []
    not_recommended_for: List[str] = ["tts"]

    if transcript_available:
        primary.append("stt")
    if turns_available:
        primary.append("diarisation")
    if transcript_available and turns_available and (
        speaker_count >= 2 or overlap_duration > 0.0 or input_mode in {"speaker_pair", "stereo"}
    ):
        primary.append("conversation_modelling")

    if transcript_available and timestamps_available:
        secondary.append("evaluation")
    if speaker_count >= 2 and input_mode in {"speaker_pair", "stereo"}:
        secondary.append("speaker_separation")

    # Only remove the TTS warning when the pipeline has explicit evidence.
    tts_ready = bool(record.get("tts_ready"))
    clean_audio = bool(validation.get("passed")) and float(
        (audio.get("snr_db_estimate") or 0.0)
    ) >= 20.0
    stable_turns = overlap_duration <= 0.1 and turns_available
    if tts_ready and clean_audio and stable_turns and timestamps_available:
        not_recommended_for = []
        secondary.append("tts")

    return {
        "primary": _ordered_unique(primary),
        "secondary": _ordered_unique(secondary),
        "not_recommended_for": _ordered_unique(not_recommended_for),
    }
