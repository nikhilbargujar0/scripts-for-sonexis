"""Kaldi, RTTM, and CTM exports."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .common import audio_path, c_sort, segment_rows, session_id


def _write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line if line.endswith("\n") else line + "\n" for line in c_sort(lines)), encoding="utf-8")


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").replace("\t", " ").split())


def export_kaldi_bundle(record: Dict, output_root: str) -> Dict[str, str]:
    sid = session_id(record)
    rows = segment_rows(record)
    base = Path(output_root) / "exports" / "kaldi" / sid
    wav = audio_path(record)
    wav_scp = [f"{sid} {wav}"]
    segments: List[str] = []
    text_lines: List[str] = []
    utt2spk: List[str] = []
    utt2dur: List[str] = []
    utt2lang: List[str] = []
    ctm: List[str] = []

    spk2utts: Dict[str, List[str]] = {}
    for row in rows:
        utt = row["utt_id"]
        spk = row["speaker_ascii"]
        start = float(row.get("start") or 0.0)
        end = float(row.get("end") or start)
        dur = max(0.0, end - start)
        text = _clean_text(str(row.get("text") or ""))
        segments.append(f"{utt} {sid} {start:.3f} {end:.3f}")
        if text:
            text_lines.append(f"{utt} {text}")
        utt2spk.append(f"{utt} {spk}")
        utt2dur.append(f"{utt} {dur:.3f}")
        utt2lang.append(f"{utt} {row.get('language') or 'und'}")
        spk2utts.setdefault(spk, []).append(utt)
        words = row.get("words") or []
        for word in words:
            w_start = float(word.get("start") or start)
            w_end = float(word.get("end") or w_start)
            w_dur = max(0.0, w_end - w_start)
            token = _clean_text(str(word.get("text") or ""))
            if token:
                conf = float(word.get("probability") or 1.0)
                ctm.append(f"{sid} 1 {w_start:.3f} {w_dur:.3f} {token} {conf:.3f}")

    spk2utt = [f"{spk} {' '.join(c_sort(utts))}" for spk, utts in spk2utts.items()]
    rttm = [
        f"SPEAKER {sid} 1 {float(turn.get('start') or 0.0):.3f} "
        f"{float(turn.get('duration') or (float(turn.get('end') or 0.0) - float(turn.get('start') or 0.0))):.3f} "
        f"<NA> <NA> {turn.get('speaker') or 'SPEAKER_00'} <NA> <NA>"
        for turn in record.get("speaker_segmentation", []) or []
    ]

    files = {
        "kaldi_wav_scp": base / "wav.scp",
        "kaldi_segments": base / "segments",
        "kaldi_text": base / "text",
        "kaldi_utt2spk": base / "utt2spk",
        "kaldi_spk2utt": base / "spk2utt",
        "kaldi_utt2dur": base / "utt2dur",
        "kaldi_utt2lang": base / "utt2lang",
        "rttm": base / f"{sid}.rttm",
        "ctm": base / f"{sid}.ctm",
    }
    for key, path in files.items():
        data = {
            "kaldi_wav_scp": wav_scp,
            "kaldi_segments": segments,
            "kaldi_text": text_lines,
            "kaldi_utt2spk": utt2spk,
            "kaldi_spk2utt": spk2utt,
            "kaldi_utt2dur": utt2dur,
            "kaldi_utt2lang": utt2lang,
            "rttm": rttm,
            "ctm": ctm,
        }[key]
        _write_lines(path, data)
    return {key: str(path.resolve()) for key, path in files.items()}
