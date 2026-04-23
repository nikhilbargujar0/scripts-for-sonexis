"""metadata_extraction.py

Rule-based + light-ML metadata extraction. Absolutely no LLM calls.

Four tiers of metadata:

* ``audio``           - duration, loudness, snr, reverberation, device class
* ``speaker``         - per-speaker WPM, pause frequency, accent proxy,
                         dominant speaking style
* ``language``        - filled in by ``language_detection.py`` and just
                         echoed here for completeness
* ``conversation``    - turn count, avg turn length, topic keywords,
                         rule-based intent
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .diarisation import SpeakerTurn
from .transcription import Transcript, TranscriptSegment
from .utils.metadata_fields import inferred_field, measured_field

log = logging.getLogger(__name__)

Segment = Tuple[float, float]


# ---------------------------------------------------------------------------
#  Audio-level features
# ---------------------------------------------------------------------------

def _rms_db(wav: np.ndarray) -> float:
    if wav.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    if rms < 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)


def _estimate_snr_db(wav: np.ndarray, speech_segments: List[Segment], sr: int) -> float:
    """Very coarse SNR: speech-frame energy vs. non-speech-frame energy."""
    if not speech_segments:
        return 0.0
    mask = np.zeros(len(wav), dtype=bool)
    for s, e in speech_segments:
        mask[int(s * sr) : int(e * sr)] = True
    speech = wav[mask]
    noise = wav[~mask]
    if speech.size == 0 or noise.size == 0:
        return 0.0
    ps = float(np.mean(speech.astype(np.float64) ** 2)) + 1e-12
    pn = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    return 10.0 * math.log10(ps / pn)


def _reverb_estimate(wav: np.ndarray, sr: int) -> float:
    """Proxy for RT60 using the decay of the short-term RMS envelope.

    We avoid a full autocorrelation over long files (which produces
    nonsense values when the recording spans many minutes) by picking a
    handful of high-energy ~2 s windows and averaging the time each
    envelope takes to fall below 1/e of its peak.
    """
    if wav.size < sr:
        return 0.0
    try:
        import librosa

        frame_length = 1024
        hop_length = 512
        env = librosa.feature.rms(
            y=wav, frame_length=frame_length, hop_length=hop_length
        )[0]
        if env.size == 0 or env.max() < 1e-6:
            return 0.0
        hop_s = hop_length / sr
        window_frames = int(round(2.0 / hop_s))     # ~2 s analysis windows
        if window_frames < 8:
            return 0.0

        # Pick up to 6 high-energy windows spread through the recording.
        n_windows = min(6, max(1, env.size // window_frames))
        step = max(1, env.size // (n_windows + 1))
        estimates: list[float] = []
        for k in range(1, n_windows + 1):
            start = min(env.size - window_frames, k * step)
            chunk = env[start : start + window_frames]
            if chunk.size < 8:
                continue
            peak = float(chunk.max())
            if peak < 1e-6:
                continue
            peak_idx = int(np.argmax(chunk))
            tail = chunk[peak_idx:]
            threshold = peak * (1.0 / math.e)
            below = np.where(tail < threshold)[0]
            if below.size == 0:
                continue
            estimates.append(float(below[0] * hop_s))
        if not estimates:
            return 0.0
        # Use the median to ignore outlier windows (music, claps, etc.).
        return float(np.clip(np.median(estimates), 0.0, 3.0))
    except Exception:
        return 0.0


def _spectral_centroid_khz(wav: np.ndarray, sr: int) -> float:
    try:
        import librosa
        sc = librosa.feature.spectral_centroid(y=wav, sr=sr)
        return float(np.mean(sc)) / 1000.0
    except Exception:
        return 0.0


def _effective_bandwidth_hz(wav: np.ndarray, sr: int) -> Tuple[int, float]:
    try:
        import librosa

        rolloff = librosa.feature.spectral_rolloff(
            y=wav,
            sr=sr,
            roll_percent=0.95,
        )[0]
        if rolloff.size == 0:
            return 0, 0.2
        value = int(round(float(np.median(rolloff))))
        confidence = 0.9 if value > 0 else 0.2
        return value, confidence
    except Exception:
        return 0, 0.2


def _lufs_estimate(wav: np.ndarray) -> Tuple[float, float, str]:
    """Return (value, confidence, method) using a safe local fallback.

    We intentionally avoid claiming BS.1770 compliance when that library is
    not present. The fallback is still useful for relative filtering.
    """
    rms_db = _rms_db(wav)
    if rms_db <= -120.0:
        return -70.0, 0.3, "rms_fallback_estimate"
    return round(rms_db - 0.69, 2), 0.45, "rms_fallback_estimate"


def _classify_environment(snr_db: float, rt60_s: float) -> Tuple[str, float]:
    """Return (environment_label, confidence).

    Confidence reflects how unambiguous the SNR + RT60 signal is.
    Borderline values get lower confidence; unknown fallback at 0.3.
    """
    if snr_db >= 20 and rt60_s < 0.3:
        return "studio", 0.85
    if snr_db >= 15 and rt60_s < 0.4:
        return "quiet_indoor", 0.80
    if snr_db >= 12 and rt60_s < 0.5:
        return "quiet_indoor", 0.65
    if rt60_s >= 0.7:
        return "reverberant_indoor", 0.75
    if rt60_s >= 0.5:
        return "reverberant_indoor", 0.60
    if snr_db < 5:
        return "noisy", 0.70
    if snr_db < 8:
        return "noisy", 0.55
    # Borderline: not enough signal to commit.
    return "unknown", 0.30


def _classify_noise_level(snr_db: float) -> Tuple[str, float]:
    """Return coarse noise level from SNR estimate."""
    if snr_db >= 20:
        return "low", 0.80
    if snr_db >= 12:
        return "moderate", 0.70
    if snr_db >= 5:
        return "high", 0.65
    if snr_db > 0:
        return "very_high", 0.65
    return "unknown", 0.30


def _classify_device(spectral_centroid_khz: float, sr: int) -> Tuple[str, float]:
    """Return (device_label, confidence).

    Heuristic only — frequency rolloff and sample rate constrain the class.
    Confidence is lower when centroid falls in ambiguous boundary zones.
    """
    if sr <= 8000:
        return "telephony_narrowband", 0.90
    if spectral_centroid_khz < 1.1:
        return "telephony_or_lapel", 0.75
    if spectral_centroid_khz < 1.5:
        return "telephony_or_lapel", 0.55
    if spectral_centroid_khz < 2.0:
        return "phone_mic", 0.65
    if spectral_centroid_khz < 2.5:
        return "phone_mic", 0.55
    if spectral_centroid_khz < 3.5:
        return "laptop_or_headset", 0.60
    return "studio_mic", 0.65


def extract_audio_metadata(
    wav: np.ndarray,
    sample_rate: int,
    speech_segments: List[Segment],
    source_info: Optional[Dict] = None,
) -> Dict:
    source_info = dict(source_info or {})
    rms_db = _rms_db(wav)
    snr_db = _estimate_snr_db(wav, speech_segments, sample_rate)
    rt60 = _reverb_estimate(wav, sample_rate)
    centroid = _spectral_centroid_khz(wav, sample_rate)
    bandwidth_hz, bandwidth_conf = _effective_bandwidth_hz(wav, sample_rate)
    lufs_value, lufs_conf, lufs_method = _lufs_estimate(wav)
    env_value, env_conf = _classify_environment(snr_db, rt60)
    noise_value, noise_conf = _classify_noise_level(snr_db)
    device_value, device_conf = _classify_device(centroid, sample_rate)
    return {
        "duration_s": round(len(wav) / sample_rate, 3),
        "sample_rate": int(sample_rate),
        "sample_rate_hz": int(source_info.get("sample_rate_hz") or sample_rate),
        "processed_sample_rate_hz": int(
            source_info.get("processed_sample_rate_hz") or sample_rate
        ),
        "bit_depth": source_info.get("bit_depth"),
        "channels": int(source_info.get("channels") or 1),
        "codec": source_info.get("codec"),
        "container_format": source_info.get("container_format"),
        "rms_db": round(rms_db, 2),
        "snr_db_estimate": round(snr_db, 2),
        "rt60_s_estimate": round(rt60, 3),
        "spectral_centroid_khz": round(centroid, 3),
        "effective_bandwidth_hz": measured_field(
            bandwidth_hz,
            bandwidth_conf,
            method="spectral_rolloff_95",
        ),
        "lufs": measured_field(
            lufs_value,
            lufs_conf,
            method=lufs_method,
        ),
        "environment": {
            **inferred_field(env_value, env_conf),
            "method": "snr_rt60_heuristic",
        },
        "noise_level": {
            **inferred_field(noise_value, noise_conf),
            "method": "snr_heuristic",
        },
        "device_type": {
            **inferred_field(device_value, device_conf),
            "method": "sample_rate_spectral_centroid_heuristic",
        },
        "device_estimate": {
            **inferred_field(device_value, device_conf),
            "method": "sample_rate_spectral_centroid_heuristic",
        },
    }


# ---------------------------------------------------------------------------
#  Speaker-level features
# ---------------------------------------------------------------------------

_WORD_TOKENIZER = re.compile(r"[\w']+", flags=re.UNICODE)


def _words_in(text: str) -> List[str]:
    return _WORD_TOKENIZER.findall(text)


def _assign_words_to_speakers(
    transcript: Transcript, turns: List[SpeakerTurn]
) -> Dict[str, List[dict]]:
    """Return {speaker_id: [{"text","start","end"}]} using turn overlap."""
    out: Dict[str, List[dict]] = defaultdict(list)
    if not turns:
        # Everything falls under SPEAKER_00 when diarisation returned empty.
        for seg in transcript.segments:
            for w in seg.words or []:
                out["SPEAKER_00"].append({"text": w.text, "start": w.start, "end": w.end})
        return out

    def find_speaker(t: float) -> str:
        # Binary search-lite: turns are sorted and non-overlapping after merge.
        for turn in turns:
            if turn.start <= t < turn.end:
                return turn.speaker
        # Fall back to the nearest turn.
        nearest = min(turns, key=lambda tn: min(abs(tn.start - t), abs(tn.end - t)))
        return nearest.speaker

    for seg in transcript.segments:
        if seg.words:
            for w in seg.words:
                mid = (w.start + w.end) / 2.0
                out[find_speaker(mid)].append(
                    {"text": w.text, "start": w.start, "end": w.end}
                )
        else:
            mid = (seg.start + seg.end) / 2.0
            spk = find_speaker(mid)
            for token in _words_in(seg.text):
                out[spk].append({"text": token, "start": seg.start, "end": seg.end})
    return out


def _style_from_metrics(wpm: float, pause_rate: float, filler_ratio: float) -> Tuple[str, float]:
    """Return (speaking_style, confidence).

    Confidence reflects how unambiguous the signal is. Borderline values
    (e.g. moderate WPM + moderate pause_rate) get lower confidence.
    """
    if wpm == 0:
        return "silent", 0.9
    if filler_ratio > 0.12:
        return "hesitant", 0.80
    if filler_ratio > 0.08:
        return "hesitant", 0.65
    if wpm < 80 and pause_rate > 0.40:
        return "deliberate", 0.75
    if wpm < 90 and pause_rate > 0.35:
        return "deliberate", 0.60
    if wpm > 190:
        return "rapid", 0.80
    if wpm > 170:
        return "rapid", 0.65
    if pause_rate < 0.08:
        return "fluent", 0.70
    if pause_rate < 0.1:
        return "fluent", 0.60
    return "conversational", 0.55


def _accent_proxy(language: str, scripts: List[str]) -> Tuple[str, float]:
    """Return (accent_family, confidence).

    Coarse acoustic-family label only — true accent ID needs a trained
    acoustic model. Confidence intentionally conservative; never overclaim.
    """
    if "Devanagari" in scripts or "Gurmukhi" in scripts:
        return "indian_subcontinent", 0.70
    if language in ("hi", "pa", "mwr"):
        return "indian_subcontinent", 0.65
    if language.endswith("-Latn"):
        return "indian_english_or_romanised_indic", 0.55
    if language == "en":
        # Cannot distinguish accent from transcript alone.
        return "unknown", 0.30
    return "unknown", 0.25


def _formality_from_metrics(filler_ratio: float, pause_rate: float) -> Tuple[str, float]:
    """Very conservative formal/informal estimate from delivery signals only."""
    if filler_ratio > 0.08:
        return "informal", 0.55
    if pause_rate > 0.35:
        return "deliberate_or_formal", 0.45
    return "unknown", 0.30


def extract_speaker_metadata(
    transcript: Transcript,
    turns: List[SpeakerTurn],
    language: str,
    scripts: List[str],
    filler_lexicon: set,
    # New optional params for v2 schema fields:
    speaker_labels: Optional[Dict[str, str]] = None,    # {speaker_id: human_label}
    overlap_ratios: Optional[Dict[str, float]] = None,  # from interaction_metadata
    dominance: Optional[Dict[str, float]] = None,       # from interaction_metadata
    total_audio_duration_s: float = 0.0,                # for speech_ratio
) -> Dict[str, Dict]:
    per_speaker_words = _assign_words_to_speakers(transcript, turns)
    per_speaker_turns: Dict[str, List[SpeakerTurn]] = defaultdict(list)
    for t in turns:
        per_speaker_turns[t.speaker].append(t)

    # Ensure all speakers from turns appear even if they produced no ASR words.
    all_speakers = set(per_speaker_words.keys()) | set(per_speaker_turns.keys())

    result: Dict[str, Dict] = {}
    for speaker in all_speakers:
        words = per_speaker_words.get(speaker, [])
        total_speech_s = sum(t.duration() for t in per_speaker_turns.get(speaker, []))
        n_words = len(words)
        wpm = (n_words / total_speech_s * 60.0) if total_speech_s > 0 else 0.0

        # Pause rate = fraction of inter-word gaps that exceed 0.4 s
        gaps = []
        sw = sorted(words, key=lambda w: w["start"])
        for a, b in zip(sw, sw[1:]):
            gap = max(0.0, b["start"] - a["end"])
            gaps.append(gap)
        pause_rate = float(sum(1 for g in gaps if g > 0.4) / max(1, len(gaps)))
        total_minutes = max(total_speech_s / 60.0, 1e-6)
        pause_frequency = sum(1 for g in gaps if g > 0.4) / total_minutes

        filler_hits = sum(1 for w in words if w["text"].lower().strip("'.,!?") in filler_lexicon)
        filler_ratio = filler_hits / max(1, n_words)

        style_value, style_conf = _style_from_metrics(wpm, pause_rate, filler_ratio)
        accent_value, accent_conf = _accent_proxy(language, scripts)
        formality_value, formality_conf = _formality_from_metrics(filler_ratio, pause_rate)

        # speech_ratio: fraction of total audio this speaker is speaking.
        speech_ratio: Optional[float] = None
        if total_audio_duration_s > 0:
            speech_ratio = round(total_speech_s / total_audio_duration_s, 4)

        result[speaker] = {
            "speaker_id": speaker,
            "label": (speaker_labels or {}).get(speaker, speaker),
            "total_speaking_time_s": round(total_speech_s, 3),
            "word_count": n_words,
            "wpm": round(wpm, 2),
            "pause_rate": round(pause_rate, 3),
            "pause_frequency_per_min": round(pause_frequency, 3),
            "filler_ratio": round(filler_ratio, 3),
            "speaking_style": {
                **inferred_field(style_value, style_conf),
                "method": "wpm_pause_filler_heuristic",
            },
            "formality": {
                **inferred_field(formality_value, formality_conf),
                "method": "filler_pause_heuristic",
            },
            "accent": {
                **inferred_field(accent_value, accent_conf),
                "method": "language_script_proxy",
            },
            "region": {
                **inferred_field("unknown", 0.0),
                "method": "not_available_without_user_metadata",
            },
            "dialect": {
                **inferred_field("unknown", 0.0),
                "method": "not_available_without_user_metadata",
            },
            "turn_count": len(per_speaker_turns.get(speaker, [])),
            "speech_ratio": speech_ratio,
            "overlap_ratio": round((overlap_ratios or {}).get(speaker, 0.0), 4),
            "dominance_score": round((dominance or {}).get(speaker, 0.0), 4),
        }
    return result


# ---------------------------------------------------------------------------
#  Conversation-level features
# ---------------------------------------------------------------------------

_STOPWORDS = {
    # English
    "the", "a", "an", "and", "or", "but", "if", "so", "of", "to", "for", "on",
    "in", "at", "by", "with", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their", "its", "do", "does", "did", "have", "has", "had", "not", "no",
    "yes", "ok", "okay", "yeah", "yep", "nope", "hi", "hello", "hey",
    # Hindi/Hinglish romanised fillers
    "hai", "hain", "ho", "hoon", "tha", "thi", "the", "kya", "kyun", "kab",
    "kahan", "kaise", "kaisa", "main", "tum", "aap", "hum", "toh", "to", "bhi",
    "bas", "haan", "nahi", "nahin", "ji", "mera", "meri", "tera", "teri",
    "apna", "apni", "sab", "kuch", "kuchh", "bahut", "na", "ke", "ki", "ka",
    "se", "par", "yeh", "vo", "aur",
}


def _extract_keywords(texts: List[str], top_k: int = 10) -> List[str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return _extract_keywords_counter(texts, top_k)
    cleaned = [re.sub(r"[^\w\s]", " ", t.lower()) for t in texts if t and t.strip()]
    if not cleaned:
        return []
    try:
        vec = TfidfVectorizer(
            stop_words=list(_STOPWORDS),
            max_features=256,
            ngram_range=(1, 2),
            min_df=1,
        )
        mat = vec.fit_transform(cleaned)
        scores = mat.sum(axis=0).A1
        vocab = vec.get_feature_names_out()
        order = np.argsort(-scores)
        keywords: List[str] = []
        for idx in order:
            term = vocab[idx]
            if any(tok in _STOPWORDS for tok in term.split()):
                continue
            if term.isdigit():
                continue
            keywords.append(term)
            if len(keywords) >= top_k:
                break
        return keywords
    except Exception:
        return _extract_keywords_counter(texts, top_k)


def _extract_keywords_counter(texts: List[str], top_k: int) -> List[str]:
    cnt: Counter = Counter()
    for text in texts:
        for tok in _words_in(text.lower()):
            if tok in _STOPWORDS or tok.isdigit() or len(tok) < 3:
                continue
            cnt[tok] += 1
    return [t for t, _ in cnt.most_common(top_k)]


INTENT_RULES: List[Tuple[str, List[str]]] = [
    ("greeting", [r"\b(hi|hello|hey|namaste|namaskar|sat sri akaal|kem cho)\b"]),
    ("farewell", [r"\b(bye|goodbye|alvida|phir milenge|see you)\b"]),
    ("question", [r"\?", r"\b(what|when|where|why|how|kya|kab|kahan|kyun|kaise)\b"]),
    ("complaint", [r"\b(complain|issue|problem|dikkat|pareshan|galat)\b"]),
    ("request", [r"\b(please|kripya|kindly|could you|can you|mujhe chahiye)\b"]),
    ("confirmation", [r"\b(ok|okay|sure|theek|thik|accha|haan|yes)\b"]),
    ("negation", [r"\b(no|nahi|nahin|nope|bilkul nahi)\b"]),
    ("support_inquiry", [r"\b(support|help|madad|sahayata|assist)\b"]),
    ("transaction", [r"\b(buy|purchase|order|payment|kharid|paisa|price|kitna)\b"]),
]

DOMAIN_RULES: List[Tuple[str, List[str]]] = [
    ("customer_support", [r"\b(support|help|issue|complaint|ticket|refund|madad|sahayata)\b"]),
    ("sales", [r"\b(buy|purchase|order|pricing|quote|demo|kharid|kitna)\b"]),
    ("onboarding", [r"\b(onboarding|setup|sign up|signup|register|login|account setup)\b"]),
    ("healthcare", [r"\b(doctor|clinic|hospital|medicine|appointment|symptom|health)\b"]),
    ("education", [r"\b(student|teacher|class|course|lesson|school|college|exam)\b"]),
    ("finance", [r"\b(bank|loan|emi|credit|debit|invoice|payment|balance|policy)\b"]),
    ("travel", [r"\b(flight|hotel|booking|travel|trip|train|bus|cab|ticket)\b"]),
]


def _classify_intent(text: str) -> List[str]:
    hits: List[str] = []
    lower = text.lower()
    for label, patterns in INTENT_RULES:
        if any(re.search(p, lower) for p in patterns):
            hits.append(label)
    return hits or ["informational"]


def _infer_domain(full_text: str, keywords: List[str], intents: List[str]) -> Tuple[str, float]:
    lower = full_text.lower()
    matched: List[str] = []
    for label, patterns in DOMAIN_RULES:
        if any(re.search(pattern, lower) for pattern in patterns):
            matched.append(label)

    if len(matched) >= 2:
        return "mixed", 0.55
    if len(matched) == 1:
        return matched[0], 0.68
    if "support_inquiry" in intents or "complaint" in intents:
        return "customer_support", 0.58
    if "transaction" in intents:
        return "sales", 0.56
    if keywords:
        return "casual", 0.42
    return "unknown", 0.2


def extract_conversation_metadata(
    transcript: Transcript,
    turns: List[SpeakerTurn],
) -> Dict:
    turn_durations = [t.duration() for t in turns]
    avg_turn = float(np.mean(turn_durations)) if turn_durations else 0.0
    seg_texts = [s.text for s in transcript.segments if s.text.strip()]
    keywords = _extract_keywords(seg_texts)
    full_text = " ".join(seg_texts)
    intents = _classify_intent(full_text)
    domain_value, domain_conf = _infer_domain(full_text, keywords, intents)

    # Aggregate per-segment quality signals so a dataset curator can filter
    # at the conversation level without reading every transcript segment.
    scored = [s for s in transcript.segments if s.text.strip()]
    if scored:
        scores = [float(s.quality_score) for s in scored]
        durations = [max(0.0, s.end - s.start) for s in scored]
        total_dur = sum(durations) or 1.0
        weighted = sum(sc * d for sc, d in zip(scores, durations)) / total_dur
        quality_summary = {
            "mean_quality_score": round(float(np.mean(scores)), 4),
            "median_quality_score": round(float(np.median(scores)), 4),
            "duration_weighted_quality_score": round(float(weighted), 4),
            "low_quality_segment_ratio": round(
                float(sum(1 for s in scores if s < 0.35) / len(scores)), 4
            ),
            "high_quality_segment_ratio": round(
                float(sum(1 for s in scores if s >= 0.7) / len(scores)), 4
            ),
            "segment_count": len(scored),
        }
    else:
        quality_summary = {
            "mean_quality_score": 0.0,
            "median_quality_score": 0.0,
            "duration_weighted_quality_score": 0.0,
            "low_quality_segment_ratio": 0.0,
            "high_quality_segment_ratio": 0.0,
            "segment_count": 0,
        }

    return {
        "turn_count": len(turns),
        "speaker_count": len({t.speaker for t in turns}),
        "avg_turn_length_s": round(avg_turn, 3),
        "total_speech_time_s": round(sum(turn_durations), 3),
        "topic": inferred_field(keywords[0] if keywords else "unknown", 0.58 if keywords else 0.2),
        "sub_topic": inferred_field(keywords[1] if len(keywords) > 1 else "unknown", 0.46 if len(keywords) > 1 else 0.2),
        "domain": {
            **inferred_field(domain_value, domain_conf),
            "method": "keyword_intent_heuristic",
        },
        "topic_keywords": keywords,
        "intents": intents,
        "quality": quality_summary,
    }
