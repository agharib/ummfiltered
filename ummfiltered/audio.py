from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ummfiltered.models import Word


def compute_rms_db(samples: np.ndarray) -> float:
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    if rms == 0:
        return -100.0
    return 20 * np.log10(rms)


def extract_audio_pcm(video_path: Path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", "1",
            tmp_path,
        ],
        capture_output=True,
        check=True,
    )

    import wave
    with wave.open(tmp_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    Path(tmp_path).unlink()
    return samples, sample_rate


def find_silence_boundaries(
    samples: np.ndarray,
    sample_rate: int,
    filler_start: float,
    filler_end: float,
    threshold_db: float = -40,
    max_expansion_ms: float = 300,
    window_ms: float = 20,
) -> tuple[float, float]:
    max_expansion_s = max_expansion_ms / 1000.0
    window_samples = int(sample_rate * window_ms / 1000.0)

    start_sample = int(filler_start * sample_rate)
    min_start_sample = int((filler_start - max_expansion_s) * sample_rate)
    min_start_sample = max(0, min_start_sample)

    new_start_sample = start_sample
    pos = start_sample - window_samples
    while pos >= min_start_sample:
        chunk = samples[max(0, pos):pos + window_samples]
        if compute_rms_db(chunk) < threshold_db:
            new_start_sample = pos
            pos -= window_samples
        else:
            break

    end_sample = int(filler_end * sample_rate)
    max_end_sample = int((filler_end + max_expansion_s) * sample_rate)
    max_end_sample = min(len(samples), max_end_sample)

    new_end_sample = end_sample
    pos = end_sample
    while pos + window_samples <= max_end_sample:
        chunk = samples[pos:pos + window_samples]
        if compute_rms_db(chunk) < threshold_db:
            new_end_sample = pos + window_samples
            pos += window_samples
        else:
            break

    return new_start_sample / sample_rate, new_end_sample / sample_rate


def find_speech_onset(
    samples: np.ndarray,
    sample_rate: int,
    region_start: float,
    region_end: float,
    window_ms: float = 10,
    noise_floor_ms: float = 30,
    threshold_factor: float = 3.0,
) -> float:
    window = int(window_ms / 1000.0 * sample_rate)
    step = int(0.005 * sample_rate)

    noise_start = int(region_start * sample_rate)
    noise_end = noise_start + int(noise_floor_ms / 1000.0 * sample_rate)
    noise_end = min(noise_end, int(region_end * sample_rate))
    if noise_end <= noise_start:
        return region_end
    noise_rms = np.sqrt(np.mean(samples[noise_start:noise_end] ** 2)) + 1e-10
    threshold = noise_rms * threshold_factor

    end_sample = int(region_end * sample_rate)
    start_sample = int(region_start * sample_rate) + window

    onset_sample = end_sample
    pos = end_sample
    while pos > start_sample:
        rms = np.sqrt(np.mean(samples[pos - window:pos] ** 2))
        if rms > threshold:
            onset_sample = pos - window
        else:
            break
        pos -= step

    return onset_sample / sample_rate


def find_speech_tail(
    samples: np.ndarray,
    sample_rate: int,
    region_start: float,
    region_end: float,
    window_ms: float = 10,
    noise_floor_ms: float = 30,
    threshold_factor: float = 3.0,
) -> float:
    window = int(window_ms / 1000.0 * sample_rate)
    step = int(0.005 * sample_rate)

    noise_start = int(region_end * sample_rate) - int(noise_floor_ms / 1000.0 * sample_rate)
    noise_end = int(region_end * sample_rate)
    noise_start = max(noise_start, int(region_start * sample_rate))
    if noise_end <= noise_start:
        return region_start
    noise_rms = np.sqrt(np.mean(samples[noise_start:noise_end] ** 2)) + 1e-10
    threshold = noise_rms * threshold_factor

    start_sample = int(region_start * sample_rate)
    end_sample = int(region_end * sample_rate) - window

    tail_sample = start_sample
    pos = start_sample
    while pos < end_sample:
        rms = np.sqrt(np.mean(samples[pos:pos + window] ** 2))
        if rms > threshold:
            tail_sample = pos + window
        else:
            break
        pos += step

    return tail_sample / sample_rate


PRECEDING_MARGIN_S = 0.05
FOLLOWING_GAP_THRESHOLD_S = 0.15


def protect_adjacent_words(
    new_start: float,
    new_end: float,
    non_filler_words: list[Word],
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[float, float] | None:
    closest_word_end = max(
        (w.end for w in non_filler_words if w.end <= new_start),
        default=-float("inf"),
    )
    preceding_gap = new_start - closest_word_end
    if preceding_gap < 0.1 and closest_word_end > 0:
        new_start = closest_word_end + PRECEDING_MARGIN_S
        if new_start >= new_end:
            return None

    closest_word_start = min(
        (w.start for w in non_filler_words if w.start > new_start),
        default=float("inf"),
    )
    following_gap = closest_word_start - new_end
    if closest_word_start < float("inf"):
        if following_gap < 0:
            new_end = closest_word_start - PRECEDING_MARGIN_S
            if new_start >= new_end:
                return None
        elif following_gap < FOLLOWING_GAP_THRESHOLD_S:
            onset = find_speech_onset(
                samples, sample_rate,
                region_start=max(closest_word_start - 0.10, new_start),
                region_end=closest_word_start + 0.05,
            )
            safe_end = min(onset - 0.05, closest_word_start - 0.05)
            new_end = min(new_end, safe_end)
            if new_start >= new_end:
                return None

    return new_start, new_end


AUDIO_CROSSFADE_MS = 40


def smooth_rendered_audio(
    rendered_samples: np.ndarray,
    sample_rate: int,
    cut_points: list[float],
    room_tone: np.ndarray,
    crossfade_ms: float = AUDIO_CROSSFADE_MS,
) -> np.ndarray:
    result = rendered_samples.copy().astype(np.float64)
    cf_samples = int(crossfade_ms / 1000.0 * sample_rate)

    for cp in cut_points:
        cp_sample = int(cp * sample_rate)

        fade_before = min(cf_samples, cp_sample)
        if fade_before > 0:
            start = cp_sample - fade_before
            fade_out = np.linspace(1.0, 0.0, fade_before)
            fade_in_rt = np.linspace(0.0, 1.0, fade_before)
            rt = _get_room_tone_fill(room_tone, fade_before)
            result[start:cp_sample] = result[start:cp_sample] * fade_out + rt * fade_in_rt

        fade_after = min(cf_samples, len(result) - cp_sample)
        if fade_after > 0:
            end = cp_sample + fade_after
            fade_in = np.linspace(0.0, 1.0, fade_after)
            fade_out_rt = np.linspace(1.0, 0.0, fade_after)
            rt = _get_room_tone_fill(room_tone, fade_after)
            result[cp_sample:end] = result[cp_sample:end] * fade_in + rt * fade_out_rt

    return result.astype(np.float32)


def _get_room_tone_fill(room_tone: np.ndarray, length: int) -> np.ndarray:
    if len(room_tone) == 0:
        return np.zeros(length, dtype=np.float64)
    tone = np.tile(room_tone, (length // len(room_tone)) + 1)[:length]
    return tone.copy().astype(np.float64)


def extract_room_tone(
    samples: np.ndarray,
    sample_rate: int,
    words: list[Word] | None = None,
    target_s: float = 0.5,
    max_db: float = -50,
) -> np.ndarray:
    target_samples = int(target_s * sample_rate)
    window_ms = 30
    window = int(window_ms / 1000.0 * sample_rate)

    if words:
        gaps: list[tuple[int, int]] = []
        sorted_words = sorted(words, key=lambda w: w.start)
        for i in range(len(sorted_words) - 1):
            gap_start = int(sorted_words[i].end * sample_rate)
            gap_end = int(sorted_words[i + 1].start * sample_rate)
            if gap_end - gap_start >= window:
                gaps.append((gap_start, gap_end))
        if sorted_words:
            pre_start = 0
            pre_end = int(sorted_words[0].start * sample_rate)
            if pre_end - pre_start >= window:
                gaps.append((pre_start, pre_end))
            post_start = int(sorted_words[-1].end * sample_rate)
            post_end = len(samples)
            if post_end - post_start >= window:
                gaps.append((post_start, post_end))
    else:
        gaps = [(0, len(samples))]

    quiet_chunks: list[tuple[float, int]] = []
    for gap_start, gap_end in gaps:
        pos = gap_start
        while pos + window <= gap_end:
            chunk = samples[pos:pos + window]
            db = compute_rms_db(chunk)
            if db < max_db:
                quiet_chunks.append((db, pos))
            pos += window

    quiet_chunks.sort(key=lambda x: x[0])

    collected = np.array([], dtype=samples.dtype)
    for _, pos in quiet_chunks:
        chunk = samples[pos:pos + window]
        collected = np.concatenate([collected, chunk])
        if len(collected) >= target_samples:
            break

    if len(collected) < window:
        return np.zeros(target_samples, dtype=samples.dtype)

    return collected[:target_samples] if len(collected) >= target_samples else collected
