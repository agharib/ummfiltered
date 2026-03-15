from __future__ import annotations

from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
import wave

import numpy as np

from ummfiltered.ffmpeg_tools import ffmpeg_cmd
from ummfiltered.models import EditDecisionList, SeamReport, SeamReportEntry

if TYPE_CHECKING:
    from ummfiltered.models import Word


def compute_rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -100.0
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    if rms == 0:
        return -100.0
    return 20 * np.log10(rms)


def compute_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _as_audio_matrix(samples: np.ndarray) -> tuple[np.ndarray, bool]:
    if samples.ndim == 1:
        return samples[:, None].astype(np.float32), True
    return samples.astype(np.float32), False


def _restore_audio_shape(samples: np.ndarray, was_mono: bool) -> np.ndarray:
    if was_mono:
        return samples[:, 0].astype(np.float32)
    return samples.astype(np.float32)


def downmix_audio(samples: np.ndarray) -> np.ndarray:
    matrix, _ = _as_audio_matrix(samples)
    return np.mean(matrix, axis=1).astype(np.float32)


def extract_audio_matrix(
    video_path: Path,
    sample_rate: int | None = None,
    channel_count: int | None = None,
) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    cmd = ffmpeg_cmd(
        "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
    )
    if sample_rate is not None:
        cmd.extend(["-ar", str(sample_rate)])
    if channel_count is not None:
        cmd.extend(["-ac", str(channel_count)])
    cmd.append(tmp_path)

    subprocess.run(cmd, capture_output=True, check=True)

    with wave.open(tmp_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        channels = max(1, wf.getnchannels())
        actual_rate = wf.getframerate()
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        audio = samples.reshape(-1, channels)

    Path(tmp_path).unlink(missing_ok=True)
    return audio, actual_rate


def extract_audio_pcm(video_path: Path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    audio, actual_rate = extract_audio_matrix(video_path, sample_rate=sample_rate, channel_count=1)
    return audio[:, 0], actual_rate


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

    mono = downmix_audio(samples) if samples.ndim > 1 else samples

    start_sample = int(filler_start * sample_rate)
    min_start_sample = int((filler_start - max_expansion_s) * sample_rate)
    min_start_sample = max(0, min_start_sample)

    new_start_sample = start_sample
    pos = start_sample - window_samples
    while pos >= min_start_sample:
        chunk = mono[max(0, pos):pos + window_samples]
        if compute_rms_db(chunk) < threshold_db:
            new_start_sample = pos
            pos -= window_samples
        else:
            break

    end_sample = int(filler_end * sample_rate)
    max_end_sample = int((filler_end + max_expansion_s) * sample_rate)
    max_end_sample = min(len(mono), max_end_sample)

    new_end_sample = end_sample
    pos = end_sample
    while pos + window_samples <= max_end_sample:
        chunk = mono[pos:pos + window_samples]
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
    mono = downmix_audio(samples) if samples.ndim > 1 else samples
    window = int(window_ms / 1000.0 * sample_rate)
    step = int(0.005 * sample_rate)

    noise_start = int(region_start * sample_rate)
    noise_end = noise_start + int(noise_floor_ms / 1000.0 * sample_rate)
    noise_end = min(noise_end, int(region_end * sample_rate))
    if noise_end <= noise_start:
        return region_end
    noise_rms = np.sqrt(np.mean(mono[noise_start:noise_end] ** 2)) + 1e-10
    threshold = noise_rms * threshold_factor

    end_sample = int(region_end * sample_rate)
    start_sample = int(region_start * sample_rate) + window

    onset_sample = end_sample
    pos = end_sample
    while pos > start_sample:
        rms = np.sqrt(np.mean(mono[pos - window:pos] ** 2))
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
    mono = downmix_audio(samples) if samples.ndim > 1 else samples
    window = int(window_ms / 1000.0 * sample_rate)
    step = int(0.005 * sample_rate)

    noise_start = int(region_end * sample_rate) - int(noise_floor_ms / 1000.0 * sample_rate)
    noise_end = int(region_end * sample_rate)
    noise_start = max(noise_start, int(region_start * sample_rate))
    if noise_end <= noise_start:
        return region_start
    noise_rms = np.sqrt(np.mean(mono[noise_start:noise_end] ** 2)) + 1e-10
    threshold = noise_rms * threshold_factor

    start_sample = int(region_start * sample_rate)
    end_sample = int(region_end * sample_rate) - window

    tail_sample = start_sample
    pos = start_sample
    while pos < end_sample:
        rms = np.sqrt(np.mean(mono[pos:pos + window] ** 2))
        if rms > threshold:
            tail_sample = pos + window
        else:
            break
        pos += step

    return tail_sample / sample_rate


ADJACENT_WORD_MARGIN_S = 0.01
FOLLOWING_GAP_THRESHOLD_S = 0.15


def protect_adjacent_words(
    new_start: float,
    new_end: float,
    non_filler_words: list[Word],
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[float, float] | None:
    overlapping_start = [
        word for word in non_filler_words
        if word.start < new_start < word.end
    ]
    if overlapping_start:
        new_start = max(word.end for word in overlapping_start) + ADJACENT_WORD_MARGIN_S
        if new_start >= new_end:
            return None

    overlapping_end = [
        word for word in non_filler_words
        if word.start < new_end < word.end
    ]
    if overlapping_end:
        new_end = min(word.start for word in overlapping_end) - ADJACENT_WORD_MARGIN_S
        if new_start >= new_end:
            return None

    closest_word_end = max(
        (w.end for w in non_filler_words if w.end <= new_start),
        default=-float("inf"),
    )
    preceding_gap = new_start - closest_word_end
    if preceding_gap < 0.1 and closest_word_end > 0:
        new_start = closest_word_end + ADJACENT_WORD_MARGIN_S
        if new_start >= new_end:
            return None

    closest_word_start = min(
        (w.start for w in non_filler_words if w.start > new_start),
        default=float("inf"),
    )
    following_gap = closest_word_start - new_end
    if closest_word_start < float("inf"):
        if following_gap < 0:
            new_end = closest_word_start - ADJACENT_WORD_MARGIN_S
            if new_start >= new_end:
                return None
        elif following_gap < FOLLOWING_GAP_THRESHOLD_S:
            onset = find_speech_onset(
                samples,
                sample_rate,
                region_start=max(closest_word_start - 0.10, new_start),
                region_end=closest_word_start + 0.05,
            )
            safe_end = min(
                onset - ADJACENT_WORD_MARGIN_S,
                closest_word_start - ADJACENT_WORD_MARGIN_S,
            )
            new_end = min(new_end, safe_end)
            if new_start >= new_end:
                return None

    return new_start, new_end


AUDIO_CROSSFADE_MS = 40
NATURALNESS_WINDOW_MS = 20
MORPH_BRIDGE_MS = 12
ANCHOR_SEARCH_MS = 40
TAIL_PRESERVE_MS = 56
MICRO_BRIDGE_MS = 28
CHANNEL_GUARD_MARGIN = 0.2
CHANNEL_GUARD_FACTOR = 1.15
LOW_ENERGY_THRESHOLD_DB = -38.0
RISKY_SEAM_THRESHOLD = 1.1


@dataclass
class CutNaturalnessMetrics:
    amplitude_jump: float
    dc_jump: float
    rms_jump_db: float
    center_drop_db: float
    spectral_jump: float
    score: float


@dataclass
class _CutEvaluation:
    downmix: CutNaturalnessMetrics
    channels: list[CutNaturalnessMetrics]


@dataclass
class _Boundary:
    index: int
    cut_sample: int
    left_len: int
    right_len: int
    allow_room_tone: bool
    max_left_blend: int
    max_right_blend: int


@dataclass
class _AudioPiece:
    kind: str
    samples: np.ndarray
    output_start_sample: int
    output_end_sample: int
    decision_index: int | None = None

    @property
    def length(self) -> int:
        return self.output_end_sample - self.output_start_sample


def _spectral_distance(before: np.ndarray, after: np.ndarray) -> float:
    if before.size == 0 or after.size == 0:
        return 0.0
    length = min(len(before), len(after))
    if length < 4:
        return 0.0
    before = before[-length:]
    after = after[:length]
    window = np.hanning(length)
    window_before = before * window
    window_after = after * window
    before_fft = np.abs(np.fft.rfft(window_before))
    after_fft = np.abs(np.fft.rfft(window_after))
    before_norm = before_fft / max(np.sum(before_fft), 1e-8)
    after_norm = after_fft / max(np.sum(after_fft), 1e-8)
    return float(np.mean(np.abs(before_norm - after_norm)))


def measure_cut_naturalness(
    samples: np.ndarray,
    sample_rate: int,
    cut_point: float,
    window_ms: float = NATURALNESS_WINDOW_MS,
) -> CutNaturalnessMetrics:
    if getattr(samples, "ndim", 1) > 1:
        samples = downmix_audio(samples)
    if len(samples) < 4:
        return CutNaturalnessMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    cp_sample = int(cut_point * sample_rate)
    cp_sample = max(1, min(cp_sample, len(samples) - 1))
    window_samples = max(16, int(sample_rate * window_ms / 1000.0))

    before = samples[max(0, cp_sample - window_samples):cp_sample]
    after = samples[cp_sample:min(len(samples), cp_sample + window_samples)]
    center_half = max(8, window_samples // 2)
    center = samples[max(0, cp_sample - center_half):min(len(samples), cp_sample + center_half)]

    if before.size == 0 or after.size == 0 or center.size == 0:
        return CutNaturalnessMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    amplitude_jump = float(abs(after[0] - before[-1]))
    dc_jump = float(abs(np.mean(after) - np.mean(before)))
    before_db = compute_rms_db(before)
    after_db = compute_rms_db(after)
    center_db = compute_rms_db(center)
    rms_jump_db = abs(after_db - before_db)
    center_drop_db = max(0.0, min(before_db, after_db) - center_db)
    spectral_jump = _spectral_distance(before, after)

    score = (
        amplitude_jump * 6.0
        + dc_jump * 4.0
        + rms_jump_db * 0.05
        + center_drop_db * 0.4
        + spectral_jump * 8.0
    )
    return CutNaturalnessMetrics(
        amplitude_jump=amplitude_jump,
        dc_jump=dc_jump,
        rms_jump_db=rms_jump_db,
        center_drop_db=center_drop_db,
        spectral_jump=spectral_jump,
        score=score,
    )


def _evaluate_cut(samples: np.ndarray, sample_rate: int, cut_sample: int) -> _CutEvaluation:
    matrix, _ = _as_audio_matrix(samples)
    cut_time = cut_sample / sample_rate
    return _CutEvaluation(
        downmix=measure_cut_naturalness(downmix_audio(matrix), sample_rate, cut_time),
        channels=[
            measure_cut_naturalness(matrix[:, channel], sample_rate, cut_time)
            for channel in range(matrix.shape[1])
        ],
    )


def _candidate_allowed(raw_eval: _CutEvaluation, candidate_eval: _CutEvaluation) -> bool:
    for raw_channel, candidate_channel in zip(raw_eval.channels, candidate_eval.channels):
        allowed_score = max(
            raw_channel.score + CHANNEL_GUARD_MARGIN,
            raw_channel.score * CHANNEL_GUARD_FACTOR + 0.05,
        )
        if candidate_channel.score > allowed_score:
            return False
    return True


def _resample_multichannel(samples: np.ndarray, target_length: int) -> np.ndarray:
    matrix, _ = _as_audio_matrix(samples)
    if target_length <= 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.float64)
    if len(matrix) == 0:
        return np.zeros((target_length, matrix.shape[1]), dtype=np.float64)
    if len(matrix) == target_length:
        return matrix.astype(np.float64)

    source_positions = np.linspace(0.0, 1.0, len(matrix), endpoint=True)
    target_positions = np.linspace(0.0, 1.0, target_length, endpoint=True)
    channels = [
        np.interp(target_positions, source_positions, matrix[:, channel])
        for channel in range(matrix.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float64)


def _get_room_tone_fill(room_tone: np.ndarray, length: int, channels: int) -> np.ndarray:
    matrix, _ = _as_audio_matrix(room_tone)
    if matrix.shape[1] != channels:
        if matrix.shape[1] == 1:
            matrix = np.repeat(matrix, channels, axis=1)
        else:
            matrix = matrix[:, :channels]
    if len(matrix) == 0:
        return np.zeros((length, channels), dtype=np.float64)
    repeats = (length // len(matrix)) + 1
    tone = np.tile(matrix, (repeats, 1))[:length]
    return tone.astype(np.float64)


def _apply_room_tone_cut(
    samples: np.ndarray,
    cp_sample: int,
    cf_samples: int,
    room_tone: np.ndarray,
) -> np.ndarray:
    matrix, _ = _as_audio_matrix(samples)
    result = matrix.copy().astype(np.float64)
    channels = result.shape[1]

    fade_before = min(cf_samples, cp_sample)
    if fade_before > 0:
        start = cp_sample - fade_before
        fade_out = np.linspace(1.0, 0.0, fade_before)[:, None]
        fade_in_rt = np.linspace(0.0, 1.0, fade_before)[:, None]
        rt = _get_room_tone_fill(room_tone, fade_before, channels)
        result[start:cp_sample] = result[start:cp_sample] * fade_out + rt * fade_in_rt

    fade_after = min(cf_samples, len(result) - cp_sample)
    if fade_after > 0:
        end = cp_sample + fade_after
        fade_in = np.linspace(0.0, 1.0, fade_after)[:, None]
        fade_out_rt = np.linspace(1.0, 0.0, fade_after)[:, None]
        rt = _get_room_tone_fill(room_tone, fade_after, channels)
        result[cp_sample:end] = result[cp_sample:end] * fade_in + rt * fade_out_rt

    return result.astype(np.float32)


def _apply_boundary_morph_cut(
    samples: np.ndarray,
    cp_sample: int,
    bridge_samples: int,
    left_shift: int = 0,
    right_shift: int = 0,
) -> np.ndarray:
    matrix, _ = _as_audio_matrix(samples)
    result = matrix.copy().astype(np.float64)
    left_end = max(0, cp_sample - left_shift)
    right_start = min(len(result), cp_sample + right_shift)
    available = min(bridge_samples, left_end, len(result) - right_start)
    if available < 8:
        return result.astype(np.float32)

    tail = result[left_end - available:left_end]
    head = result[right_start:right_start + available]
    tail_seed = _resample_multichannel(tail, available * 2)
    head_seed = _resample_multichannel(head, available * 2)
    weights = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, available * 2))[:, None]
    blended = tail_seed * (1.0 - weights) + head_seed * weights

    target_start = max(0, cp_sample - available)
    target_end = min(len(result), cp_sample + available)
    if target_end - target_start != len(blended):
        blended = _resample_multichannel(blended, target_end - target_start)
    result[target_start:target_end] = blended
    return result.astype(np.float32)


def _apply_tail_preserving_cut(
    samples: np.ndarray,
    cp_sample: int,
    bridge_samples: int,
    left_shift: int = 0,
    right_shift: int = 0,
) -> np.ndarray:
    matrix, _ = _as_audio_matrix(samples)
    result = matrix.copy().astype(np.float64)
    if bridge_samples < 8:
        return result.astype(np.float32)

    left_end = max(0, cp_sample - left_shift)
    right_start = min(len(result), cp_sample + right_shift)
    left_span = min(bridge_samples, left_end)
    right_span = min(max(6, bridge_samples // 2), len(result) - right_start)
    if left_span < 8 or right_span < 6:
        return result.astype(np.float32)

    target_start = max(0, cp_sample - left_span)
    target_end = min(len(result), cp_sample + right_span)
    target_length = target_end - target_start
    if target_length < 12:
        return result.astype(np.float32)

    left_window = result[left_end - left_span:left_end]
    right_window = result[right_start:right_start + right_span]
    left_seed = _resample_multichannel(left_window, target_length)
    right_seed = _resample_multichannel(right_window, target_length)
    positions = np.linspace(0.0, 1.0, target_length, dtype=np.float64)[:, None]
    fade_in = np.clip(positions ** 2.4, 0.0, 1.0)
    fade_out = np.sqrt(np.clip(1.0 - fade_in ** 2, 0.0, 1.0))
    result[target_start:target_end] = left_seed * fade_out + right_seed * fade_in
    return result.astype(np.float32)


def _apply_equal_power_crossfade_cut(
    samples: np.ndarray,
    cp_sample: int,
    cf_samples: int,
    left_shift: int = 0,
    right_shift: int = 0,
) -> np.ndarray:
    matrix, _ = _as_audio_matrix(samples)
    if cf_samples < 4:
        return matrix.astype(np.float32)

    result = matrix.copy().astype(np.float64)
    left_end = max(0, cp_sample - left_shift)
    right_start = min(len(result), cp_sample + right_shift)
    left_window = result[max(0, left_end - cf_samples):left_end]
    right_window = result[right_start:min(len(result), right_start + cf_samples)]
    if len(left_window) < 4 or len(right_window) < 4:
        return result.astype(np.float32)

    target_start = max(0, cp_sample - cf_samples)
    target_end = min(len(result), cp_sample + cf_samples)
    target_length = target_end - target_start
    if target_length < 8:
        return result.astype(np.float32)

    left_seed = _resample_multichannel(left_window, target_length)
    right_seed = _resample_multichannel(right_window, target_length)
    theta = np.linspace(0.0, np.pi / 2.0, target_length)[:, None]
    fade_out = np.cos(theta)
    fade_in = np.sin(theta)
    result[target_start:target_end] = left_seed * fade_out + right_seed * fade_in
    return result.astype(np.float32)


def _apply_micro_bridge_cut(
    samples: np.ndarray,
    cp_sample: int,
    bridge_samples: int,
    left_shift: int = 0,
    right_shift: int = 0,
) -> np.ndarray:
    matrix, _ = _as_audio_matrix(samples)
    result = matrix.copy().astype(np.float64)
    if bridge_samples < 8:
        return result.astype(np.float32)

    left_end = max(0, cp_sample - left_shift)
    right_start = min(len(result), cp_sample + right_shift)
    left_span = min(max(8, bridge_samples + bridge_samples // 2), left_end)
    right_span = min(max(6, bridge_samples), len(result) - right_start)
    if left_span < 8 or right_span < 6:
        return result.astype(np.float32)

    overlap = min(max(6, bridge_samples // 2), left_span, right_span)
    bridge_left = result[left_end - left_span:left_end]
    bridge_right = result[right_start:right_start + right_span]
    overlap_left = bridge_left[-overlap:]
    overlap_right = bridge_right[:overlap]
    fade = np.linspace(0.0, 1.0, overlap, dtype=np.float64)[:, None]
    center = overlap_left * (1.0 - fade) + overlap_right * fade
    bridge = np.concatenate([bridge_left[:-overlap], center, bridge_right[overlap:]], axis=0)

    target_start = max(0, cp_sample - left_span)
    target_end = min(len(result), cp_sample + right_span)
    target_length = target_end - target_start
    if target_length < 12:
        return result.astype(np.float32)
    bridge = _resample_multichannel(bridge, target_length)
    result[target_start:target_end] = bridge
    return result.astype(np.float32)


def _build_ambient_pad(
    source_audio: np.ndarray,
    sample_rate: int,
    left_time: float,
    right_time: float,
    duration_samples: int,
    room_tone: np.ndarray,
) -> np.ndarray:
    matrix, _ = _as_audio_matrix(source_audio)
    channels = matrix.shape[1]
    if duration_samples <= 0:
        return np.zeros((0, channels), dtype=np.float32)

    search_samples = int(sample_rate * 0.35)
    window_samples = max(16, int(sample_rate * 0.03))
    candidates: list[tuple[float, np.ndarray]] = []

    def _collect(anchor_sample: int, direction: int) -> None:
        start = max(0, anchor_sample - search_samples)
        end = min(len(matrix), anchor_sample + search_samples)
        if end - start < window_samples:
            return
        pos = start
        while pos + window_samples <= end:
            chunk = matrix[pos:pos + window_samples]
            score = compute_rms_db(downmix_audio(chunk))
            candidates.append((score, chunk.copy()))
            pos += window_samples

    _collect(int(left_time * sample_rate), -1)
    _collect(int(right_time * sample_rate), 1)
    candidates.sort(key=lambda item: item[0])

    low_energy = [chunk for score, chunk in candidates if score <= LOW_ENERGY_THRESHOLD_DB]
    if low_energy:
        tiled = np.concatenate(low_energy, axis=0)
        repeats = (duration_samples // len(tiled)) + 1
        return np.tile(tiled, (repeats, 1))[:duration_samples].astype(np.float32)

    room_fill = _get_room_tone_fill(room_tone, duration_samples, channels)
    return room_fill.astype(np.float32)


def _candidate_offsets(limit: int, sample_rate: int) -> list[int]:
    if limit <= 0:
        return [0]
    step = max(1, int(sample_rate * 0.004))
    offsets = list(range(0, limit + 1, step))
    if offsets[-1] != limit:
        offsets.append(limit)
    return offsets


def _guided_candidate_offsets(
    samples: np.ndarray,
    cp_sample: int,
    limit: int,
    sample_rate: int,
    *,
    side: str,
) -> list[int]:
    base_offsets = _candidate_offsets(limit, sample_rate)
    if limit <= 0:
        return base_offsets

    mono = downmix_audio(samples)
    step = max(1, int(sample_rate * 0.004))
    scored: list[tuple[float, int]] = []
    for offset in base_offsets:
        if side == "left":
            idx = max(1, cp_sample - offset)
        else:
            idx = min(len(mono) - 2, cp_sample + offset)
        window_start = max(0, idx - step)
        window_end = min(len(mono), idx + step)
        window = mono[window_start:window_end]
        if window.size == 0:
            continue
        amplitude = abs(float(mono[idx]))
        energy = compute_rms(window)
        zero_cross_bonus = 0.0
        if idx > 0 and idx + 1 < len(mono):
            if mono[idx - 1] == 0 or mono[idx] == 0 or np.signbit(mono[idx - 1]) != np.signbit(mono[idx]):
                zero_cross_bonus = 0.05
        score = amplitude + energy * 0.75 - zero_cross_bonus
        scored.append((score, offset))

    scored.sort(key=lambda item: (item[0], item[1]))
    prioritized = [offset for _score, offset in scored[:8]]
    midpoint = limit // 2 if limit > 1 else limit
    combined = sorted(set([0, midpoint, limit, *prioritized, *base_offsets[:4]]))
    return [offset for offset in combined if offset <= limit]


def optimize_audio_seams(
    samples: np.ndarray,
    sample_rate: int,
    boundaries: list[_Boundary],
    room_tone: np.ndarray,
    crossfade_options_ms: tuple[int, ...] = (8, 16, 24, 40),
) -> tuple[np.ndarray, SeamReport]:
    matrix, was_mono = _as_audio_matrix(samples)
    result = matrix.copy()
    report = SeamReport()

    for boundary in boundaries:
        cp_sample = max(1, min(boundary.cut_sample, len(result) - 1))
        raw_eval = _evaluate_cut(result, sample_rate, cp_sample)
        best_audio = result.copy()
        best_eval = raw_eval
        best_strategy = "raw"
        best_left_shift = 0
        best_right_shift = 0
        best_duration_ms = 0.0
        best_notes = ""

        left_limit = min(
            boundary.left_len // 2,
            boundary.max_left_blend,
            int(sample_rate * ANCHOR_SEARCH_MS / 1000.0),
        )
        right_limit = min(
            boundary.right_len // 2,
            boundary.max_right_blend,
            int(sample_rate * ANCHOR_SEARCH_MS / 1000.0),
        )
        left_offsets = _guided_candidate_offsets(
            result,
            cp_sample,
            left_limit,
            sample_rate,
            side="left",
        )
        right_offsets = _guided_candidate_offsets(
            result,
            cp_sample,
            right_limit,
            sample_rate,
            side="right",
        )
        risky_seam = (
            raw_eval.downmix.score >= RISKY_SEAM_THRESHOLD
            or raw_eval.downmix.center_drop_db > 0.65
            or raw_eval.downmix.spectral_jump > 0.14
        )

        tail_bridge_samples = min(
            max(10, int(TAIL_PRESERVE_MS / 1000.0 * sample_rate)),
            boundary.max_left_blend,
            max(1, boundary.max_right_blend),
        )
        if tail_bridge_samples >= 8:
            for left_shift in left_offsets:
                for right_shift in right_offsets:
                    candidate = _apply_tail_preserving_cut(
                        result,
                        cp_sample,
                        tail_bridge_samples,
                        left_shift=left_shift,
                        right_shift=right_shift,
                    )
                    candidate_eval = _evaluate_cut(candidate, sample_rate, cp_sample)
                    if not _candidate_allowed(raw_eval, candidate_eval):
                        continue
                    if candidate_eval.downmix.score + 1e-6 < best_eval.downmix.score:
                        best_audio = candidate
                        best_eval = candidate_eval
                        best_strategy = "tail_preserving_bridge"
                        best_left_shift = left_shift
                        best_right_shift = right_shift
                        best_duration_ms = float(TAIL_PRESERVE_MS)
                        best_notes = "tail_preserving"

        crossfade_found = False
        for crossfade_ms in crossfade_options_ms:
            cf_samples = min(
                int(sample_rate * crossfade_ms / 1000.0),
                boundary.max_left_blend,
                boundary.max_right_blend,
            )
            if cf_samples < 4:
                continue
            for left_shift in left_offsets:
                for right_shift in right_offsets:
                    candidate = _apply_equal_power_crossfade_cut(
                        result,
                        cp_sample,
                        cf_samples,
                        left_shift=left_shift,
                        right_shift=right_shift,
                    )
                    candidate_eval = _evaluate_cut(candidate, sample_rate, cp_sample)
                    if not _candidate_allowed(raw_eval, candidate_eval):
                        continue
                    if candidate_eval.downmix.score + 1e-6 < best_eval.downmix.score:
                        best_audio = candidate
                        best_eval = candidate_eval
                        best_strategy = "equal_power_crossfade"
                        best_left_shift = left_shift
                        best_right_shift = right_shift
                        best_duration_ms = float(crossfade_ms)
                        best_notes = ""
                        crossfade_found = True

        if risky_seam and not boundary.allow_room_tone:
            micro_bridge_samples = min(
                max(8, int(MICRO_BRIDGE_MS / 1000.0 * sample_rate)),
                boundary.max_left_blend,
                boundary.max_right_blend,
            )
            if micro_bridge_samples >= 8:
                for left_shift in left_offsets:
                    for right_shift in right_offsets:
                        candidate = _apply_micro_bridge_cut(
                            result,
                            cp_sample,
                            micro_bridge_samples,
                            left_shift=left_shift,
                            right_shift=right_shift,
                        )
                        candidate_eval = _evaluate_cut(candidate, sample_rate, cp_sample)
                        if not _candidate_allowed(raw_eval, candidate_eval):
                            continue
                        if candidate_eval.downmix.score + 1e-6 < best_eval.downmix.score:
                            best_audio = candidate
                            best_eval = candidate_eval
                            best_strategy = "micro_bridge"
                            best_left_shift = left_shift
                            best_right_shift = right_shift
                            best_duration_ms = float(MICRO_BRIDGE_MS)
                            best_notes = "risky_speech_seam"

        if boundary.allow_room_tone:
            cf_samples = min(
                int(sample_rate * AUDIO_CROSSFADE_MS / 1000.0),
                boundary.max_left_blend,
                boundary.max_right_blend,
            )
            if cf_samples >= 4:
                candidate = _apply_room_tone_cut(result, cp_sample, cf_samples, room_tone)
                candidate_eval = _evaluate_cut(candidate, sample_rate, cp_sample)
                if _candidate_allowed(raw_eval, candidate_eval) and candidate_eval.downmix.score + 1e-6 < best_eval.downmix.score:
                    best_audio = candidate
                    best_eval = candidate_eval
                    best_strategy = "room_tone"
                    best_left_shift = 0
                    best_right_shift = 0
                    best_duration_ms = float(AUDIO_CROSSFADE_MS)
                    best_notes = ""

        if not crossfade_found or best_strategy == "raw":
            morph_samples = min(
                max(8, int(MORPH_BRIDGE_MS / 1000.0 * sample_rate)),
                boundary.max_left_blend,
                boundary.max_right_blend,
            )
            for left_shift in left_offsets:
                for right_shift in right_offsets:
                    if morph_samples < 8:
                        continue
                    candidate = _apply_boundary_morph_cut(
                        result,
                        cp_sample,
                        morph_samples,
                        left_shift=left_shift,
                        right_shift=right_shift,
                    )
                    candidate_eval = _evaluate_cut(candidate, sample_rate, cp_sample)
                    if not _candidate_allowed(raw_eval, candidate_eval):
                        continue
                    if candidate_eval.downmix.score + 1e-6 < best_eval.downmix.score:
                        best_audio = candidate
                        best_eval = candidate_eval
                        best_strategy = "boundary_morph"
                        best_left_shift = left_shift
                        best_right_shift = right_shift
                        best_duration_ms = float(MORPH_BRIDGE_MS)
                        best_notes = "morph fallback"

        result = best_audio
        report.entries.append(
            SeamReportEntry(
                seam_index=boundary.index,
                output_time=cp_sample / sample_rate,
                chosen_strategy=best_strategy,
                before_score=raw_eval.downmix.score,
                after_score=best_eval.downmix.score,
                left_shift_ms=best_left_shift * 1000.0 / sample_rate,
                right_shift_ms=best_right_shift * 1000.0 / sample_rate,
                duration_ms=best_duration_ms,
                accepted=True,
                notes=best_notes,
            )
        )

    return _restore_audio_shape(result, was_mono), report


def analyze_audio_seams(
    samples: np.ndarray,
    sample_rate: int,
    cut_points: list[float],
) -> SeamReport:
    matrix, _ = _as_audio_matrix(samples)
    if len(matrix) == 0:
        return SeamReport()

    cuts = [max(0, min(int(cp * sample_rate), len(matrix) - 1)) for cp in cut_points]
    report = SeamReport()
    prev = 0
    for index, cut_sample in enumerate(cuts):
        next_cut = cuts[index + 1] if index + 1 < len(cuts) else len(matrix)
        left_len = cut_sample - prev
        right_len = next_cut - cut_sample
        evaluation = _evaluate_cut(matrix, sample_rate, cut_sample)
        report.entries.append(
            SeamReportEntry(
                seam_index=index,
                output_time=cut_sample / sample_rate,
                chosen_strategy="measured",
                before_score=evaluation.downmix.score,
                after_score=evaluation.downmix.score,
                left_shift_ms=0.0,
                right_shift_ms=0.0,
                duration_ms=0.0,
                accepted=True,
            )
        )
        prev = cut_sample
    return report


def smooth_rendered_audio(
    rendered_samples: np.ndarray,
    sample_rate: int,
    cut_points: list[float],
    room_tone: np.ndarray,
    crossfade_ms: float = AUDIO_CROSSFADE_MS,
) -> np.ndarray:
    matrix, was_mono = _as_audio_matrix(rendered_samples)
    cut_samples = [max(1, min(int(cp * sample_rate), len(matrix) - 1)) for cp in cut_points]
    boundaries: list[_Boundary] = []
    prev = 0
    for index, cut_sample in enumerate(cut_samples):
        next_cut = cut_samples[index + 1] if index + 1 < len(cut_samples) else len(matrix)
        boundaries.append(
            _Boundary(
                index=index,
                cut_sample=cut_sample,
                left_len=cut_sample - prev,
                right_len=next_cut - cut_sample,
                allow_room_tone=True,
                max_left_blend=cut_sample - prev,
                max_right_blend=next_cut - cut_sample,
            )
        )
        prev = cut_sample

    optimized, _report = optimize_audio_seams(
        rendered_samples,
        sample_rate,
        boundaries,
        room_tone,
        crossfade_options_ms=(8, 16, 24, int(crossfade_ms)),
    )
    optimized_matrix, _ = _as_audio_matrix(optimized)
    return _restore_audio_shape(optimized_matrix, was_mono).astype(np.float32)


def extract_room_tone(
    samples: np.ndarray,
    sample_rate: int,
    words: list[Word] | None = None,
    target_s: float = 0.5,
    max_db: float = -50,
) -> np.ndarray:
    matrix, was_mono = _as_audio_matrix(samples)
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
            pre_end = int(sorted_words[0].start * sample_rate)
            if pre_end >= window:
                gaps.append((0, pre_end))
            post_start = int(sorted_words[-1].end * sample_rate)
            if len(matrix) - post_start >= window:
                gaps.append((post_start, len(matrix)))
    else:
        gaps = [(0, len(matrix))]

    quiet_chunks: list[tuple[float, np.ndarray]] = []
    for gap_start, gap_end in gaps:
        pos = gap_start
        while pos + window <= gap_end:
            chunk = matrix[pos:pos + window]
            db = compute_rms_db(downmix_audio(chunk))
            if db < max_db:
                quiet_chunks.append((db, chunk.copy()))
            pos += window

    quiet_chunks.sort(key=lambda item: item[0])
    if not quiet_chunks:
        zeros = np.zeros((target_samples, matrix.shape[1]), dtype=np.float32)
        return _restore_audio_shape(zeros, was_mono)

    collected = []
    collected_samples = 0
    for _db, chunk in quiet_chunks:
        collected.append(chunk)
        collected_samples += len(chunk)
        if collected_samples >= target_samples:
            break

    room = np.concatenate(collected, axis=0)[:target_samples].astype(np.float32)
    return _restore_audio_shape(room, was_mono)


def assemble_audio_track(
    source_audio: np.ndarray,
    sample_rate: int,
    edit_plan: EditDecisionList,
    room_tone: np.ndarray,
) -> tuple[np.ndarray, SeamReport]:
    source_matrix, was_mono = _as_audio_matrix(source_audio)
    room_matrix, _ = _as_audio_matrix(room_tone)

    pieces: list[_AudioPiece] = []
    assembled_parts: list[np.ndarray] = []
    output_cursor = 0

    for index, decision in enumerate(edit_plan.decisions):
        start_sample = max(0, min(int(decision.source_start * sample_rate), len(source_matrix)))
        end_sample = max(start_sample, min(int(decision.source_end * sample_rate), len(source_matrix)))
        segment_audio = source_matrix[start_sample:end_sample]
        segment_piece = _AudioPiece(
            kind="source",
            samples=segment_audio,
            output_start_sample=output_cursor,
            output_end_sample=output_cursor + len(segment_audio),
            decision_index=index,
        )
        pieces.append(segment_piece)
        assembled_parts.append(segment_audio)
        output_cursor += len(segment_audio)

        gap_duration = decision.pause_after + decision.transition_duration_after
        if gap_duration > 0 and index < len(edit_plan.decisions) - 1:
            next_decision = edit_plan.decisions[index + 1]
            gap_samples = int(gap_duration * sample_rate)
            ambient = _build_ambient_pad(
                source_matrix,
                sample_rate,
                decision.source_end,
                next_decision.source_start,
                gap_samples,
                room_matrix,
            )
            gap_piece = _AudioPiece(
                kind="gap",
                samples=ambient,
                output_start_sample=output_cursor,
                output_end_sample=output_cursor + len(ambient),
                decision_index=None,
            )
            pieces.append(gap_piece)
            assembled_parts.append(ambient)
            output_cursor += len(ambient)

    if not assembled_parts:
        empty = np.zeros((0, source_matrix.shape[1]), dtype=np.float32)
        return _restore_audio_shape(empty, was_mono), SeamReport()

    assembled = np.concatenate(assembled_parts, axis=0).astype(np.float32)
    boundaries: list[_Boundary] = []
    for index in range(len(pieces) - 1):
        left = pieces[index]
        right = pieces[index + 1]
        if left.kind == "gap":
            max_left_blend = left.length
        else:
            left_decision = edit_plan.decisions[left.decision_index]
            max_left_blend = min(
                left.length,
                max(0, int(round(left_decision.trail_padding * sample_rate))),
            )
        if right.kind == "gap":
            max_right_blend = right.length
        else:
            right_decision = edit_plan.decisions[right.decision_index]
            max_right_blend = min(
                right.length,
                max(0, int(round(right_decision.lead_padding * sample_rate))),
            )
        boundary = _Boundary(
            index=index,
            cut_sample=left.output_end_sample,
            left_len=left.length,
            right_len=right.length,
            allow_room_tone=(left.kind == "gap" or right.kind == "gap"),
            max_left_blend=max_left_blend,
            max_right_blend=max_right_blend,
        )
        boundaries.append(boundary)

    optimized, report = optimize_audio_seams(assembled, sample_rate, boundaries, room_matrix)
    optimized_matrix, _ = _as_audio_matrix(optimized)
    return _restore_audio_shape(optimized_matrix, was_mono), report
