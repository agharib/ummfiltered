from pathlib import Path

import numpy as np

from ummfiltered.audio import compute_rms_db, extract_audio_pcm, find_silence_boundaries, protect_adjacent_words
from ummfiltered.config import FILLER_MARGIN_END_MS, FILLER_MARGIN_START_MS, MAX_EXPANSION_MS, SILENCE_THRESHOLD_DB
from ummfiltered.detect import detect_fillers, filter_fillers_by_context
from ummfiltered.models import (
    CutAdjustment, FillerSegment, Segment, VerificationResult, Word,
)
from ummfiltered.transcribe import transcribe


SegmentMap = list[tuple[float, float, float]]


def build_segment_map(segments: list[Segment]) -> SegmentMap:
    seg_map: SegmentMap = []
    output_offset = 0.0
    for seg in segments:
        seg_map.append((seg.start, seg.end, output_offset))
        output_offset += seg.end - seg.start
    return seg_map


def map_output_to_original(output_time: float, seg_map: SegmentMap) -> float:
    for i, (orig_start, orig_end, out_offset) in enumerate(seg_map):
        seg_duration = orig_end - orig_start
        is_last = i == len(seg_map) - 1
        if is_last:
            if output_time <= out_offset + seg_duration:
                return orig_start + (output_time - out_offset)
        else:
            if output_time < out_offset + seg_duration:
                return orig_start + (output_time - out_offset)
    last_orig_start, last_orig_end, last_out_offset = seg_map[-1]
    return last_orig_end


def _is_near_known_filler(
    output_filler: FillerSegment,
    original_fillers: list[FillerSegment],
    seg_map: SegmentMap,
    tolerance_s: float = 1.0,
) -> bool:
    orig_time = map_output_to_original(output_filler.start, seg_map)
    for of in original_fillers:
        if abs(orig_time - of.start) < tolerance_s and output_filler.word == of.word:
            return True
    return False


def check_remaining_fillers(
    output_words: list[Word],
    original_fillers: list[FillerSegment],
    segments: list[Segment],
    aggressive: bool = False,
    min_confidence: float = 0.15,
) -> tuple[list[FillerSegment], list[FillerSegment]]:
    seg_map = build_segment_map(segments)
    detected = detect_fillers(output_words, aggressive=aggressive)
    detected = filter_fillers_by_context(detected, output_words, min_confidence=min_confidence)

    remaining: list[FillerSegment] = []
    new: list[FillerSegment] = []

    for f in detected:
        if _is_near_known_filler(f, original_fillers, seg_map):
            remaining.append(f)
        else:
            new.append(f)

    return remaining, new


def _build_expected_words(
    original_words: list[Word],
    cut_fillers: list[FillerSegment],
) -> list[Word]:
    filler_ranges = [(f.start, f.end) for f in cut_fillers]
    expected: list[Word] = []
    for w in original_words:
        is_filler = any(
            f_start <= w.start and w.end <= f_end
            for f_start, f_end in filler_ranges
        )
        if not is_filler:
            expected.append(w)
    return expected


def _fuzzy_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) < 2 or len(b) < 2:
        return False
    if a in b or b in a:
        return True
    return False


def _lcs_alignment(
    expected: list[str], actual: list[str],
) -> set[int]:
    m, n = len(expected), len(actual)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if _fuzzy_match(expected[i - 1], actual[j - 1]):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    matched_expected: set[int] = set()
    i, j = m, n
    while i > 0 and j > 0:
        if _fuzzy_match(expected[i - 1], actual[j - 1]):
            matched_expected.add(i - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return matched_expected


def _is_near_cut(word: Word, cut_fillers: list[FillerSegment], tolerance: float = 0.5) -> bool:
    for f in cut_fillers:
        if abs(word.start - f.end) < tolerance or abs(word.end - f.start) < tolerance:
            return True
    return False


def check_word_integrity(
    original_words: list[Word],
    output_words: list[Word],
    cut_fillers: list[FillerSegment],
    segments: list[Segment],
) -> tuple[list[Word], list[tuple[Word, int]]]:
    expected = _build_expected_words(original_words, cut_fillers)
    expected_texts = [w.text for w in expected]
    actual_texts = [w.text for w in output_words]

    matched = _lcs_alignment(expected_texts, actual_texts)
    lost: list[Word] = [
        expected[i] for i in range(len(expected))
        if i not in matched and _is_near_cut(expected[i], cut_fillers)
    ]

    seg_map = build_segment_map(segments)
    cut_boundaries: list[float] = []
    output_offset = 0.0
    for i, seg in enumerate(segments):
        if i > 0:
            cut_boundaries.append(output_offset)
        output_offset += seg.end - seg.start

    damaged: list[tuple[Word, int]] = []
    for ow in output_words:
        for ci, boundary in enumerate(cut_boundaries):
            if abs(ow.start - boundary) < 0.3 or abs(ow.end - boundary) < 0.3:
                orig_time = map_output_to_original(ow.start, seg_map)
                has_exact = False
                for ew in expected:
                    if abs(ew.start - orig_time) < 0.5 and ew.text == ow.text:
                        has_exact = True
                        break
                if has_exact:
                    break
                for ew in expected:
                    if abs(ew.start - orig_time) < 0.5 and ew.text != ow.text:
                        is_truncated = (
                            len(ow.text) >= 2
                            and len(ew.text) >= 2
                            and (ew.text.startswith(ow.text) or ew.text.endswith(ow.text)
                                 or ow.text.startswith(ew.text) or ow.text.endswith(ew.text))
                        )
                        if is_truncated:
                            damaged.append((ew, ci))
                        break

    return lost, damaged


VOLUME_JUMP_THRESHOLD_DB = 6.0
DC_OFFSET_THRESHOLD = 0.05
WINDOW_MS = 50


def check_audio_smoothness(
    samples: np.ndarray,
    sample_rate: int,
    cut_points: list[float],
    volume_threshold_db: float = VOLUME_JUMP_THRESHOLD_DB,
    dc_threshold: float = DC_OFFSET_THRESHOLD,
) -> list[tuple[float, float]]:
    window_samples = int(sample_rate * WINDOW_MS / 1000.0)
    discontinuities: list[tuple[float, float]] = []

    for cp in cut_points:
        cp_sample = int(cp * sample_rate)
        before_start = max(0, cp_sample - window_samples)
        after_end = min(len(samples), cp_sample + window_samples)

        if cp_sample - before_start < window_samples // 2:
            continue
        if after_end - cp_sample < window_samples // 2:
            continue

        before = samples[before_start:cp_sample]
        after = samples[cp_sample:after_end]

        db_before = compute_rms_db(before)
        db_after = compute_rms_db(after)
        db_diff = abs(db_after - db_before)

        if db_diff > volume_threshold_db:
            discontinuities.append((cp, db_diff))
            continue

        mean_before = float(np.mean(before))
        mean_after = float(np.mean(after))
        dc_jump = abs(mean_after - mean_before)

        if dc_jump > dc_threshold:
            discontinuities.append((cp, db_diff))

    return discontinuities


def verify_output(
    output_path: Path,
    original_words: list[Word],
    original_fillers: list[FillerSegment],
    segments: list[Segment],
    model_size: str = "medium",
    aggressive: bool = False,
    min_confidence: float = 0.15,
) -> VerificationResult:
    output_words = transcribe(str(output_path), model_size=model_size)

    remaining, new = check_remaining_fillers(
        output_words, original_fillers, segments,
        aggressive=aggressive, min_confidence=min_confidence,
    )

    lost, damaged = check_word_integrity(
        original_words, output_words, original_fillers, segments,
    )

    samples, sample_rate = extract_audio_pcm(output_path)
    cut_points: list[float] = []
    output_offset = 0.0
    for i, seg in enumerate(segments):
        if i > 0:
            cut_points.append(output_offset)
        output_offset += seg.end - seg.start

    audio_discs = check_audio_smoothness(samples, sample_rate, cut_points)

    return VerificationResult(
        remaining_fillers=remaining,
        new_fillers=new,
        lost_words=lost,
        damaged_words=damaged,
        audio_discontinuities=audio_discs,
    )


EXPANSION_INCREMENT_MS = 150.0
EXPANSION_DECREMENT_MS = 100.0
CROSSFADE_INCREMENT_MS = 40.0


def apply_adjustments(
    adjustments: dict[int, CutAdjustment],
    result: VerificationResult,
    segments: list[Segment] | None = None,
) -> None:
    for dw, cut_idx in result.damaged_words:
        if cut_idx < len(adjustments):
            adjustments[cut_idx].expansion_ms = max(
                100.0, adjustments[cut_idx].expansion_ms - EXPANSION_DECREMENT_MS
            )



def rebuild_cuts(
    adjustments: dict[int, CutAdjustment],
    samples: np.ndarray,
    sample_rate: int,
    non_filler_words: list[Word] | None = None,
) -> tuple[list[FillerSegment], dict[int, float]]:
    active = {k: v for k, v in adjustments.items() if not v.skip}

    expanded_fillers: list[FillerSegment] = []
    crossfade_map: dict[int, float] = {}

    audio_duration = len(samples) / sample_rate
    margin_start_s = FILLER_MARGIN_START_MS / 1000.0
    margin_end_s = FILLER_MARGIN_END_MS / 1000.0
    for idx, adj in sorted(active.items()):
        filler_start = max(0.0, min(adj.filler.start - margin_start_s, audio_duration))
        filler_end = max(filler_start, min(adj.filler.end + margin_end_s, audio_duration))
        if filler_start >= filler_end:
            continue
        new_start, new_end = find_silence_boundaries(
            samples, sample_rate, filler_start, filler_end,
            threshold_db=SILENCE_THRESHOLD_DB,
            max_expansion_ms=adj.expansion_ms,
        )
        if non_filler_words is not None:
            protected = protect_adjacent_words(
                new_start, new_end, non_filler_words, samples, sample_rate,
            )
            if protected is None:
                continue
            new_start, new_end = protected
        expanded_fillers.append(FillerSegment(
            start=new_start, end=new_end,
            word=adj.filler.word, confidence=adj.filler.confidence,
            source=adj.filler.source,
        ))
        crossfade_map[len(expanded_fillers) - 1] = adj.crossfade_ms

    return expanded_fillers, crossfade_map
