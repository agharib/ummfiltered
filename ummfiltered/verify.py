from pathlib import Path

import numpy as np

from ummfiltered.audio import compute_rms_db, extract_audio_pcm, find_silence_boundaries, protect_adjacent_words
from ummfiltered.config import (
    CROSSFADE_MS,
    FILLER_MARGIN_END_MS,
    FILLER_MARGIN_START_MS,
    MAX_EXPANSION_MS,
    PHRASE_MARGIN_END_BONUS_MS,
    PHRASE_MARGIN_START_BONUS_MS,
    SILENCE_THRESHOLD_DB,
)
from ummfiltered.detect import detect_fillers, filter_fillers_by_context
from ummfiltered.models import (
    CutAdjustment, FillerSegment, Segment, VerificationResult, Word,
)
from ummfiltered.transcribe import transcribe


SegmentMap = list[tuple[float, float, float]]


def build_segment_map(
    segments: list[Segment],
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> SegmentMap:
    seg_map: SegmentMap = []
    output_offset = 0.0
    for i, seg in enumerate(segments):
        seg_map.append((seg.start, seg.end, output_offset))
        output_offset += seg.end - seg.start
        output_offset += pause_overrides.get(i, 0.0) if pause_overrides else 0.0
        output_offset += transition_durations.get(i + 1, 0.0) if transition_durations else 0.0
    return seg_map


def map_output_to_original(output_time: float, seg_map: SegmentMap) -> float:
    for i, (orig_start, orig_end, out_offset) in enumerate(seg_map):
        seg_duration = orig_end - orig_start
        out_end = out_offset + seg_duration
        if output_time < out_offset:
            return orig_start
        is_last = i == len(seg_map) - 1
        if output_time < out_end or (is_last and output_time <= out_end):
            return orig_start + (output_time - out_offset)
    _last_orig_start, last_orig_end, _last_out_offset = seg_map[-1]
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


def _remap_filler_to_original(
    output_filler: FillerSegment,
    seg_map: SegmentMap,
) -> FillerSegment:
    original_start = map_output_to_original(output_filler.start, seg_map)
    original_end = map_output_to_original(output_filler.end, seg_map)
    if original_end < original_start:
        original_end = original_start
    return FillerSegment(
        start=original_start,
        end=original_end,
        word=output_filler.word,
        confidence=output_filler.confidence,
        source=output_filler.source,
    )


def check_remaining_fillers(
    output_words: list[Word],
    original_fillers: list[FillerSegment],
    segments: list[Segment],
    aggressive: bool = False,
    min_confidence: float = 0.15,
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> tuple[list[FillerSegment], list[FillerSegment]]:
    seg_map = build_segment_map(
        segments,
        pause_overrides=pause_overrides,
        transition_durations=transition_durations,
    )
    detected = detect_fillers(output_words, aggressive=aggressive)
    detected = filter_fillers_by_context(detected, output_words, min_confidence=min_confidence)

    remaining: list[FillerSegment] = []
    new: list[FillerSegment] = []

    for f in detected:
        original_filler = _remap_filler_to_original(f, seg_map)
        if _is_near_known_filler(f, original_fillers, seg_map):
            remaining.append(original_filler)
        else:
            new.append(original_filler)

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
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> tuple[list[Word], list[tuple[Word, int]]]:
    expected = _build_expected_words(original_words, cut_fillers)
    expected_texts = [w.text for w in expected]
    actual_texts = [w.text for w in output_words]

    matched = _lcs_alignment(expected_texts, actual_texts)
    lost: list[Word] = [
        expected[i] for i in range(len(expected))
        if i not in matched and _is_near_cut(expected[i], cut_fillers)
    ]

    seg_map = build_segment_map(
        segments,
        pause_overrides=pause_overrides,
        transition_durations=transition_durations,
    )
    cut_boundaries: list[float] = []
    output_offset = 0.0
    for i, seg in enumerate(segments):
        output_offset += seg.end - seg.start
        output_offset += pause_overrides.get(i, 0.0) if pause_overrides else 0.0
        if i < len(segments) - 1:
            cut_boundaries.append(output_offset)
            transition_duration = transition_durations.get(i + 1, 0.0) if transition_durations else 0.0
            if transition_duration > 0:
                output_offset += transition_duration
                cut_boundaries.append(output_offset)

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
    model_size: str = "large",
    aggressive: bool = False,
    min_confidence: float = 0.15,
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> VerificationResult:
    output_words = transcribe(str(output_path), model_size=model_size)

    remaining, new = check_remaining_fillers(
        output_words, original_fillers, segments,
        aggressive=aggressive,
        min_confidence=min_confidence,
        pause_overrides=pause_overrides,
        transition_durations=transition_durations,
    )

    lost, damaged = check_word_integrity(
        original_words,
        output_words,
        original_fillers,
        segments,
        pause_overrides=pause_overrides,
        transition_durations=transition_durations,
    )

    samples, sample_rate = extract_audio_pcm(output_path)
    cut_points = list(_build_cut_points(
        segments,
        pause_overrides=pause_overrides,
        transition_durations=transition_durations,
    ).values())

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


def _expansion_increment_for_filler(filler: FillerSegment) -> float:
    increment = EXPANSION_INCREMENT_MS
    if " " in filler.word:
        increment += 100.0
    return increment


def _filler_margin_seconds(filler: FillerSegment) -> tuple[float, float]:
    start_ms = FILLER_MARGIN_START_MS
    end_ms = FILLER_MARGIN_END_MS
    if " " in filler.word:
        start_ms += PHRASE_MARGIN_START_BONUS_MS
        end_ms += PHRASE_MARGIN_END_BONUS_MS
    return start_ms / 1000.0, end_ms / 1000.0


def apply_adjustments(
    adjustments: dict[int, CutAdjustment],
    result: VerificationResult,
    segments: list[Segment] | None = None,
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> None:
    prioritized_keys: set[int] = set()
    for filler in result.remaining_fillers:
        match_key = _find_adjustment_for_filler(adjustments, filler)
        if match_key is not None:
            prioritized_keys.add(match_key)
            adjustments[match_key].filler = FillerSegment(
                start=min(adjustments[match_key].filler.start, filler.start),
                end=max(adjustments[match_key].filler.end, filler.end),
                word=adjustments[match_key].filler.word,
                confidence=max(adjustments[match_key].filler.confidence, filler.confidence),
                source=adjustments[match_key].filler.source,
            )
            adjustments[match_key].expansion_ms += _expansion_increment_for_filler(filler)

    for filler in result.new_fillers:
        next_key = max(adjustments.keys(), default=-1) + 1
        adjustments[next_key] = CutAdjustment(
            filler=filler,
            expansion_ms=MAX_EXPANSION_MS,
            crossfade_ms=CROSSFADE_MS,
        )

    for dw, cut_idx in result.damaged_words:
        adjustment_key = _adjustment_key_for_cut_index(adjustments, cut_idx)
        if adjustment_key is not None:
            if adjustment_key in prioritized_keys:
                continue
            adjustments[adjustment_key].expansion_ms = max(
                100.0, adjustments[adjustment_key].expansion_ms - EXPANSION_DECREMENT_MS
            )

    if segments:
        cut_points = _build_cut_points(
            segments,
            pause_overrides=pause_overrides,
            transition_durations=transition_durations,
        )
        for cut_time, _severity in result.audio_discontinuities:
            cut_idx = _find_cut_index(cut_points, cut_time)
            adjustment_key = _adjustment_key_for_cut_index(adjustments, cut_idx)
            if adjustment_key is not None:
                adjustments[adjustment_key].crossfade_ms += CROSSFADE_INCREMENT_MS


def _find_adjustment_for_filler(
    adjustments: dict[int, CutAdjustment],
    filler: FillerSegment,
) -> int | None:
    candidates = [
        (
            abs(adj.filler.start - filler.start) + abs(adj.filler.end - filler.end),
            key,
        )
        for key, adj in adjustments.items()
        if adj.filler.word == filler.word
    ]
    if not candidates:
        return None
    return min(candidates)[1]


def _ordered_adjustment_keys(adjustments: dict[int, CutAdjustment]) -> list[int]:
    return [
        key for key, adjustment in sorted(
            adjustments.items(),
            key=lambda item: (item[1].filler.start, item[1].filler.end, item[0]),
        )
        if not adjustment.skip
    ]


def _adjustment_key_for_cut_index(
    adjustments: dict[int, CutAdjustment],
    cut_idx: int | None,
) -> int | None:
    if cut_idx is None:
        return None
    ordered_keys = _ordered_adjustment_keys(adjustments)
    if cut_idx < 0 or cut_idx >= len(ordered_keys):
        return None
    return ordered_keys[cut_idx]

def _build_cut_points(
    segments: list[Segment],
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> dict[int, float]:
    cut_points: dict[int, float] = {}
    output_offset = 0.0
    for i, seg in enumerate(segments):
        output_offset += seg.end - seg.start
        output_offset += pause_overrides.get(i, 0.0) if pause_overrides else 0.0
        if i < len(segments) - 1:
            cut_points[i] = output_offset
            output_offset += transition_durations.get(i + 1, 0.0) if transition_durations else 0.0
    return cut_points


def _find_cut_index(cut_points: dict[int, float], cut_time: float) -> int | None:
    if not cut_points:
        return None
    return min(cut_points, key=lambda idx: abs(cut_points[idx] - cut_time))



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
    for idx, adj in sorted(
        active.items(),
        key=lambda item: (item[1].filler.start, item[1].filler.end, item[0]),
    ):
        margin_start_s, margin_end_s = _filler_margin_seconds(adj.filler)
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
        seg_idx = len(expanded_fillers) - 1
        crossfade_s = adj.crossfade_ms / 1000.0
        crossfade_map[seg_idx] = max(crossfade_map.get(seg_idx, 0.0), crossfade_s)
        crossfade_map[seg_idx + 1] = max(crossfade_map.get(seg_idx + 1, 0.0), crossfade_s)

    return expanded_fillers, crossfade_map
