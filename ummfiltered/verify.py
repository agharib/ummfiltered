from __future__ import annotations

from pathlib import Path

import numpy as np

from ummfiltered.audio import analyze_audio_seams, compute_rms_db, extract_audio_pcm, find_silence_boundaries, protect_adjacent_words
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
from ummfiltered.edit_plan import boundary_points, build_edit_decision_list, build_segment_map as build_segment_map_from_plan, cut_points as cut_points_from_plan, map_output_to_original as map_output_to_original_from_plan
from ummfiltered.models import CutAdjustment, EditDecisionList, FillerSegment, Segment, VerificationResult, Word
from ummfiltered.transcribe import transcribe


SegmentMap = list[tuple[float, float, float]]
SEAM_ALERT_THRESHOLD = 2.8


def build_segment_map(
    segments: list[Segment],
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> SegmentMap:
    edit_plan = build_edit_decision_list(
        segments,
        pause_overrides=pause_overrides,
        transition_durations=transition_durations,
    )
    return build_segment_map_from_plan(edit_plan)


def map_output_to_original(output_time: float, seg_map: SegmentMap) -> float:
    for index, (orig_start, orig_end, out_offset) in enumerate(seg_map):
        seg_duration = orig_end - orig_start
        out_end = out_offset + seg_duration
        if output_time < out_offset:
            return orig_start
        is_last = index == len(seg_map) - 1
        if output_time < out_end or (is_last and output_time <= out_end):
            return orig_start + (output_time - out_offset)
    if not seg_map:
        return 0.0
    _last_start, last_end, _last_offset = seg_map[-1]
    return last_end


def _is_near_known_filler(
    output_filler: FillerSegment,
    original_fillers: list[FillerSegment],
    edit_plan: EditDecisionList,
    tolerance_s: float = 1.0,
) -> bool:
    orig_time = map_output_to_original_from_plan(output_filler.start, edit_plan)
    for original_filler in original_fillers:
        if abs(orig_time - original_filler.start) < tolerance_s and output_filler.word == original_filler.word:
            return True
    return False


def _remap_filler_to_original(
    output_filler: FillerSegment,
    edit_plan: EditDecisionList,
) -> FillerSegment:
    original_start = map_output_to_original_from_plan(output_filler.start, edit_plan)
    original_end = map_output_to_original_from_plan(output_filler.end, edit_plan)
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
    edit_plan: EditDecisionList | None = None,
) -> tuple[list[FillerSegment], list[FillerSegment]]:
    if edit_plan is None:
        edit_plan = build_edit_decision_list(
            segments,
            pause_overrides=pause_overrides,
            transition_durations=transition_durations,
        )

    detected = detect_fillers(output_words, aggressive=aggressive)
    detected = filter_fillers_by_context(detected, output_words, min_confidence=min_confidence)

    remaining: list[FillerSegment] = []
    new: list[FillerSegment] = []
    for filler in detected:
        original_filler = _remap_filler_to_original(filler, edit_plan)
        if _is_near_known_filler(filler, original_fillers, edit_plan):
            remaining.append(original_filler)
        else:
            new.append(original_filler)
    return remaining, new


def _build_expected_words(
    original_words: list[Word],
    cut_fillers: list[FillerSegment],
) -> list[Word]:
    def _removed_by_cut(word: Word, start: float, end: float) -> bool:
        overlap = max(0.0, min(word.end, end) - max(word.start, start))
        duration = max(1e-6, word.end - word.start)
        midpoint = (word.start + word.end) / 2.0
        return overlap >= duration * 0.5 or start <= midpoint <= end

    filler_ranges = [(filler.start, filler.end) for filler in cut_fillers]
    expected: list[Word] = []
    for word in original_words:
        is_filler = any(_removed_by_cut(word, start, end) for start, end in filler_ranges)
        if not is_filler:
            expected.append(word)
    return expected


def build_reference_contract(
    original_words: list[Word],
    cut_fillers: list[FillerSegment],
) -> list[str]:
    return [word.text for word in _build_expected_words(original_words, cut_fillers)]


def _fuzzy_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) < 2 or len(b) < 2:
        return False
    if a in b or b in a:
        return True
    return False


def _lcs_alignment(expected: list[str], actual: list[str]) -> set[int]:
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
    for filler in cut_fillers:
        if abs(word.start - filler.end) < tolerance or abs(word.end - filler.start) < tolerance:
            return True
    return False


def _max_missing_run(expected_count: int, matched: set[int]) -> int:
    max_run = 0
    current_run = 0
    for index in range(expected_count):
        if index in matched:
            current_run = 0
            continue
        current_run += 1
        max_run = max(max_run, current_run)
    return max_run


def _missing_expected_tokens(expected: list[Word], matched: set[int]) -> list[str]:
    return [
        expected[index].text
        for index in range(len(expected))
        if index not in matched
    ]


def _word_integrity_stats(
    original_words: list[Word],
    output_words: list[Word],
    cut_fillers: list[FillerSegment],
    edit_plan: EditDecisionList,
) -> tuple[list[Word], list[tuple[Word, int]], float, int]:
    expected = _build_expected_words(original_words, cut_fillers)
    expected_texts = [word.text for word in expected]
    actual_texts = [word.text for word in output_words]

    matched = _lcs_alignment(expected_texts, actual_texts)
    lost = [
        expected[index]
        for index in range(len(expected))
        if index not in matched and _is_near_cut(expected[index], cut_fillers)
    ]

    cut_boundaries = boundary_points(edit_plan)
    damaged: list[tuple[Word, int]] = []
    for output_word in output_words:
        for cut_index, boundary in enumerate(cut_boundaries):
            if abs(output_word.start - boundary) < 0.3 or abs(output_word.end - boundary) < 0.3:
                orig_time = map_output_to_original_from_plan(output_word.start, edit_plan)
                has_exact = False
                for expected_word in expected:
                    if abs(expected_word.start - orig_time) < 0.5 and expected_word.text == output_word.text:
                        has_exact = True
                        break
                if has_exact:
                    break
                for expected_word in expected:
                    if abs(expected_word.start - orig_time) < 0.5 and expected_word.text != output_word.text:
                        is_truncated = (
                            len(output_word.text) >= 2
                            and len(expected_word.text) >= 2
                            and (
                                expected_word.text.startswith(output_word.text)
                                or expected_word.text.endswith(output_word.text)
                                or output_word.text.startswith(expected_word.text)
                                or output_word.text.endswith(expected_word.text)
                            )
                        )
                        if is_truncated:
                            damaged.append((expected_word, cut_index))
                        break

    recall = len(matched) / max(1, len(expected))
    max_missing_run = _max_missing_run(len(expected), matched)
    return lost, damaged, recall, max_missing_run


def check_word_integrity(
    original_words: list[Word],
    output_words: list[Word],
    cut_fillers: list[FillerSegment],
    segments: list[Segment],
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> tuple[list[Word], list[tuple[Word, int]]]:
    edit_plan = build_edit_decision_list(
        segments,
        pause_overrides=pause_overrides,
        transition_durations=transition_durations,
    )
    lost, damaged, _recall, _max_missing = _word_integrity_stats(
        original_words,
        output_words,
        cut_fillers,
        edit_plan,
    )
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
    mono = samples if samples.ndim == 1 else np.mean(samples, axis=1)
    window_samples = int(sample_rate * WINDOW_MS / 1000.0)
    discontinuities: list[tuple[float, float]] = []

    for cut_point in cut_points:
        cut_sample = int(cut_point * sample_rate)
        before_start = max(0, cut_sample - window_samples)
        after_end = min(len(mono), cut_sample + window_samples)
        if cut_sample - before_start < window_samples // 2:
            continue
        if after_end - cut_sample < window_samples // 2:
            continue

        before = mono[before_start:cut_sample]
        after = mono[cut_sample:after_end]
        db_before = compute_rms_db(before)
        db_after = compute_rms_db(after)
        db_diff = abs(db_after - db_before)
        if db_diff > volume_threshold_db:
            discontinuities.append((cut_point, db_diff))
            continue

        mean_before = float(np.mean(before))
        mean_after = float(np.mean(after))
        dc_jump = abs(mean_after - mean_before)
        if dc_jump > dc_threshold:
            discontinuities.append((cut_point, db_diff))
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
    edit_plan: EditDecisionList | None = None,
    reference_fillers: list[FillerSegment] | None = None,
) -> VerificationResult:
    if edit_plan is None:
        edit_plan = build_edit_decision_list(
            segments,
            pause_overrides=pause_overrides,
            transition_durations=transition_durations,
        )

    output_words = transcribe(str(output_path), model_size=model_size)
    remaining, new = check_remaining_fillers(
        output_words,
        original_fillers,
        segments,
        aggressive=aggressive,
        min_confidence=min_confidence,
        pause_overrides=pause_overrides,
        transition_durations=transition_durations,
        edit_plan=edit_plan,
    )

    lost, damaged, recall, max_missing_run = _word_integrity_stats(
        original_words,
        output_words,
        reference_fillers or original_fillers,
        edit_plan,
    )
    expected_words = _build_expected_words(original_words, reference_fillers or original_fillers)
    matched_tokens = _lcs_alignment(
        [word.text for word in expected_words],
        [word.text for word in output_words],
    )
    missing_words = [
        expected_words[index]
        for index in range(len(expected_words))
        if index not in matched_tokens
    ]
    missing_tokens = [word.text for word in missing_words]
    contract_tokens = [word.text for word in expected_words]
    contract_intact = (
        not missing_tokens
        and not lost
        and not damaged
        and recall >= 0.999
        and max_missing_run == 0
    )

    samples, sample_rate = extract_audio_pcm(output_path)
    seam_report = analyze_audio_seams(samples, sample_rate, boundary_points(edit_plan))
    audio_discontinuities = [
        (entry.output_time, entry.after_score)
        for entry in seam_report.entries
        if entry.after_score >= SEAM_ALERT_THRESHOLD
    ]
    audio_discontinuities.extend(
        check_audio_smoothness(samples, sample_rate, boundary_points(edit_plan))
    )

    return VerificationResult(
        remaining_fillers=remaining,
        new_fillers=new,
        lost_words=lost,
        damaged_words=damaged,
        audio_discontinuities=audio_discontinuities,
        preserved_word_recall=recall,
        max_missing_run=max_missing_run,
        seam_report=seam_report,
        missing_words=missing_words,
        output_words=output_words,
        contract_tokens=contract_tokens,
        missing_tokens=missing_tokens,
        contract_intact=contract_intact,
    )


EXPANSION_INCREMENT_MS = 150.0
EXPANSION_DECREMENT_MS = 100.0
CROSSFADE_INCREMENT_MS = 40.0
LOST_WORD_DECREMENT_MS = 120.0


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

    narrowed_for_lost_words: set[int] = set()
    for lost_word in result.lost_words:
        adjustment_key = _find_adjustment_for_lost_word(adjustments, lost_word)
        if adjustment_key is None or adjustment_key in prioritized_keys:
            continue
        if adjustment_key in narrowed_for_lost_words:
            continue
        adjustments[adjustment_key].expansion_ms = max(
            80.0,
            adjustments[adjustment_key].expansion_ms - LOST_WORD_DECREMENT_MS,
        )
        narrowed_for_lost_words.add(adjustment_key)

    for damaged_word, cut_idx in result.damaged_words:
        adjustment_key = _adjustment_key_for_cut_index(adjustments, cut_idx)
        if adjustment_key is not None:
            if adjustment_key in prioritized_keys:
                continue
            adjustments[adjustment_key].expansion_ms = max(
                100.0,
                adjustments[adjustment_key].expansion_ms - EXPANSION_DECREMENT_MS,
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
            abs(adjustment.filler.start - filler.start) + abs(adjustment.filler.end - filler.end),
            key,
        )
        for key, adjustment in adjustments.items()
        if adjustment.filler.word == filler.word
    ]
    if not candidates:
        return None
    return min(candidates)[1]


def _find_adjustment_for_lost_word(
    adjustments: dict[int, CutAdjustment],
    lost_word: Word,
) -> int | None:
    candidates = []
    for key, adjustment in adjustments.items():
        filler = adjustment.filler
        distance = min(
            abs(lost_word.end - filler.start),
            abs(lost_word.start - filler.end),
        )
        candidates.append((distance, key))
    if not candidates:
        return None
    best_distance, best_key = min(candidates)
    if best_distance > 0.75:
        return None
    return best_key


def _ordered_adjustment_keys(adjustments: dict[int, CutAdjustment]) -> list[int]:
    return [
        key
        for key, adjustment in sorted(
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
    edit_plan = build_edit_decision_list(
        segments,
        pause_overrides=pause_overrides,
        transition_durations=transition_durations,
    )
    return cut_points_from_plan(edit_plan)


def _find_cut_index(cut_points: dict[int, float], cut_time: float) -> int | None:
    if not cut_points:
        return None
    return min(cut_points, key=lambda index: abs(cut_points[index] - cut_time))


def rebuild_cuts(
    adjustments: dict[int, CutAdjustment],
    samples: np.ndarray,
    sample_rate: int,
    non_filler_words: list[Word] | None = None,
) -> tuple[list[FillerSegment], dict[int, float]]:
    active = {key: adjustment for key, adjustment in adjustments.items() if not adjustment.skip}
    expanded_fillers: list[FillerSegment] = []
    crossfade_map: dict[int, float] = {}

    audio_duration = len(samples) / sample_rate
    for index, adjustment in sorted(
        active.items(),
        key=lambda item: (item[1].filler.start, item[1].filler.end, item[0]),
    ):
        margin_start_s, margin_end_s = _filler_margin_seconds(adjustment.filler)
        filler_start = max(0.0, min(adjustment.filler.start - margin_start_s, audio_duration))
        filler_end = max(filler_start, min(adjustment.filler.end + margin_end_s, audio_duration))
        if filler_start >= filler_end:
            continue
        new_start, new_end = find_silence_boundaries(
            samples,
            sample_rate,
            filler_start,
            filler_end,
            threshold_db=SILENCE_THRESHOLD_DB,
            max_expansion_ms=adjustment.expansion_ms,
        )
        if non_filler_words is not None:
            protected = protect_adjacent_words(
                new_start,
                new_end,
                non_filler_words,
                samples,
                sample_rate,
            )
            if protected is None:
                continue
            new_start, new_end = protected
        expanded_fillers.append(
            FillerSegment(
                start=new_start,
                end=new_end,
                word=adjustment.filler.word,
                confidence=adjustment.filler.confidence,
                source=adjustment.filler.source,
            )
        )
        seg_idx = len(expanded_fillers) - 1
        crossfade_s = adjustment.crossfade_ms / 1000.0
        crossfade_map[seg_idx] = max(crossfade_map.get(seg_idx, 0.0), crossfade_s)
        crossfade_map[seg_idx + 1] = max(crossfade_map.get(seg_idx + 1, 0.0), crossfade_s)

    return expanded_fillers, crossfade_map
