from __future__ import annotations

from dataclasses import replace

from ummfiltered.models import (
    DetectionSource,
    FillerSegment,
    PhraseAction,
    PhraseCandidate,
    PhraseDecision,
    PhraseReport,
    PhraseWindow,
    SeamReport,
    Word,
)


MAX_PHRASE_GAP_S = 0.45
MAX_PHRASE_GAP_WORDS = 2
PHRASE_COMPRESSION_FLOOR = 0.72
TARGET_SHORTEN_RATIO = 0.3
MIN_SHORTEN_RETAIN_S = 0.08
MAX_SHORTEN_RETAIN_S = 0.18
KEEP_CONTEXTUAL_DENSITY = 0.82


def _non_filler_words(words: list[Word], fillers: list[FillerSegment]) -> list[Word]:
    filler_ranges = [(filler.start, filler.end) for filler in fillers]
    return [
        word
        for word in words
        if not any(
            start <= (word.start + word.end) / 2.0 <= end
            for start, end in filler_ranges
        )
    ]


def _words_between(
    words: list[Word],
    start: float,
    end: float,
) -> list[Word]:
    return [
        word
        for word in words
        if word.start >= start - 1e-6 and word.end <= end + 1e-6
    ]


def _context_window(
    filler_indices: list[int],
    fillers: list[FillerSegment],
    non_filler_words: list[Word],
) -> tuple[float, float, int]:
    first = fillers[filler_indices[0]]
    last = fillers[filler_indices[-1]]

    preceding = [
        word for word in non_filler_words
        if word.end <= first.start + 1e-6
    ]
    following = [
        word for word in non_filler_words
        if word.start >= last.end - 1e-6
    ]

    start = preceding[-1].start if preceding else first.start
    end = following[0].end if following else last.end
    preserved_count = len(
        _words_between(non_filler_words, first.start, last.end)
    )
    return start, end, preserved_count


def build_phrase_windows(
    fillers: list[FillerSegment],
    words: list[Word],
    *,
    max_gap_s: float = MAX_PHRASE_GAP_S,
    max_gap_words: int = MAX_PHRASE_GAP_WORDS,
) -> list[PhraseWindow]:
    if not fillers:
        return []

    non_filler_words = _non_filler_words(words, fillers)
    windows: list[list[int]] = [[0]]

    for index in range(1, len(fillers)):
        previous = fillers[index - 1]
        current = fillers[index]
        gap_words = _words_between(non_filler_words, previous.end, current.start)
        gap_s = max(0.0, current.start - previous.end)
        if gap_s <= max_gap_s or len(gap_words) <= max_gap_words:
            windows[-1].append(index)
            continue
        windows.append([index])

    return [
        PhraseWindow(
            start=context_start,
            end=context_end,
            filler_indices=filler_indices,
            preserved_word_count=preserved_count,
        )
        for filler_indices in windows
        for context_start, context_end, preserved_count in [
            _context_window(filler_indices, fillers, non_filler_words)
        ]
    ]


def _retained_duration(filler: FillerSegment) -> float:
    raw = (filler.end - filler.start) * TARGET_SHORTEN_RATIO
    return min(MAX_SHORTEN_RETAIN_S, max(MIN_SHORTEN_RETAIN_S, raw))


def _removed_duration(
    filler: FillerSegment,
    decision: PhraseDecision,
) -> float:
    duration = max(0.0, filler.end - filler.start)
    if decision.action == PhraseAction.KEEP:
        return 0.0
    if decision.action == PhraseAction.DELETE:
        return duration
    return max(0.0, duration - decision.retained_duration)


def _window_filler_duration(window: PhraseWindow, fillers: list[FillerSegment]) -> float:
    return sum(
        max(0.0, fillers[index].end - fillers[index].start)
        for index in window.filler_indices
    )


def _candidate_score(
    window: PhraseWindow,
    fillers: list[FillerSegment],
    decisions: list[PhraseDecision],
) -> tuple[float, float]:
    removed_duration = sum(
        _removed_duration(fillers[decision.filler_index], decision)
        for decision in decisions
    )
    window_duration = max(0.001, window.duration)
    compression_ratio = max(0.0, 1.0 - (removed_duration / window_duration))
    filler_density = _window_filler_duration(window, fillers) / window_duration
    kept_fillers = sum(decision.action == PhraseAction.KEEP for decision in decisions)
    shortened_fillers = sum(decision.action == PhraseAction.SHORTEN for decision in decisions)
    score = (
        max(0.0, PHRASE_COMPRESSION_FLOOR - compression_ratio) * 4.0
        + filler_density * 0.7
        + kept_fillers * 0.35
        + shortened_fillers * 0.1
    )
    return compression_ratio, score


def _choose_decisions(window: PhraseWindow, fillers: list[FillerSegment]) -> tuple[list[PhraseDecision], str]:
    decisions = [
        PhraseDecision(
            filler_index=index,
            action=PhraseAction.DELETE,
            reason="default_delete",
        )
        for index in window.filler_indices
    ]
    rejected_reason = ""
    compression_ratio, _score = _candidate_score(window, fillers, decisions)

    contextual_indices = [
        index
        for index in window.filler_indices
        if fillers[index].source == DetectionSource.CONTEXTUAL
    ]
    if contextual_indices and compression_ratio < KEEP_CONTEXTUAL_DENSITY:
        for decision in decisions:
            if decision.filler_index in contextual_indices:
                decision.action = PhraseAction.KEEP
                decision.reason = "contextual_keep"
        compression_ratio, _score = _candidate_score(window, fillers, decisions)
        rejected_reason = "all_delete_rejected_for_contextual_filler"

    if len(window.filler_indices) >= 2 and compression_ratio < KEEP_CONTEXTUAL_DENSITY:
        ordered = sorted(
            decisions,
            key=lambda decision: (
                fillers[decision.filler_index].end - fillers[decision.filler_index].start,
                -decision.filler_index,
            ),
            reverse=True,
        )
        for decision in ordered:
            if decision.action != PhraseAction.DELETE:
                continue
            decision.action = PhraseAction.SHORTEN
            decision.retained_duration = _retained_duration(fillers[decision.filler_index])
            decision.reason = "dense_phrase_shorten"
            compression_ratio, _score = _candidate_score(window, fillers, decisions)
            rejected_reason = rejected_reason or "all_delete_rejected_for_phrase_density"
            if compression_ratio >= KEEP_CONTEXTUAL_DENSITY:
                break

    if (
        len(window.filler_indices) > 1
        and window.preserved_word_count <= 1
        and compression_ratio < PHRASE_COMPRESSION_FLOOR
    ):
        for decision in decisions:
            if decision.action == PhraseAction.DELETE:
                decision.action = PhraseAction.KEEP
                decision.reason = "cadence_keep"
                compression_ratio, _score = _candidate_score(window, fillers, decisions)
                rejected_reason = rejected_reason or "all_delete_rejected_for_cadence"
                if compression_ratio >= PHRASE_COMPRESSION_FLOOR:
                    break

    return decisions, rejected_reason


def plan_phrase_candidates(
    fillers: list[FillerSegment],
    words: list[Word],
) -> list[PhraseCandidate]:
    windows = build_phrase_windows(fillers, words)
    candidates: list[PhraseCandidate] = []
    for window in windows:
        decisions, rejected_reason = _choose_decisions(window, fillers)
        compression_ratio, score = _candidate_score(window, fillers, decisions)
        candidates.append(
            PhraseCandidate(
                window_start=window.start,
                window_end=window.end,
                filler_indices=list(window.filler_indices),
                decisions=decisions,
                compression_ratio=compression_ratio,
                filler_density=_window_filler_duration(window, fillers) / max(0.001, window.duration),
                score=score,
                preserved_word_count=window.preserved_word_count,
                kept_fillers=sum(decision.action == PhraseAction.KEEP for decision in decisions),
                shortened_fillers=sum(decision.action == PhraseAction.SHORTEN for decision in decisions),
                deleted_fillers=sum(decision.action == PhraseAction.DELETE for decision in decisions),
                rejected_reason=rejected_reason,
            )
        )
    return candidates


def _shortened_cut_span(filler: FillerSegment, retained_duration: float) -> FillerSegment:
    filler_duration = max(0.0, filler.end - filler.start)
    retained = min(filler_duration, max(0.0, retained_duration))
    removed_duration = max(0.0, filler_duration - retained)
    if removed_duration <= 0.02:
        return replace(filler)
    trim_each_side = retained / 2.0
    new_start = min(filler.end, filler.start + trim_each_side)
    new_end = max(new_start, filler.end - trim_each_side)
    return FillerSegment(
        start=new_start,
        end=new_end,
        word=filler.word,
        confidence=filler.confidence,
        source=filler.source,
    )


def apply_phrase_candidates(
    fillers: list[FillerSegment],
    candidates: list[PhraseCandidate],
) -> tuple[list[FillerSegment], list[FillerSegment]]:
    decision_by_index: dict[int, PhraseDecision] = {
        decision.filler_index: decision
        for candidate in candidates
        for decision in candidate.decisions
    }
    cut_fillers: list[FillerSegment] = []
    allowed_fillers: list[FillerSegment] = []
    for index, filler in enumerate(fillers):
        decision = decision_by_index.get(index)
        if decision is None or decision.action == PhraseAction.DELETE:
            cut_fillers.append(filler)
            continue
        allowed_fillers.append(filler)
        if decision.action == PhraseAction.SHORTEN:
            shortened = _shortened_cut_span(filler, decision.retained_duration)
            if shortened.end - shortened.start >= 0.02:
                cut_fillers.append(shortened)
    return cut_fillers, allowed_fillers


def build_phrase_report(
    candidates: list[PhraseCandidate],
    edit_plan,
    seam_report: SeamReport | None = None,
) -> PhraseReport:
    entries: list[PhraseCandidate] = []
    for candidate in candidates:
        seam_scores: list[float] = []
        if seam_report is not None:
            for seam_entry in seam_report.entries:
                if seam_entry.seam_index >= len(edit_plan.decisions) - 1:
                    continue
                decision = edit_plan.decisions[seam_entry.seam_index]
                removed_gap = decision.removed_gap_after
                if removed_gap is None:
                    continue
                if removed_gap.end < candidate.window_start or removed_gap.start > candidate.window_end:
                    continue
                seam_scores.append(seam_entry.after_score)
        worst_seam_score = max(seam_scores, default=0.0)
        score = candidate.score + worst_seam_score * 0.4
        entries.append(
            replace(
                candidate,
                worst_seam_score=worst_seam_score,
                score=score,
            )
        )
    return PhraseReport(entries=entries)
