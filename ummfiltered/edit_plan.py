from __future__ import annotations

from ummfiltered.models import EditDecision, EditDecisionList, RemovedGap, Segment, Word


def _segment_paddings(segment: Segment, preserved_words: list[Word] | None) -> tuple[float, float]:
    if not preserved_words:
        duration = max(0.0, segment.end - segment.start)
        return duration, duration

    segment_words = [
        word
        for word in preserved_words
        if word.end > segment.start and word.start < segment.end
    ]
    if not segment_words:
        duration = max(0.0, segment.end - segment.start)
        return duration, duration

    first_word = min(segment_words, key=lambda word: (word.start, word.end))
    last_word = max(segment_words, key=lambda word: (word.end, word.start))
    lead_padding = max(0.0, first_word.start - segment.start)
    trail_padding = max(0.0, segment.end - last_word.end)
    return lead_padding, trail_padding


def build_edit_decision_list(
    segments: list[Segment],
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
    preserved_words: list[Word] | None = None,
    contract_tokens: list[str] | None = None,
) -> EditDecisionList:
    decisions: list[EditDecision] = []
    output_cursor = 0.0
    for index, segment in enumerate(segments):
        duration = max(0.0, segment.end - segment.start)
        output_end = output_cursor + duration
        lead_padding, trail_padding = _segment_paddings(segment, preserved_words)
        removed_gap_after = None
        if index < len(segments) - 1:
            removed_gap_after = RemovedGap(
                start=segment.end,
                end=segments[index + 1].start,
            )
        decision = EditDecision(
            index=index,
            source_start=segment.start,
            source_end=segment.end,
            output_start=output_cursor,
            output_end=output_end,
            transition_type=segment.transition_type,
            lead_padding=lead_padding,
            trail_padding=trail_padding,
            pause_after=pause_overrides.get(index, 0.0) if pause_overrides else 0.0,
            transition_duration_after=(
                transition_durations.get(index + 1, 0.0) if transition_durations else 0.0
            ),
            removed_gap_after=removed_gap_after,
        )
        decisions.append(decision)
        output_cursor = output_end + decision.pause_after + decision.transition_duration_after
    return EditDecisionList(
        decisions=decisions,
        total_output_duration=output_cursor,
        contract_tokens=contract_tokens or [word.text for word in (preserved_words or [])],
    )


def edit_plan_to_segments(edit_plan: EditDecisionList) -> list[Segment]:
    return [
        Segment(
            start=decision.source_start,
            end=decision.source_end,
            transition_type=decision.transition_type,
            visual_gap_score=1.0,
        )
        for decision in edit_plan.decisions
    ]


def map_output_to_original(output_time: float, edit_plan: EditDecisionList) -> float:
    if not edit_plan.decisions:
        return 0.0

    for decision in edit_plan.decisions:
        if output_time < decision.output_start:
            return decision.source_start
        if output_time <= decision.output_end:
            return decision.source_start + (output_time - decision.output_start)

    return edit_plan.decisions[-1].source_end


def build_segment_map(edit_plan: EditDecisionList) -> list[tuple[float, float, float]]:
    return [
        (decision.source_start, decision.source_end, decision.output_start)
        for decision in edit_plan.decisions
    ]


def cut_points(edit_plan: EditDecisionList) -> dict[int, float]:
    return {
        decision.index: decision.output_end
        for decision in edit_plan.decisions[:-1]
    }


def boundary_points(edit_plan: EditDecisionList) -> list[float]:
    points: list[float] = []
    for index, decision in enumerate(edit_plan.decisions[:-1]):
        points.append(decision.output_end)
        next_start = edit_plan.decisions[index + 1].output_start
        if next_start > decision.output_end + 1e-6:
            points.append(next_start)
    return points
