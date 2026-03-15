from __future__ import annotations

from ummfiltered.models import EditDecision, EditDecisionList, RemovedGap, Segment


def build_edit_decision_list(
    segments: list[Segment],
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> EditDecisionList:
    decisions: list[EditDecision] = []
    output_cursor = 0.0
    for index, segment in enumerate(segments):
        duration = max(0.0, segment.end - segment.start)
        output_end = output_cursor + duration
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
            pause_after=pause_overrides.get(index, 0.0) if pause_overrides else 0.0,
            transition_duration_after=(
                transition_durations.get(index + 1, 0.0) if transition_durations else 0.0
            ),
            removed_gap_after=removed_gap_after,
        )
        decisions.append(decision)
        output_cursor = output_end + decision.pause_after + decision.transition_duration_after
    return EditDecisionList(decisions=decisions, total_output_duration=output_cursor)


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
