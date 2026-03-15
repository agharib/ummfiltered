from ummfiltered.models import (
    DetectionSource,
    FillerSegment,
    PhraseAction,
    SeamReport,
    SeamReportEntry,
    Segment,
    TransitionType,
    Word,
)
from ummfiltered.edit_plan import build_edit_decision_list
from ummfiltered.phrase_planner import (
    apply_phrase_candidates,
    build_phrase_report,
    build_phrase_windows,
    plan_phrase_candidates,
)


class TestPhraseWindows:
    def test_groups_fillers_when_gap_is_short(self):
        fillers = [
            FillerSegment(1.0, 1.2, "um", 0.8, DetectionSource.DICTIONARY),
            FillerSegment(1.45, 1.6, "uh", 0.8, DetectionSource.DICTIONARY),
        ]
        words = [
            Word("well", 0.6, 0.9, 0.98),
            Word("um", 1.0, 1.2, 0.7),
            Word("uh", 1.45, 1.6, 0.7),
            Word("today", 1.7, 2.0, 0.99),
        ]

        windows = build_phrase_windows(fillers, words)

        assert len(windows) == 1
        assert windows[0].filler_indices == [0, 1]

    def test_splits_fillers_when_gap_is_large_with_many_words(self):
        fillers = [
            FillerSegment(1.0, 1.2, "um", 0.8, DetectionSource.DICTIONARY),
            FillerSegment(2.4, 2.6, "uh", 0.8, DetectionSource.DICTIONARY),
        ]
        words = [
            Word("well", 0.6, 0.9, 0.98),
            Word("um", 1.0, 1.2, 0.7),
            Word("this", 1.3, 1.45, 0.99),
            Word("is", 1.5, 1.6, 0.99),
            Word("a", 1.62, 1.68, 0.99),
            Word("test", 1.72, 1.95, 0.99),
            Word("uh", 2.4, 2.6, 0.7),
        ]

        windows = build_phrase_windows(fillers, words)

        assert len(windows) == 2


class TestPhraseCandidates:
    def test_prefers_shorten_for_dense_phrase(self):
        fillers = [
            FillerSegment(1.0, 1.3, "um", 0.8, DetectionSource.DICTIONARY),
            FillerSegment(1.45, 1.75, "uh", 0.8, DetectionSource.DICTIONARY),
        ]
        words = [
            Word("I", 0.85, 0.95, 0.99),
            Word("um", 1.0, 1.3, 0.8),
            Word("uh", 1.45, 1.75, 0.8),
            Word("think", 1.8, 2.05, 0.99),
        ]

        candidates = plan_phrase_candidates(fillers, words)

        assert len(candidates) == 1
        assert candidates[0].shortened_fillers >= 1
        assert candidates[0].deleted_fillers < 2

    def test_contextual_filler_falls_back_to_keep(self):
        fillers = [
            FillerSegment(1.0, 1.2, "like", 0.5, DetectionSource.CONTEXTUAL),
        ]
        words = [
            Word("it", 0.8, 0.92, 0.99),
            Word("like", 1.0, 1.2, 0.5),
            Word("works", 1.25, 1.55, 0.99),
        ]

        candidates = plan_phrase_candidates(fillers, words)

        assert len(candidates) == 1
        assert candidates[0].decisions[0].action == PhraseAction.KEEP

    def test_apply_phrase_candidates_returns_cut_and_allowed_fillers(self):
        fillers = [
            FillerSegment(1.0, 1.3, "um", 0.8, DetectionSource.DICTIONARY),
            FillerSegment(1.45, 1.75, "like", 0.5, DetectionSource.CONTEXTUAL),
        ]
        words = [
            Word("I", 0.85, 0.95, 0.99),
            Word("um", 1.0, 1.3, 0.8),
            Word("like", 1.45, 1.75, 0.5),
            Word("think", 1.8, 2.05, 0.99),
        ]

        candidates = plan_phrase_candidates(fillers, words)
        cut_fillers, allowed_fillers = apply_phrase_candidates(fillers, candidates)

        assert len(cut_fillers) >= 1
        assert any(filler.word == "like" for filler in allowed_fillers)


class TestPhraseReport:
    def test_attaches_worst_seam_score_to_phrase_window(self):
        segments = [
            Segment(0.0, 1.0, TransitionType.HARD, 1.0),
            Segment(1.3, 2.5, TransitionType.HARD, 1.0),
        ]
        edit_plan = build_edit_decision_list(segments)
        candidates = [
            plan_phrase_candidates(
                [FillerSegment(1.0, 1.3, "um", 0.8, DetectionSource.DICTIONARY)],
                [
                    Word("hello", 0.7, 0.95, 0.99),
                    Word("um", 1.0, 1.3, 0.8),
                    Word("world", 1.35, 1.7, 0.99),
                ],
            )[0]
        ]
        seam_report = SeamReport(entries=[
            SeamReportEntry(
                seam_index=0,
                output_time=1.0,
                chosen_strategy="raw",
                before_score=1.0,
                after_score=1.6,
                left_shift_ms=0.0,
                right_shift_ms=0.0,
                duration_ms=0.0,
                accepted=True,
            )
        ])

        phrase_report = build_phrase_report(candidates, edit_plan, seam_report=seam_report)

        assert phrase_report.entries[0].worst_seam_score == 1.6
        assert phrase_report.entries[0].score >= candidates[0].score
