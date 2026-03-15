import numpy as np
from unittest.mock import patch

from ummfiltered.cut_planner import build_keep_segments, classify_transitions, compute_frame_similarity
from ummfiltered.models import DetectionSource, FillerSegment, Segment, TransitionType, Word


class TestBuildKeepSegments:
    def test_single_filler_produces_two_segments(self):
        fillers = [FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY)]
        segments = build_keep_segments(fillers, video_duration=10.0)
        assert len(segments) == 2
        assert segments[0].start == 0.0
        assert segments[0].end == 5.0
        assert segments[1].start == 5.5
        assert segments[1].end == 10.0

    def test_no_fillers_returns_full_video(self):
        segments = build_keep_segments([], video_duration=10.0)
        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 10.0

    def test_adjacent_fillers_merge_gap_no_words(self):
        fillers = [
            FillerSegment(3.0, 3.3, "um", 0.4, DetectionSource.DICTIONARY),
            FillerSegment(3.4, 3.7, "uh", 0.3, DetectionSource.DICTIONARY),
        ]
        segments = build_keep_segments(fillers, video_duration=10.0)
        assert len(segments) == 2
        assert segments[0].end == 3.0
        assert segments[1].start == 3.7

    def test_short_gap_kept_when_contains_words(self):
        fillers = [
            FillerSegment(3.0, 3.3, "um", 0.4, DetectionSource.DICTIONARY),
            FillerSegment(3.5, 3.8, "uh", 0.3, DetectionSource.DICTIONARY),
        ]
        words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("so", 3.3, 3.4, 0.98),
            Word("world", 4.0, 4.5, 0.92),
        ]
        segments = build_keep_segments(fillers, video_duration=10.0, words=words)
        assert len(segments) == 3
        assert segments[1].start == 3.3
        assert segments[1].end == 3.5

    def test_filler_at_start(self):
        fillers = [FillerSegment(0.0, 0.5, "um", 0.4, DetectionSource.DICTIONARY)]
        segments = build_keep_segments(fillers, video_duration=10.0)
        assert len(segments) == 1
        assert segments[0].start == 0.5


class TestFrameSimilarity:
    def test_identical_frames_score_1(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = compute_frame_similarity(frame, frame)
        assert score > 0.99

    def test_different_frames_score_low(self):
        frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 255, dtype=np.uint8)
        score = compute_frame_similarity(frame_a, frame_b)
        assert score < 0.1


class TestClassifyTransitions:
    def test_only_moves_boundaries_deeper_into_kept_segments(self):
        segments = [
            Segment(0.0, 5.0, TransitionType.HARD, 1.0),
            Segment(6.0, 10.0, TransitionType.HARD, 1.0),
        ]

        def get_frame_at(time_s: float) -> np.ndarray:
            return np.array([[[time_s]]], dtype=np.float32)

        with patch(
            "ummfiltered.cut_planner.compute_frame_similarity",
            side_effect=lambda a, b: float(b[0, 0, 0] - a[0, 0, 0]),
        ):
            classified = classify_transitions(
                segments,
                get_frame_at,
                ssim_threshold=10.0,
                search_window=2,
                framerate=10.0,
            )

        assert classified[0].end < 5.0
        assert classified[1].start > 6.0
