import numpy as np
from pathlib import Path
from unittest.mock import patch

from ummfiltered.models import (
    CutAdjustment, DetectionSource, FillerSegment, Segment,
    TransitionType, VerificationResult, Word,
)
from ummfiltered.verify import (
    build_segment_map,
    map_output_to_original,
    check_remaining_fillers,
)


class TestBuildSegmentMap:
    def test_simple_two_segments(self):
        segments = [
            Segment(0.0, 3.0, TransitionType.HARD, 1.0),
            Segment(4.0, 10.0, TransitionType.HARD, 1.0),
        ]
        seg_map = build_segment_map(segments)
        assert len(seg_map) == 2
        assert seg_map[0] == (0.0, 3.0, 0.0)
        assert seg_map[1] == (4.0, 10.0, 3.0)

    def test_three_segments(self):
        segments = [
            Segment(0.0, 2.0, TransitionType.HARD, 1.0),
            Segment(3.0, 5.0, TransitionType.HARD, 1.0),
            Segment(6.0, 10.0, TransitionType.HARD, 1.0),
        ]
        seg_map = build_segment_map(segments)
        assert seg_map[2] == (6.0, 10.0, 4.0)


class TestMapOutputToOriginal:
    def test_timestamp_in_first_segment(self):
        segments = [
            Segment(0.0, 3.0, TransitionType.HARD, 1.0),
            Segment(4.0, 10.0, TransitionType.HARD, 1.0),
        ]
        seg_map = build_segment_map(segments)
        assert map_output_to_original(1.5, seg_map) == 1.5

    def test_timestamp_in_second_segment(self):
        segments = [
            Segment(0.0, 3.0, TransitionType.HARD, 1.0),
            Segment(4.0, 10.0, TransitionType.HARD, 1.0),
        ]
        seg_map = build_segment_map(segments)
        assert map_output_to_original(4.0, seg_map) == 5.0

    def test_timestamp_at_boundary(self):
        segments = [
            Segment(0.0, 3.0, TransitionType.HARD, 1.0),
            Segment(4.0, 10.0, TransitionType.HARD, 1.0),
        ]
        seg_map = build_segment_map(segments)
        assert map_output_to_original(3.0, seg_map) == 4.0


class TestCheckRemainingFillers:
    def test_known_filler_still_present(self):
        original_fillers = [
            FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY),
        ]
        segments = [
            Segment(0.0, 5.0, TransitionType.HARD, 1.0),
            Segment(5.5, 10.0, TransitionType.HARD, 1.0),
        ]
        output_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("um", 4.8, 5.0, 0.3),
            Word("world", 5.0, 5.5, 0.92),
        ]
        remaining, new = check_remaining_fillers(
            output_words, original_fillers, segments,
            aggressive=False, min_confidence=0.15,
        )
        assert len(remaining) == 1
        assert remaining[0].word == "um"
        assert len(new) == 0

    def test_new_filler_detected(self):
        original_fillers = [
            FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY),
        ]
        segments = [
            Segment(0.0, 5.0, TransitionType.HARD, 1.0),
            Segment(5.5, 10.0, TransitionType.HARD, 1.0),
        ]
        output_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("uh", 2.0, 2.3, 0.6),
            Word("world", 3.0, 3.5, 0.92),
        ]
        remaining, new = check_remaining_fillers(
            output_words, original_fillers, segments,
            aggressive=False, min_confidence=0.15,
        )
        assert len(remaining) == 0
        assert len(new) == 1
        assert new[0].word == "uh"

    def test_no_issues(self):
        original_fillers = [
            FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY),
        ]
        segments = [
            Segment(0.0, 5.0, TransitionType.HARD, 1.0),
            Segment(5.5, 10.0, TransitionType.HARD, 1.0),
        ]
        output_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("world", 3.0, 3.5, 0.92),
        ]
        remaining, new = check_remaining_fillers(
            output_words, original_fillers, segments,
            aggressive=False, min_confidence=0.15,
        )
        assert len(remaining) == 0
        assert len(new) == 0

from ummfiltered.verify import check_word_integrity


class TestCheckWordIntegrity:
    def test_all_words_present(self):
        original_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("um", 0.6, 0.9, 0.4),
            Word("world", 1.0, 1.5, 0.92),
        ]
        cut_fillers = [
            FillerSegment(0.6, 0.9, "um", 0.4, DetectionSource.DICTIONARY),
        ]
        output_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("world", 0.6, 1.1, 0.92),
        ]
        segments = [
            Segment(0.0, 0.6, TransitionType.HARD, 1.0),
            Segment(0.9, 1.5, TransitionType.HARD, 1.0),
        ]
        lost, damaged = check_word_integrity(
            original_words, output_words, cut_fillers, segments,
        )
        assert len(lost) == 0
        assert len(damaged) == 0

    def test_detects_lost_word(self):
        original_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("um", 0.6, 0.9, 0.4),
            Word("beautiful", 1.0, 1.4, 0.92),
            Word("world", 1.5, 2.0, 0.92),
        ]
        cut_fillers = [
            FillerSegment(0.6, 0.9, "um", 0.4, DetectionSource.DICTIONARY),
        ]
        output_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("world", 0.6, 1.1, 0.92),
        ]
        segments = [
            Segment(0.0, 0.6, TransitionType.HARD, 1.0),
            Segment(0.9, 2.0, TransitionType.HARD, 1.0),
        ]
        lost, damaged = check_word_integrity(
            original_words, output_words, cut_fillers, segments,
        )
        assert len(lost) == 1
        assert lost[0].text == "beautiful"

    def test_detects_damaged_word_at_cut(self):
        original_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("um", 0.6, 0.9, 0.4),
            Word("world", 1.0, 1.5, 0.92),
        ]
        cut_fillers = [
            FillerSegment(0.6, 0.9, "um", 0.4, DetectionSource.DICTIONARY),
        ]
        output_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("orld", 0.6, 0.9, 0.4),
        ]
        segments = [
            Segment(0.0, 0.6, TransitionType.HARD, 1.0),
            Segment(0.9, 1.5, TransitionType.HARD, 1.0),
        ]
        lost, damaged = check_word_integrity(
            original_words, output_words, cut_fillers, segments,
        )
        assert len(damaged) >= 1

from ummfiltered.verify import check_audio_smoothness


class TestCheckAudioSmoothness:
    def test_smooth_audio_no_discontinuities(self):
        sample_rate = 16000
        np.random.seed(42)
        samples = np.random.normal(0, 0.01, sample_rate * 10).astype(np.float32)
        cut_points = [3.0, 6.0]
        discs = check_audio_smoothness(samples, sample_rate, cut_points)
        assert len(discs) == 0

    def test_detects_volume_jump(self):
        sample_rate = 16000
        samples = np.zeros(sample_rate * 4, dtype=np.float32)
        samples[:sample_rate * 2] = 0.01
        samples[sample_rate * 2:] = 0.5
        cut_points = [2.0]
        discs = check_audio_smoothness(samples, sample_rate, cut_points)
        assert len(discs) == 1
        assert discs[0][0] == 2.0

    def test_detects_dc_offset_jump(self):
        sample_rate = 16000
        samples = np.full(sample_rate * 4, 0.01, dtype=np.float32)
        cut_sample = sample_rate * 2
        samples[cut_sample:] += 0.3
        cut_points = [2.0]
        discs = check_audio_smoothness(samples, sample_rate, cut_points)
        assert len(discs) >= 1

from ummfiltered.verify import verify_output


class TestVerifyOutput:
    @patch("ummfiltered.verify.transcribe")
    @patch("ummfiltered.verify.extract_audio_pcm")
    def test_returns_clean_when_no_issues(self, mock_audio, mock_transcribe):
        mock_transcribe.return_value = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("world", 0.6, 1.1, 0.92),
        ]
        mock_audio.return_value = (
            np.random.normal(0, 0.01, 16000 * 2).astype(np.float32),
            16000,
        )

        original_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("um", 0.6, 0.9, 0.4),
            Word("world", 1.0, 1.5, 0.92),
        ]
        fillers = [
            FillerSegment(0.6, 0.9, "um", 0.4, DetectionSource.DICTIONARY),
        ]
        segments = [
            Segment(0.0, 0.6, TransitionType.HARD, 1.0),
            Segment(0.9, 1.5, TransitionType.HARD, 1.0),
        ]

        result = verify_output(
            output_path=Path("/fake/output.mp4"),
            original_words=original_words,
            original_fillers=fillers,
            segments=segments,
            model_size="medium",
            aggressive=False,
            min_confidence=0.15,
        )
        assert isinstance(result, VerificationResult)
        assert result.is_clean()

from ummfiltered.verify import apply_adjustments, rebuild_cuts


class TestApplyAdjustments:
    def test_widens_expansion_for_remaining_filler(self):
        f = FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY)
        adjustments = {0: CutAdjustment(filler=f, expansion_ms=300.0, crossfade_ms=40.0)}
        result = VerificationResult(
            remaining_fillers=[f], new_fillers=[], lost_words=[],
            damaged_words=[], audio_discontinuities=[],
        )
        apply_adjustments(adjustments, result)
        assert adjustments[0].expansion_ms == 450.0

    def test_increases_crossfade_for_audio_discontinuity(self):
        f = FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY)
        adjustments = {0: CutAdjustment(filler=f, expansion_ms=300.0, crossfade_ms=40.0)}
        result = VerificationResult(
            remaining_fillers=[], new_fillers=[], lost_words=[],
            damaged_words=[], audio_discontinuities=[(5.0, 8.0)],
        )
        segments = [
            Segment(0.0, 5.0, TransitionType.HARD, 1.0),
            Segment(5.5, 10.0, TransitionType.HARD, 1.0),
        ]
        apply_adjustments(adjustments, result, segments=segments)
        assert adjustments[0].crossfade_ms == 80.0

    def test_reduces_expansion_for_damaged_word(self):
        f = FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY)
        damaged_word = Word("world", 5.5, 6.0, 0.4)
        adjustments = {0: CutAdjustment(filler=f, expansion_ms=300.0, crossfade_ms=40.0)}
        result = VerificationResult(
            remaining_fillers=[], new_fillers=[], lost_words=[],
            damaged_words=[(damaged_word, 0)], audio_discontinuities=[],
        )
        apply_adjustments(adjustments, result)
        assert adjustments[0].expansion_ms == 200.0

    def test_skips_filler_when_lost_word_very_close(self):
        f = FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY)
        lost_word = Word("hello", 4.5, 4.92, 0.95)
        adjustments = {0: CutAdjustment(filler=f, expansion_ms=300.0, crossfade_ms=40.0)}
        result = VerificationResult(
            remaining_fillers=[], new_fillers=[], lost_words=[lost_word],
            damaged_words=[], audio_discontinuities=[],
        )
        apply_adjustments(adjustments, result)
        assert adjustments[0].skip is True

    def test_reduces_expansion_for_lost_word_nearby(self):
        f = FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY)
        lost_word = Word("hello", 4.0, 4.5, 0.95)
        adjustments = {0: CutAdjustment(filler=f, expansion_ms=300.0, crossfade_ms=40.0)}
        result = VerificationResult(
            remaining_fillers=[], new_fillers=[], lost_words=[lost_word],
            damaged_words=[], audio_discontinuities=[],
        )
        apply_adjustments(adjustments, result)
        assert adjustments[0].skip is False
        assert adjustments[0].expansion_ms == 200.0

    def test_adds_new_fillers(self):
        f_orig = FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY)
        f_new = FillerSegment(8.0, 8.3, "uh", 0.6, DetectionSource.DICTIONARY)
        adjustments = {0: CutAdjustment(filler=f_orig, expansion_ms=300.0, crossfade_ms=40.0)}
        result = VerificationResult(
            remaining_fillers=[], new_fillers=[f_new], lost_words=[],
            damaged_words=[], audio_discontinuities=[],
        )
        apply_adjustments(adjustments, result)
        assert len(adjustments) == 2
        assert adjustments[1].filler.word == "uh"


class TestRebuildCuts:
    def test_rebuilds_with_wider_expansion(self):
        f = FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY)
        adjustments = {0: CutAdjustment(filler=f, expansion_ms=450.0, crossfade_ms=40.0)}
        samples = np.zeros(16000 * 10, dtype=np.float32)
        fillers, crossfades = rebuild_cuts(adjustments, samples, 16000)
        assert len(fillers) == 1

    def test_skips_marked_fillers(self):
        f = FillerSegment(5.0, 5.5, "um", 0.4, DetectionSource.DICTIONARY)
        adjustments = {0: CutAdjustment(filler=f, expansion_ms=300.0, crossfade_ms=40.0, skip=True)}
        samples = np.zeros(16000 * 10, dtype=np.float32)
        fillers, crossfades = rebuild_cuts(adjustments, samples, 16000)
        assert len(fillers) == 0
