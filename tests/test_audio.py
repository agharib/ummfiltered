import numpy as np
import pytest
from ummfiltered.audio import (
    assemble_audio_track,
    compute_rms_db,
    extract_room_tone,
    find_silence_boundaries,
    find_speech_onset,
    measure_cut_naturalness,
    protect_adjacent_words,
    smooth_rendered_audio,
)
from ummfiltered.edit_plan import build_edit_decision_list
from ummfiltered.models import Segment, TransitionType, Word


class TestComputeRmsDb:
    def test_empty_samples_returns_very_low_db(self):
        silence = np.array([], dtype=np.float32)
        assert compute_rms_db(silence) < -90

    def test_silence_returns_very_low_db(self):
        silence = np.zeros(1000, dtype=np.float32)
        assert compute_rms_db(silence) < -90

    def test_full_scale_sine(self):
        t = np.linspace(0, 1, 44100, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        db = compute_rms_db(sine)
        assert -4 < db < 0

    def test_half_amplitude(self):
        t = np.linspace(0, 1, 44100, dtype=np.float32)
        sine = 0.5 * np.sin(2 * np.pi * 440 * t)
        db = compute_rms_db(sine)
        assert -10 < db < -2


class TestFindSilenceBoundaries:
    def test_expands_into_silence(self):
        sample_rate = 1000
        samples = np.zeros(3000, dtype=np.float32)
        samples[1000:2000] = 0.5

        start, end = find_silence_boundaries(
            samples, sample_rate,
            filler_start=1.0, filler_end=2.0,
            threshold_db=-40, max_expansion_ms=300
        )
        assert start < 1.0
        assert end > 2.0

    def test_clamps_to_max_expansion(self):
        sample_rate = 1000
        samples = np.zeros(5000, dtype=np.float32)

        start, end = find_silence_boundaries(
            samples, sample_rate,
            filler_start=2.0, filler_end=3.0,
            threshold_db=-40, max_expansion_ms=300
        )
        assert start >= 2.0 - 0.3 - 0.01
        assert end <= 3.0 + 0.3 + 0.01


class TestFindSpeechOnset:
    def test_finds_onset_after_silence(self):
        sample_rate = 16000
        samples = np.zeros(sample_rate, dtype=np.float32)
        onset_at = int(0.5 * sample_rate)
        samples[onset_at:] = 0.3
        result = find_speech_onset(samples, sample_rate, 0.0, 0.8)
        assert abs(result - 0.5) < 0.02

    def test_returns_region_end_for_all_silence(self):
        sample_rate = 16000
        samples = np.zeros(sample_rate, dtype=np.float32)
        result = find_speech_onset(samples, sample_rate, 0.0, 0.5)
        assert result == 0.5


class TestProtectAdjacentWords:
    def test_moves_start_past_overlapping_preserved_word(self):
        sample_rate = 16000
        samples = np.zeros(sample_rate * 10, dtype=np.float32)
        non_filler_words = [Word("we", 6.58, 6.80, 0.95)]
        result = protect_adjacent_words(
            new_start=6.77, new_end=7.03,
            non_filler_words=non_filler_words,
            samples=samples, sample_rate=sample_rate,
        )
        assert result is not None
        start, _end = result
        assert start >= 6.80 + 0.009

    def test_moves_end_before_overlapping_preserved_word(self):
        sample_rate = 16000
        samples = np.zeros(sample_rate * 10, dtype=np.float32)
        non_filler_words = [Word("expect", 7.04, 7.26, 0.95)]
        result = protect_adjacent_words(
            new_start=6.80, new_end=7.12,
            non_filler_words=non_filler_words,
            samples=samples, sample_rate=sample_rate,
        )
        assert result is not None
        _start, end = result
        assert end <= 7.04 - 0.009

    def test_pulls_start_away_from_preceding_word(self):
        sample_rate = 16000
        samples = np.zeros(sample_rate * 10, dtype=np.float32)
        non_filler_words = [Word("hello", 0.0, 1.95, 0.95)]
        result = protect_adjacent_words(
            new_start=1.97, new_end=2.50,
            non_filler_words=non_filler_words,
            samples=samples, sample_rate=sample_rate,
        )
        assert result is not None
        start, end = result
        assert start >= 1.95 + 0.009

    def test_returns_none_when_protection_collapses_region(self):
        sample_rate = 16000
        samples = np.zeros(sample_rate * 10, dtype=np.float32)
        non_filler_words = [
            Word("hello", 0.0, 2.499, 0.95),
            Word("world", 2.505, 3.0, 0.95),
        ]
        result = protect_adjacent_words(
            new_start=2.500, new_end=2.504,
            non_filler_words=non_filler_words,
            samples=samples, sample_rate=sample_rate,
        )
        assert result is None

    def test_no_adjustment_when_words_far_away(self):
        sample_rate = 16000
        samples = np.zeros(sample_rate * 10, dtype=np.float32)
        non_filler_words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("world", 5.0, 5.5, 0.95),
        ]
        result = protect_adjacent_words(
            new_start=2.0, new_end=3.0,
            non_filler_words=non_filler_words,
            samples=samples, sample_rate=sample_rate,
        )
        assert result is not None
        start, end = result
        assert start == 2.0
        assert end == 3.0


class TestSmoothRenderedAudio:
    def test_clamps_cut_points_past_audio_end(self):
        samples = np.zeros(1600, dtype=np.float32)
        room_tone = np.zeros(200, dtype=np.float32)
        result = smooth_rendered_audio(
            rendered_samples=samples,
            sample_rate=16000,
            cut_points=[1.0],
            room_tone=room_tone,
        )
        assert result.shape == samples.shape

    def test_improves_naturalness_for_hard_splice(self):
        sample_rate = 16000
        t = np.linspace(0, 0.04, int(sample_rate * 0.04), endpoint=False, dtype=np.float32)
        left = 0.4 * np.sin(2 * np.pi * 220 * t)
        right = 0.4 * np.sin(2 * np.pi * 220 * t + np.pi / 2)
        samples = np.concatenate([left, right]).astype(np.float32)
        cut_point = len(left) / sample_rate
        baseline = measure_cut_naturalness(samples, sample_rate, cut_point)

        smoothed = smooth_rendered_audio(
            rendered_samples=samples,
            sample_rate=sample_rate,
            cut_points=[cut_point],
            room_tone=np.zeros(int(sample_rate * 0.02), dtype=np.float32),
        )
        improved = measure_cut_naturalness(smoothed, sample_rate, cut_point)

        assert improved.score < baseline.score
        assert improved.amplitude_jump < baseline.amplitude_jump

    def test_avoids_creating_energy_hole_in_smooth_signal(self):
        sample_rate = 16000
        t = np.linspace(0, 0.08, int(sample_rate * 0.08), endpoint=False, dtype=np.float32)
        samples = 0.4 * np.sin(2 * np.pi * 220 * t)
        cut_point = 0.04
        baseline = measure_cut_naturalness(samples, sample_rate, cut_point)

        smoothed = smooth_rendered_audio(
            rendered_samples=samples,
            sample_rate=sample_rate,
            cut_points=[cut_point],
            room_tone=np.zeros(int(sample_rate * 0.02), dtype=np.float32),
        )
        improved = measure_cut_naturalness(smoothed, sample_rate, cut_point)

        assert improved.center_drop_db <= baseline.center_drop_db + 0.05
        assert improved.score <= baseline.score + 0.05


class TestMeasureCutNaturalness:
    def test_handles_uneven_windows_near_clip_end(self):
        sample_rate = 16000
        t = np.linspace(0, 0.03, int(sample_rate * 0.03), endpoint=False, dtype=np.float32)
        samples = 0.3 * np.sin(2 * np.pi * 220 * t)
        metrics = measure_cut_naturalness(samples, sample_rate, cut_point=0.026)
        assert np.isfinite(metrics.score)


class TestAssembleAudioTrack:
    def test_keeps_raw_boundary_when_no_safe_padding_exists(self):
        sample_rate = 1000
        source_audio = np.zeros((300, 2), dtype=np.float32)
        source_audio[:100, 0] = 0.5
        source_audio[:100, 1] = 0.4
        source_audio[200:, 0] = -0.5
        source_audio[200:, 1] = -0.4

        segments = [
            Segment(0.0, 0.1, TransitionType.HARD, 1.0),
            Segment(0.2, 0.3, TransitionType.HARD, 1.0),
        ]
        preserved_words = [
            Word("left", 0.0, 0.1, 0.99),
            Word("right", 0.2, 0.3, 0.99),
        ]
        edit_plan = build_edit_decision_list(segments, preserved_words=preserved_words)
        assembled, report = assemble_audio_track(
            source_audio=source_audio,
            sample_rate=sample_rate,
            edit_plan=edit_plan,
            room_tone=np.zeros((20, 2), dtype=np.float32),
        )

        expected = np.concatenate([source_audio[:100], source_audio[200:]], axis=0)
        np.testing.assert_allclose(assembled, expected)
        assert report.entries
        assert report.entries[0].chosen_strategy == "raw"
