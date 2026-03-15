import numpy as np
import pytest
from ummfiltered.audio import compute_rms_db, find_silence_boundaries, extract_room_tone, find_speech_onset, protect_adjacent_words
from ummfiltered.models import Word


class TestComputeRmsDb:
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
        assert start >= 1.95 + 0.04

    def test_returns_none_when_protection_collapses_region(self):
        sample_rate = 16000
        samples = np.zeros(sample_rate * 10, dtype=np.float32)
        non_filler_words = [
            Word("hello", 0.0, 2.49, 0.95),
            Word("world", 2.52, 3.0, 0.95),
        ]
        result = protect_adjacent_words(
            new_start=2.50, new_end=2.51,
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
