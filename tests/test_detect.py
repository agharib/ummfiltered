from ummfiltered.detect import detect_fillers_dictionary, detect_fillers_contextual, detect_fillers, filter_fillers_by_context, expand_zero_duration_fillers
from ummfiltered.models import Word, FillerSegment, DetectionSource


class TestDictionaryDetection:
    def test_detects_um(self):
        words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("um", 0.6, 0.9, 0.4),
            Word("world", 1.0, 1.5, 0.92),
        ]
        fillers = detect_fillers_dictionary(words)
        assert len(fillers) == 1
        assert fillers[0].word == "um"
        assert fillers[0].source == DetectionSource.DICTIONARY

    def test_detects_multi_word_filler(self):
        words = [
            Word("you", 0.0, 0.2, 0.9),
            Word("know", 0.2, 0.5, 0.9),
            Word("the", 0.6, 0.8, 0.95),
        ]
        fillers = detect_fillers_dictionary(words)
        assert len(fillers) == 1
        assert fillers[0].word == "you know"
        assert fillers[0].start == 0.0
        assert fillers[0].end == 0.5

    def test_ignores_normal_words(self):
        words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("world", 0.6, 1.0, 0.92),
        ]
        fillers = detect_fillers_dictionary(words)
        assert len(fillers) == 0

    def test_custom_filler_list(self):
        words = [Word("anyway", 0.0, 0.5, 0.8)]
        fillers = detect_fillers_dictionary(words, custom_fillers=["anyway"])
        assert len(fillers) == 1


class TestContextualDetection:
    def test_flags_low_confidence_word_with_pauses(self):
        words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("like", 1.2, 1.5, 0.3),
            Word("world", 2.0, 2.5, 0.92),
        ]
        fillers = detect_fillers_contextual(words)
        assert len(fillers) == 1
        assert fillers[0].source == DetectionSource.CONTEXTUAL

    def test_ignores_high_confidence_word(self):
        words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("like", 0.6, 0.9, 0.95),
            Word("pizza", 1.0, 1.5, 0.92),
        ]
        fillers = detect_fillers_contextual(words)
        assert len(fillers) == 0


class TestDetectFillers:
    def test_combines_dictionary_and_contextual_when_aggressive(self):
        words = [
            Word("um", 0.0, 0.3, 0.4),
            Word("hello", 0.5, 1.0, 0.95),
            Word("like", 1.8, 2.0, 0.3),
            Word("world", 2.5, 3.0, 0.92),
        ]
        fillers = detect_fillers(words, aggressive=True)
        assert len(fillers) == 2

    def test_only_dictionary_when_not_aggressive(self):
        words = [
            Word("um", 0.0, 0.3, 0.4),
            Word("hello", 0.5, 1.0, 0.95),
            Word("like", 1.8, 2.0, 0.3),
            Word("world", 2.5, 3.0, 0.92),
        ]
        fillers = detect_fillers(words, aggressive=False)
        assert len(fillers) == 1


class TestFilterFillersByContext:
    def test_keeps_low_confidence_filler_with_pause(self):
        words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("uh", 1.0, 1.3, 0.00),
            Word("world", 2.0, 2.5, 0.92),
        ]
        fillers = [FillerSegment(1.0, 1.3, "uh", 0.00, DetectionSource.DICTIONARY)]
        kept = filter_fillers_by_context(fillers, words, min_confidence=0.15)
        assert len(kept) == 1

    def test_filters_low_confidence_filler_without_pause(self):
        words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("uh", 0.55, 0.6, 0.00),
            Word("world", 0.65, 1.0, 0.92),
        ]
        fillers = [FillerSegment(0.55, 0.6, "uh", 0.00, DetectionSource.DICTIONARY)]
        kept = filter_fillers_by_context(fillers, words, min_confidence=0.15)
        assert len(kept) == 0

    def test_keeps_high_confidence_filler_regardless(self):
        words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("uh", 0.55, 0.6, 0.80),
            Word("world", 0.65, 1.0, 0.92),
        ]
        fillers = [FillerSegment(0.55, 0.6, "uh", 0.80, DetectionSource.DICTIONARY)]
        kept = filter_fillers_by_context(fillers, words, min_confidence=0.15)
        assert len(kept) == 1

    def test_filters_low_confidence_non_single_filler(self):
        words = [
            Word("you", 1.0, 1.2, 0.01),
            Word("know", 1.2, 1.5, 0.01),
            Word("world", 2.0, 2.5, 0.92),
        ]
        fillers = [FillerSegment(1.0, 1.5, "you know", 0.01, DetectionSource.DICTIONARY)]
        kept = filter_fillers_by_context(fillers, words, min_confidence=0.15)
        assert len(kept) == 0


class TestExpandZeroDurationFillers:
    def test_expands_zero_duration_to_word_gap(self):
        words = [
            Word("hello", 0.0, 0.9, 0.95),
            Word("uh", 1.2, 1.2, 0.71),
            Word("world", 1.4, 1.9, 0.92),
        ]
        fillers = [FillerSegment(1.2, 1.2, "uh", 0.71, DetectionSource.DICTIONARY)]
        result = expand_zero_duration_fillers(fillers, words)
        assert len(result) == 1
        assert result[0].start == 0.9
        assert result[0].end == 1.4

    def test_leaves_normal_duration_filler_unchanged(self):
        words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("uh", 0.6, 0.9, 0.80),
            Word("world", 1.0, 1.5, 0.92),
        ]
        fillers = [FillerSegment(0.6, 0.9, "uh", 0.80, DetectionSource.DICTIONARY)]
        result = expand_zero_duration_fillers(fillers, words)
        assert len(result) == 1
        assert result[0].start == 0.6
        assert result[0].end == 0.9

    def test_caps_expansion_at_half_second(self):
        words = [
            Word("hello", 0.0, 0.5, 0.95),
            Word("uh", 5.0, 5.0, 0.71),
            Word("world", 10.0, 10.5, 0.92),
        ]
        fillers = [FillerSegment(5.0, 5.0, "uh", 0.71, DetectionSource.DICTIONARY)]
        result = expand_zero_duration_fillers(fillers, words)
        assert len(result) == 1
        assert result[0].start == 4.5
        assert result[0].end == 5.5
