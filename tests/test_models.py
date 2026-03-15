from ummfiltered.models import (
    CutAdjustment,
    DetectionSource,
    FillerSegment,
    Segment,
    TransitionType,
    VerificationResult,
    VideoMetadata,
    Word,
)


class TestWord:
    def test_creation(self):
        w = Word(text="um", start=1.0, end=1.5, probability=0.3)
        assert w.text == "um"
        assert w.start == 1.0
        assert w.end == 1.5
        assert w.probability == 0.3


class TestFillerSegment:
    def test_dictionary_source(self):
        f = FillerSegment(start=1.0, end=1.3, word="um", confidence=0.4, source=DetectionSource.DICTIONARY)
        assert f.source == DetectionSource.DICTIONARY

    def test_contextual_source(self):
        f = FillerSegment(start=2.0, end=2.2, word="like", confidence=0.6, source=DetectionSource.CONTEXTUAL)
        assert f.source == DetectionSource.CONTEXTUAL


class TestSegment:
    def test_hard_cut(self):
        s = Segment(start=0.0, end=5.0, transition_type=TransitionType.HARD, visual_gap_score=0.92)
        assert s.transition_type == TransitionType.HARD

    def test_interpolate(self):
        s = Segment(start=5.5, end=10.0, transition_type=TransitionType.INTERPOLATE, visual_gap_score=0.6)
        assert s.transition_type == TransitionType.INTERPOLATE


class TestCutAdjustment:
    def test_creation_with_defaults(self):
        f = FillerSegment(start=1.0, end=1.3, word="um", confidence=0.4, source=DetectionSource.DICTIONARY)
        adj = CutAdjustment(filler=f, expansion_ms=300.0, crossfade_ms=40.0)
        assert adj.filler is f
        assert adj.expansion_ms == 300.0
        assert adj.crossfade_ms == 40.0
        assert adj.skip is False

    def test_skip_flag(self):
        f = FillerSegment(start=1.0, end=1.3, word="um", confidence=0.4, source=DetectionSource.DICTIONARY)
        adj = CutAdjustment(filler=f, expansion_ms=300.0, crossfade_ms=40.0, skip=True)
        assert adj.skip is True


class TestVerificationResult:
    def test_is_clean_when_empty(self):
        result = VerificationResult(
            remaining_fillers=[], new_fillers=[], lost_words=[],
            damaged_words=[], audio_discontinuities=[],
        )
        assert result.is_clean() is True

    def test_not_clean_with_remaining_fillers(self):
        f = FillerSegment(start=1.0, end=1.3, word="um", confidence=0.4, source=DetectionSource.DICTIONARY)
        result = VerificationResult(
            remaining_fillers=[f], new_fillers=[], lost_words=[],
            damaged_words=[], audio_discontinuities=[],
        )
        assert result.is_clean() is False

    def test_not_clean_with_audio_discontinuity(self):
        result = VerificationResult(
            remaining_fillers=[], new_fillers=[], lost_words=[],
            damaged_words=[], audio_discontinuities=[(34.2, 8.5)],
        )
        assert result.is_clean() is False
