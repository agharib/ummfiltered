from ummfiltered.render import build_segment_filter
from ummfiltered.models import Segment, TransitionType


class TestBuildSegmentFilter:
    def test_single_segment(self):
        segs = [Segment(0.0, 5.0, TransitionType.HARD, 1.0)]
        vf, af = build_segment_filter(segs)
        assert "0.0" in vf
        assert "5.0" in vf

    def test_multiple_segments(self):
        segs = [
            Segment(0.0, 3.0, TransitionType.HARD, 1.0),
            Segment(4.0, 8.0, TransitionType.HARD, 0.9),
        ]
        vf, af = build_segment_filter(segs)
        assert "3.0" in vf
        assert "4.0" in vf
