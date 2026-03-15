from ummfiltered.config import compute_adaptive_pause


class TestComputeAdaptivePause:
    def test_short_filler_gets_no_pause(self):
        assert compute_adaptive_pause(0.1) == 0.0

    def test_medium_filler_gets_small_pause(self):
        result = compute_adaptive_pause(0.5)
        assert abs(result - 0.1) < 0.001

    def test_long_filler_gets_larger_pause(self):
        result = compute_adaptive_pause(1.0)
        assert abs(result - 0.25) < 0.001

    def test_boundary_filler_at_threshold(self):
        result = compute_adaptive_pause(0.166)
        assert result == 0.0

    def test_negative_duration_returns_zero(self):
        assert compute_adaptive_pause(0.0) == 0.0
