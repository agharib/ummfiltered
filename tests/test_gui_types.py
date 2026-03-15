from pathlib import Path

from ummfiltered.gui_types import default_output_path, gui_request_from_dict, resolve_gui_request


class TestGuiRequestResolution:
    def test_quality_preset_uses_large_lossless_defaults(self):
        request = gui_request_from_dict(
            {
                "inputPath": "/tmp/input.mov",
                "preset": "quality",
                "aggressive": False,
                "verifyPass": True,
                "naturalPauses": True,
                "overrides": {},
            }
        )

        resolved = resolve_gui_request(request)

        assert resolved["model_size"] == "large"
        assert resolved["quality"] == "lossless"
        assert resolved["no_refine"] is False
        assert resolved["pause_ms"] is None

    def test_visible_toggles_override_preset_defaults(self):
        request = gui_request_from_dict(
            {
                "inputPath": "/tmp/input.mov",
                "preset": "speed",
                "aggressive": True,
                "verifyPass": True,
                "naturalPauses": False,
                "overrides": {},
            }
        )

        resolved = resolve_gui_request(request)

        assert resolved["model_size"] == "base"
        assert resolved["quality"] == "matched"
        assert resolved["no_refine"] is False
        assert resolved["pause_ms"] == 0.0
        assert resolved["aggressive"] is True

    def test_advanced_overrides_take_precedence(self):
        request = gui_request_from_dict(
            {
                "inputPath": "/tmp/input.mov",
                "preset": "balanced",
                "aggressive": False,
                "verifyPass": True,
                "naturalPauses": True,
                "overrides": {
                    "modelSize": "tiny",
                    "quality": "matched",
                    "minConfidence": 0.4,
                    "customFillers": ["anyway", "right"],
                    "fixedPauseMs": 120,
                },
            }
        )

        resolved = resolve_gui_request(request)

        assert resolved["model_size"] == "tiny"
        assert resolved["quality"] == "matched"
        assert resolved["min_confidence"] == 0.4
        assert resolved["custom_fillers"] == ["anyway", "right"]
        assert resolved["pause_ms"] == 120


class TestDefaultOutputPath:
    def test_appends_ummfiltered_suffix(self):
        result = default_output_path(Path("/tmp/demo.mp4"))
        assert result == Path("/tmp/demo_ummfiltered.mp4")
