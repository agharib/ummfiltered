import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from ummfiltered.interpolate import save_frame_png, build_ncnn_command


class TestBuildNcnnCommand:
    def test_command_structure(self):
        cmd = build_ncnn_command(
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            num_frames=3,
        )
        assert "rife-ncnn-vulkan" in cmd[0]
        assert "/tmp/in" in cmd
        assert "/tmp/out" in cmd


class TestSaveFramePng:
    def test_saves_and_loads(self):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_frame_png(frame, Path(tmpdir), "test.png")
            assert path.exists()
            assert path.suffix == ".png"
