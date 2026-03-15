import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "test.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest", str(video_path),
        ],
        capture_output=True,
        check=True,
    )
    return video_path


class TestCLIParsing:
    def test_default_output_name(self):
        from ummfiltered.cli import parse_args
        args = parse_args(["video.mp4"])
        assert args.output == Path("video_ummfiltered.mp4")

    def test_custom_output(self):
        from ummfiltered.cli import parse_args
        args = parse_args(["video.mp4", "-o", "clean.mp4"])
        assert args.output == Path("clean.mp4")

    def test_all_flags(self):
        from ummfiltered.cli import parse_args
        args = parse_args([
            "video.mp4", "--interactive", "--aggressive",
            "--cloud", "deepgram", "--model-size", "large",
            "--quality", "lossless", "--dry-run",
            "--fillers", "um,uh,like",
        ])
        assert args.interactive is True
        assert args.aggressive is True
        assert args.cloud == "deepgram"
        assert args.model_size == "large"
        assert args.quality == "lossless"
        assert args.dry_run is True
        assert args.fillers == "um,uh,like"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
class TestVideoProbing:
    def test_probe_sample_video(self, sample_video: Path):
        from ummfiltered.render import probe_video
        meta = probe_video(sample_video)
        assert meta.width == 320
        assert meta.height == 240
        assert meta.framerate == 30.0
        assert meta.duration > 4.0


class TestCLIParsingRefine:
    def test_no_refine_flag(self):
        from ummfiltered.cli import parse_args
        args = parse_args(["video.mp4", "--no-refine"])
        assert args.no_refine is True

    def test_refine_default_on(self):
        from ummfiltered.cli import parse_args
        args = parse_args(["video.mp4"])
        assert args.no_refine is False
