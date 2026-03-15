import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from ummfiltered.ffmpeg_tools import ensure_ffmpeg_tools, provision_bundled_tools


class TestProvisionBundledTools:
    def test_creates_ffmpeg_and_ffprobe_shims(self, tmp_path: Path):
        ffmpeg_exe = tmp_path / "ffmpeg-real"
        ffmpeg_exe.write_text("#!/bin/sh\nexit 0\n")
        ffmpeg_exe.chmod(0o755)

        ffmpeg_path, ffprobe_path = provision_bundled_tools(
            ffmpeg_exe=ffmpeg_exe,
            shim_dir=tmp_path / "bin",
            python_executable="/usr/bin/python3",
        )

        assert Path(ffmpeg_path).is_symlink()
        assert Path(ffmpeg_path).resolve() == ffmpeg_exe
        assert Path(ffprobe_path).exists()
        assert os.access(ffprobe_path, os.X_OK)
        assert "ummfiltered.ffprobe_stub" in Path(ffprobe_path).read_text()

    def test_reuses_existing_matching_symlink(self, tmp_path: Path):
        ffmpeg_exe = tmp_path / "ffmpeg-real"
        ffmpeg_exe.write_text("#!/bin/sh\nexit 0\n")
        ffmpeg_exe.chmod(0o755)
        shim_dir = tmp_path / "bin"
        shim_dir.mkdir()
        (shim_dir / "ffmpeg").symlink_to(ffmpeg_exe)

        ffmpeg_path, ffprobe_path = provision_bundled_tools(
            ffmpeg_exe=ffmpeg_exe,
            shim_dir=shim_dir,
            python_executable="/usr/bin/python3",
        )

        assert Path(ffmpeg_path).resolve() == ffmpeg_exe
        assert Path(ffprobe_path).exists()


class TestEnsureFfmpegTools:
    @patch("ummfiltered.ffmpeg_tools.shutil.which")
    def test_reuses_system_tools(self, mock_which):
        mock_which.side_effect = ["/usr/bin/ffmpeg", "/usr/bin/ffprobe"]
        ffmpeg_path, ffprobe_path = ensure_ffmpeg_tools()
        assert ffmpeg_path == "/usr/bin/ffmpeg"
        assert ffprobe_path == "/usr/bin/ffprobe"

    @patch("ummfiltered.ffmpeg_tools._prepend_to_path")
    @patch("ummfiltered.ffmpeg_tools.provision_bundled_tools")
    @patch("ummfiltered.ffmpeg_tools.shutil.which")
    def test_provisions_bundled_tools_when_missing(
        self,
        mock_which,
        mock_provision,
        mock_prepend,
    ):
        mock_which.side_effect = [None, None]
        mock_provision.return_value = ("/tmp/bin/ffmpeg", "/tmp/bin/ffprobe")

        fake_console = MagicMock()
        ffmpeg_path, ffprobe_path = ensure_ffmpeg_tools(console=fake_console)

        assert ffmpeg_path == "/tmp/bin/ffmpeg"
        assert ffprobe_path == "/tmp/bin/ffprobe"
        mock_prepend.assert_called_once_with("/tmp/bin")
        assert fake_console.print.call_count >= 2
