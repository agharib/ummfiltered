from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def ensure_ffmpeg_tools(console=None) -> tuple[str, str]:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path and ffprobe_path:
        return ffmpeg_path, ffprobe_path

    if console is not None:
        console.print("  [yellow]ffmpeg/ffprobe not found — provisioning bundled tools...[/yellow]")

    try:
        import imageio_ffmpeg
    except ImportError:
        _install_imageio_ffmpeg()
        import imageio_ffmpeg

    ffmpeg_path, ffprobe_path = provision_bundled_tools(
        ffmpeg_exe=Path(imageio_ffmpeg.get_ffmpeg_exe()),
    )
    _prepend_to_path(str(Path(ffmpeg_path).parent))

    if console is not None:
        console.print("  [green]ffmpeg/ffprobe ready.[/green]")

    return ffmpeg_path, ffprobe_path


def ffmpeg_path() -> str:
    return ensure_ffmpeg_tools()[0]


def ffprobe_path() -> str:
    return ensure_ffmpeg_tools()[1]


def ffmpeg_cmd(*args: object) -> list[str]:
    return [ffmpeg_path(), *(str(arg) for arg in args)]


def ffprobe_cmd(*args: object) -> list[str]:
    return [ffprobe_path(), *(str(arg) for arg in args)]


def provision_bundled_tools(
    ffmpeg_exe: Path,
    shim_dir: Path | None = None,
    python_executable: str | None = None,
) -> tuple[str, str]:
    shim_dir = shim_dir or (Path.home() / ".ummfiltered" / "bin")
    python_executable = python_executable or sys.executable
    shim_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_link = shim_dir / "ffmpeg"
    _replace_symlink(ffmpeg_link, ffmpeg_exe)

    ffprobe_shim = shim_dir / "ffprobe"
    ffprobe_shim.write_text(
        "#!/bin/sh\n"
        f"exec \"{python_executable}\" -m ummfiltered.ffprobe_stub \"$@\"\n"
    )
    ffprobe_shim.chmod(ffprobe_shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return str(ffmpeg_link), str(ffprobe_shim)


def _install_imageio_ffmpeg() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "imageio-ffmpeg>=0.6.0"],
        capture_output=True,
        text=True,
        check=True,
    )


def _prepend_to_path(directory: str) -> None:
    current_path = os.environ.get("PATH", "")
    parts = [part for part in current_path.split(os.pathsep) if part]
    if directory in parts:
        return
    os.environ["PATH"] = os.pathsep.join([directory, *parts])


def _replace_symlink(link_path: Path, target: Path) -> None:
    try:
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_dir() and not link_path.is_symlink():
                raise IsADirectoryError(f"{link_path} is a directory")
            link_path.unlink()
        link_path.symlink_to(target)
    except FileExistsError:
        if link_path.is_symlink() and Path(os.readlink(link_path)) == target:
            return
        link_path.unlink(missing_ok=True)
        link_path.symlink_to(target)
