import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ummfiltered.config import INTERPOLATION_FRAMES


def save_frame_png(frame: np.ndarray, directory: Path, filename: str) -> Path:
    path = directory / filename
    Image.fromarray(frame).save(path)
    return path


def load_frame_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def build_ncnn_command(input_dir: str, output_dir: str, num_frames: int) -> list[str]:
    binary = shutil.which("rife-ncnn-vulkan") or "rife-ncnn-vulkan"
    return [
        binary,
        "-i", input_dir,
        "-o", output_dir,
        "-n", str(num_frames),
    ]


def interpolate_frames_ncnn(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    num_frames: int = INTERPOLATION_FRAMES,
) -> list[np.ndarray]:
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        save_frame_png(frame_a, Path(input_dir), "000.png")
        save_frame_png(frame_b, Path(input_dir), "001.png")

        cmd = build_ncnn_command(input_dir, output_dir, num_frames)
        subprocess.run(cmd, capture_output=True, check=True)

        frames = []
        for p in sorted(Path(output_dir).glob("*.png")):
            frames.append(load_frame_png(p))

    return frames


def interpolate_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    num_frames: int = INTERPOLATION_FRAMES,
    backend: str = "ncnn",
) -> list[np.ndarray]:
    if backend == "ncnn":
        return interpolate_frames_ncnn(frame_a, frame_b, num_frames)
    raise ValueError(f"Unknown interpolation backend: {backend}")
