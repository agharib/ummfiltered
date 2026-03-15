from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]
DESKTOP_DIR = ROOT / "desktop"
BUILD_ROOT = DESKTOP_DIR / "resources" / "macos" / "worker"
FINAL_DIR = BUILD_ROOT / "ummfiltered-gui-worker"
DIST_ROOT = BUILD_ROOT / "dist"
WORK_ROOT = BUILD_ROOT / "pyinstaller-work"
SPEC_ROOT = BUILD_ROOT / "spec"
MANIFEST_PATH = BUILD_ROOT / "manifest.json"
WORKER_NAME = "ummfiltered-gui-worker"
EXCLUDED_MODULES = [
    "IPython",
    "jax",
    "jupyter",
    "matplotlib",
    "notebook",
    "pytest",
    "py",
    "tensorboard",
    "tensorflow",
    "tkinter",
    "torch",
    "torchaudio",
    "torchvision",
    "_tkinter",
]


def main() -> int:
    if sys.platform != "darwin":
        raise SystemExit("The macOS worker bundle can only be built on macOS.")

    try:
        __import__("PyInstaller")
    except ImportError as exc:  # pragma: no cover - packaging environment guard
        raise SystemExit(
            "PyInstaller is required. Install packaging dependencies with "
            '`python -m pip install -e ".[packaging]"`.'
        ) from exc

    version = project_version()
    clean_previous_outputs()
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    SPEC_ROOT.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        WORKER_NAME,
        "--distpath",
        str(DIST_ROOT),
        "--workpath",
        str(WORK_ROOT),
        "--specpath",
        str(SPEC_ROOT),
        "--paths",
        str(ROOT),
        "--collect-all",
        "faster_whisper",
        "--collect-all",
        "ctranslate2",
        "--collect-all",
        "tokenizers",
        "--collect-all",
        "imageio_ffmpeg",
        "--collect-submodules",
        "skimage.metrics",
        "--copy-metadata",
        "faster-whisper",
        "--copy-metadata",
        "ctranslate2",
        "--copy-metadata",
        "tokenizers",
        "--copy-metadata",
        "imageio-ffmpeg",
        "--copy-metadata",
        "scikit-image",
        "--copy-metadata",
        "Pillow",
    ]
    for module_name in EXCLUDED_MODULES:
        command.extend(["--exclude-module", module_name])
    command.append(str(ROOT / "ummfiltered" / "gui_worker.py"))

    subprocess.run(command, check=True)

    built_dir = DIST_ROOT / WORKER_NAME
    if not built_dir.exists():
        raise SystemExit(f"PyInstaller completed but {built_dir} was not created.")

    shutil.copytree(built_dir, FINAL_DIR)
    binary_path = FINAL_DIR / WORKER_NAME
    binary_path.chmod(0o755)

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "version": version,
                "bundleDir": str(FINAL_DIR.relative_to(DESKTOP_DIR)),
                "binary": str((FINAL_DIR / WORKER_NAME).relative_to(DESKTOP_DIR)),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Bundled macOS worker at {FINAL_DIR}")
    return 0


def clean_previous_outputs() -> None:
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)
    if DIST_ROOT.exists():
        shutil.rmtree(DIST_ROOT)
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    if SPEC_ROOT.exists():
        shutil.rmtree(SPEC_ROOT)
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"]["version"]


if __name__ == "__main__":
    raise SystemExit(main())
