from __future__ import annotations

import os
import platform
import shutil
import ssl
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


RIFE_RELEASE_TAG = "20221029"
RIFE_REPO_BASE = "https://github.com/nihui/rife-ncnn-vulkan/releases/download"


def ensure_interpolator_backend(backend: str = "ncnn", console=None) -> str:
    if backend != "ncnn":
        raise ValueError(f"Unknown interpolation backend: {backend}")

    binary_path = shutil.which("rife-ncnn-vulkan")
    if binary_path:
        return binary_path

    if console is not None:
        console.print("  [yellow]rife-ncnn-vulkan not found — provisioning bundled interpolator...[/yellow]")

    binary_path = provision_rife_bundle()
    _prepend_to_path(str(Path(binary_path).parent))

    if console is not None:
        console.print("  [green]rife-ncnn-vulkan ready.[/green]")

    return binary_path


def provision_rife_bundle(
    install_root: Path | None = None,
    shim_dir: Path | None = None,
) -> str:
    install_root = install_root or (Path.home() / ".ummfiltered" / "interpolators")
    shim_dir = shim_dir or (Path.home() / ".ummfiltered" / "bin")
    install_root.mkdir(parents=True, exist_ok=True)
    shim_dir.mkdir(parents=True, exist_ok=True)

    asset_name = _asset_name_for_platform()
    bundle_dir = install_root / asset_name.removesuffix(".zip")
    binary_path = bundle_dir / "rife-ncnn-vulkan"

    if not binary_path.exists():
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            zip_path = tmpdir / asset_name
            _download_release_asset(asset_name, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(install_root)
        if not binary_path.exists():
            raise FileNotFoundError(f"Provisioned interpolator bundle is missing {binary_path}")

    binary_path.chmod(binary_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    shim_path = shim_dir / "rife-ncnn-vulkan"
    shim_path.write_text(
        "#!/bin/sh\n"
        f"cd \"{bundle_dir}\"\n"
        "exec \"./rife-ncnn-vulkan\" \"$@\"\n"
    )
    shim_path.chmod(shim_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(shim_path)


def _asset_name_for_platform(system: str | None = None) -> str:
    system = (system or platform.system()).lower()
    if system == "darwin":
        return f"rife-ncnn-vulkan-{RIFE_RELEASE_TAG}-macos.zip"
    if system == "linux":
        return f"rife-ncnn-vulkan-{RIFE_RELEASE_TAG}-ubuntu.zip"
    if system == "windows":
        return f"rife-ncnn-vulkan-{RIFE_RELEASE_TAG}-windows.zip"
    raise FileNotFoundError(f"Unsupported platform for rife-ncnn-vulkan: {system}")


def _download_release_asset(asset_name: str, destination: Path) -> None:
    url = f"{RIFE_REPO_BASE}/{RIFE_RELEASE_TAG}/{asset_name}"
    try:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response:
            destination.write_bytes(response.read())
        return
    except Exception:
        pass

    subprocess.run(
        ["curl", "-fsSL", url, "-o", str(destination)],
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
