import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from ummfiltered.interpolator_tools import (
    _asset_name_for_platform,
    ensure_interpolator_backend,
    provision_rife_bundle,
)


class TestAssetNameForPlatform:
    def test_darwin_asset(self):
        assert _asset_name_for_platform("Darwin") == "rife-ncnn-vulkan-20221029-macos.zip"

    def test_linux_asset(self):
        assert _asset_name_for_platform("Linux") == "rife-ncnn-vulkan-20221029-ubuntu.zip"


class TestProvisionRifeBundle:
    def test_creates_wrapper_script(self, tmp_path: Path):
        bundle_dir = tmp_path / "interpolators" / "rife-ncnn-vulkan-20221029-macos"
        bundle_dir.mkdir(parents=True)
        binary_path = bundle_dir / "rife-ncnn-vulkan"
        binary_path.write_text("#!/bin/sh\nexit 0\n")
        binary_path.chmod(0o755)

        shim_path = provision_rife_bundle(
            install_root=tmp_path / "interpolators",
            shim_dir=tmp_path / "bin",
        )

        shim = Path(shim_path)
        assert shim.exists()
        assert os.access(shim, os.X_OK)
        contents = shim.read_text()
        assert str(bundle_dir) in contents
        assert "./rife-ncnn-vulkan" in contents


class TestEnsureInterpolatorBackend:
    @patch("ummfiltered.interpolator_tools.shutil.which")
    def test_reuses_existing_binary(self, mock_which):
        mock_which.return_value = "/usr/local/bin/rife-ncnn-vulkan"
        assert ensure_interpolator_backend("ncnn") == "/usr/local/bin/rife-ncnn-vulkan"

    @patch("ummfiltered.interpolator_tools._prepend_to_path")
    @patch("ummfiltered.interpolator_tools.provision_rife_bundle")
    @patch("ummfiltered.interpolator_tools.shutil.which")
    def test_provisions_missing_backend(
        self,
        mock_which,
        mock_provision,
        mock_prepend,
    ):
        mock_which.return_value = None
        mock_provision.return_value = "/tmp/bin/rife-ncnn-vulkan"
        fake_console = MagicMock()

        backend_path = ensure_interpolator_backend("ncnn", console=fake_console)

        assert backend_path == "/tmp/bin/rife-ncnn-vulkan"
        mock_prepend.assert_called_once_with("/tmp/bin")
        assert fake_console.print.call_count >= 2
