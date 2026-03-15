from pathlib import Path

from ummfiltered.runtime_paths import app_support_dir, bin_dir, interpolators_dir


class TestRuntimePaths:
    def test_defaults_to_hidden_home_directory(self, monkeypatch):
        monkeypatch.delenv("UMMFILTERED_APP_SUPPORT", raising=False)
        expected = Path.home() / ".ummfiltered"

        assert app_support_dir() == expected
        assert bin_dir() == expected / "bin"
        assert interpolators_dir() == expected / "interpolators"

    def test_honors_app_support_override(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("UMMFILTERED_APP_SUPPORT", str(tmp_path / "Library" / "Application Support" / "UmmFiltered"))

        assert app_support_dir() == tmp_path / "Library" / "Application Support" / "UmmFiltered"
        assert bin_dir() == tmp_path / "Library" / "Application Support" / "UmmFiltered" / "bin"
        assert (
            interpolators_dir()
            == tmp_path / "Library" / "Application Support" / "UmmFiltered" / "interpolators"
        )
