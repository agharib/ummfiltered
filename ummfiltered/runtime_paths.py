from __future__ import annotations

import os
from pathlib import Path


APP_SUPPORT_ENV = "UMMFILTERED_APP_SUPPORT"


def app_support_dir() -> Path:
    override = os.environ.get(APP_SUPPORT_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ummfiltered"


def bin_dir() -> Path:
    return app_support_dir() / "bin"


def interpolators_dir() -> Path:
    return app_support_dir() / "interpolators"
