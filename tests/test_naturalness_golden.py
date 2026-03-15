from __future__ import annotations

import json
from pathlib import Path

import pytest


FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "naturalness"


def _manifests() -> list[Path]:
    return sorted(FIXTURES_ROOT.glob("*/manifest.json"))


def test_golden_set_scaffold_has_three_clip_slots():
    manifests = _manifests()
    assert len(manifests) == 3


@pytest.mark.skipif(not _manifests(), reason="golden-set scaffold is unavailable")
@pytest.mark.parametrize("manifest_path", _manifests())
def test_golden_set_clip_manifests_are_well_formed(manifest_path: Path):
    payload = json.loads(manifest_path.read_text())
    required_keys = {"id", "title", "media", "transcript", "fillers", "notes"}
    assert required_keys.issubset(payload)

    media_path = manifest_path.parent / payload["media"]
    transcript_path = manifest_path.parent / payload["transcript"]
    fillers_path = manifest_path.parent / payload["fillers"]
    notes_path = manifest_path.parent / payload["notes"]

    if not media_path.exists():
        pytest.skip(f"Golden clip media is not checked in yet for {payload['id']}.")

    assert transcript_path.exists()
    assert fillers_path.exists()
    assert notes_path.exists()
