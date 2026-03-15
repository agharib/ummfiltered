from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ummfiltered.models import GuiOverrides, GuiPreset, GuiProcessRequest


PRESET_DEFAULTS: dict[GuiPreset, dict[str, Any]] = {
    GuiPreset.SPEED: {
        "model_size": "base",
        "quality": "matched",
        "verify_pass": False,
    },
    GuiPreset.BALANCED: {
        "model_size": "medium",
        "quality": "matched",
        "verify_pass": True,
    },
    GuiPreset.QUALITY: {
        "model_size": "large",
        "quality": "lossless",
        "verify_pass": True,
    },
}


def default_output_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}_ummfiltered{path.suffix}")


def gui_request_from_dict(payload: dict[str, Any]) -> GuiProcessRequest:
    overrides_payload = payload.get("overrides") or {}
    overrides = GuiOverrides(
        modelSize=overrides_payload.get("modelSize"),
        quality=overrides_payload.get("quality"),
        minConfidence=overrides_payload.get("minConfidence"),
        customFillers=_normalize_custom_fillers(overrides_payload.get("customFillers")),
        fixedPauseMs=overrides_payload.get("fixedPauseMs"),
    )
    return GuiProcessRequest(
        inputPath=str(payload["inputPath"]),
        outputPath=str(payload["outputPath"]) if payload.get("outputPath") else None,
        preset=GuiPreset(payload.get("preset", GuiPreset.BALANCED.value)),
        aggressive=bool(payload.get("aggressive", False)),
        verifyPass=bool(payload.get("verifyPass", True)),
        naturalPauses=bool(payload.get("naturalPauses", True)),
        overrides=overrides,
    )


def resolve_gui_request(request: GuiProcessRequest) -> dict[str, Any]:
    preset_defaults = PRESET_DEFAULTS[request.preset]
    model_size = request.overrides.modelSize or preset_defaults["model_size"]
    quality = request.overrides.quality or preset_defaults["quality"]
    verify_pass = request.verifyPass
    if request.verifyPass is False and preset_defaults["verify_pass"] is True:
        verify_pass = False
    elif request.verifyPass is True and preset_defaults["verify_pass"] is False:
        verify_pass = True

    pause_ms: float | None
    if request.overrides.fixedPauseMs is not None:
        pause_ms = float(request.overrides.fixedPauseMs)
    elif request.naturalPauses:
        pause_ms = None
    else:
        pause_ms = 0.0

    output_path = Path(request.outputPath) if request.outputPath else default_output_path(request.inputPath)
    return {
        "input_path": Path(request.inputPath),
        "output_path": output_path,
        "model_size": model_size,
        "aggressive": request.aggressive,
        "quality": quality,
        "custom_fillers": request.overrides.customFillers,
        "min_confidence": request.overrides.minConfidence if request.overrides.minConfidence is not None else 0.15,
        "pause_ms": pause_ms,
        "no_refine": not verify_pass,
    }


def gui_request_to_dict(request: GuiProcessRequest) -> dict[str, Any]:
    payload = asdict(request)
    payload["preset"] = request.preset.value
    return payload


def _normalize_custom_fillers(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        normalized = [part for part in parts if part]
        return normalized or None
    if isinstance(value, list):
        normalized = [str(part).strip() for part in value if str(part).strip()]
        return normalized or None
    return None
