from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
from typing import Any

from ummfiltered.ffmpeg_tools import ensure_ffmpeg_tools
from ummfiltered.gui_types import gui_request_from_dict, resolve_gui_request
from ummfiltered.models import (
    PipelineEvent,
    PipelineEventKind,
    PipelineFinalStatus,
    PipelineResult,
)
from ummfiltered.pipeline import CancellationToken, run_pipeline


class JsonLineReporter:
    def emit(self, event: PipelineEvent) -> None:
        if event.kind == PipelineEventKind.RESULT:
            return
        payload = {
            "type": "event",
            "kind": event.kind.value,
            "stage": event.stage.value if event.stage else None,
            "message": event.message,
            "warning": event.warning,
            "stats": event.stats,
        }
        _write_json_line(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ummfiltered-gui-worker",
        description="JSONL sidecar worker for the UmmFiltered desktop app",
    )
    parser.add_argument("--request-file", type=Path, required=True, help="Path to a JSON request file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reporter = JsonLineReporter()
    cancel_token = CancellationToken()
    _install_signal_handlers(cancel_token)

    try:
        request_payload = json.loads(args.request_file.read_text())
        gui_request = gui_request_from_dict(request_payload)
        pipeline_kwargs = resolve_gui_request(gui_request)

        ensure_ffmpeg_tools()
        result = run_pipeline(
            reporter=reporter,
            cancel_token=cancel_token,
            **pipeline_kwargs,
        )
        _write_json_line(
            {
                "type": "result",
                "result": _serialize_result(result),
            }
        )
        return 0 if result.finalStatus != PipelineFinalStatus.CANCELLED else 130
    except Exception as exc:  # pragma: no cover - exercised through higher level tests
        _write_json_line(
            {
                "type": "error",
                **_friendly_error_payload(exc),
            }
        )
        return 1


def _write_json_line(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _serialize_result(result: PipelineResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["finalStatus"] = result.finalStatus.value
    return payload


def _friendly_error_payload(exc: Exception) -> dict[str, str]:
    message = str(exc) or exc.__class__.__name__
    lowered = message.lower()
    if isinstance(exc, FileNotFoundError) and ("ffmpeg" in lowered or "ffprobe" in lowered):
        return {
            "code": "missing_ffmpeg",
            "message": "ffmpeg or ffprobe could not be found for processing.",
            "details": message,
        }
    if isinstance(exc, FileNotFoundError) and "rife" in lowered:
        return {
            "code": "missing_interpolator",
            "message": "The frame interpolation binary was unavailable.",
            "details": message,
        }
    if isinstance(exc, FileNotFoundError):
        return {
            "code": "missing_input",
            "message": "The selected input file could not be read.",
            "details": message,
        }
    if isinstance(exc, PermissionError):
        return {
            "code": "unreadable_file",
            "message": "The app does not have permission to read or write one of the selected files.",
            "details": message,
        }
    if isinstance(exc, ImportError):
        return {
            "code": "missing_dependency",
            "message": "A required local transcription dependency is missing.",
            "details": message,
        }
    return {
        "code": "pipeline_failed",
        "message": "Video processing failed before completion.",
        "details": message,
    }


def _install_signal_handlers(cancel_token: CancellationToken) -> None:
    def _handler(_signum, _frame) -> None:
        cancel_token.cancel()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


if __name__ == "__main__":
    raise SystemExit(main())
