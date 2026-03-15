import json
from pathlib import Path

from ummfiltered.gui_worker import main
from ummfiltered.models import PipelineEvent, PipelineEventKind, PipelineFinalStatus, PipelineResult, PipelineStage


class TestGuiWorker:
    def test_emits_jsonl_events_and_result(self, monkeypatch, tmp_path: Path, capsys):
        request_path = tmp_path / "request.json"
        input_path = tmp_path / "in.mp4"
        input_path.write_bytes(b"video")
        request_path.write_text(
            json.dumps(
                {
                    "inputPath": str(input_path),
                    "preset": "balanced",
                    "aggressive": False,
                    "verifyPass": True,
                    "naturalPauses": True,
                    "overrides": {},
                }
            )
        )

        def fake_run_pipeline(*, reporter, cancel_token, **kwargs):
            reporter.emit(
                PipelineEvent(
                    kind=PipelineEventKind.STAGE_STARTED,
                    stage=PipelineStage.TRANSCRIBE,
                    message="Transcribing speech...",
                )
            )
            assert kwargs["model_size"] == "medium"
            return PipelineResult(
                outputPath="/tmp/out.mp4",
                removedFillers=5,
                removedSeconds=1.25,
                finalStatus=PipelineFinalStatus.SUCCESS,
            )

        monkeypatch.setattr("ummfiltered.gui_worker.ensure_ffmpeg_tools", lambda: None)
        monkeypatch.setattr("ummfiltered.gui_worker.run_pipeline", fake_run_pipeline)

        exit_code = main(["--request-file", str(request_path)])

        captured = capsys.readouterr().out.strip().splitlines()
        assert exit_code == 0
        assert len(captured) == 2
        first_line = json.loads(captured[0])
        result_line = json.loads(captured[1])
        assert first_line["type"] == "event"
        assert first_line["kind"] == "stage_started"
        assert first_line["stage"] == "transcribe"
        assert result_line["type"] == "result"
        assert result_line["result"]["removedFillers"] == 5
        assert result_line["result"]["finalStatus"] == "success"

    def test_maps_missing_ffmpeg_errors(self, monkeypatch, tmp_path: Path, capsys):
        request_path = tmp_path / "request.json"
        input_path = tmp_path / "in.mp4"
        input_path.write_bytes(b"video")
        request_path.write_text(
            json.dumps(
                {
                    "inputPath": str(input_path),
                    "preset": "speed",
                    "aggressive": False,
                    "verifyPass": False,
                    "naturalPauses": False,
                    "overrides": {},
                }
            )
        )

        monkeypatch.setattr(
            "ummfiltered.gui_worker.ensure_ffmpeg_tools",
            lambda: (_ for _ in ()).throw(FileNotFoundError("ffmpeg not found")),
        )

        exit_code = main(["--request-file", str(request_path)])

        captured = json.loads(capsys.readouterr().out.strip())
        assert exit_code == 1
        assert captured["type"] == "error"
        assert captured["code"] == "missing_ffmpeg"

    def test_maps_missing_input_errors(self, tmp_path: Path, capsys):
        request_path = tmp_path / "request.json"
        request_path.write_text(
            json.dumps(
                {
                    "inputPath": str(tmp_path / "missing.mp4"),
                    "preset": "balanced",
                    "aggressive": False,
                    "verifyPass": True,
                    "naturalPauses": True,
                    "overrides": {},
                }
            )
        )

        exit_code = main(["--request-file", str(request_path)])

        captured = json.loads(capsys.readouterr().out.strip())
        assert exit_code == 1
        assert captured["type"] == "error"
        assert captured["code"] == "missing_input"

    def test_maps_first_run_ffmpeg_download_failures(self, monkeypatch, tmp_path: Path, capsys):
        request_path = tmp_path / "request.json"
        input_path = tmp_path / "in.mp4"
        input_path.write_bytes(b"video")
        request_path.write_text(
            json.dumps(
                {
                    "inputPath": str(input_path),
                    "preset": "balanced",
                    "aggressive": False,
                    "verifyPass": True,
                    "naturalPauses": True,
                    "overrides": {},
                }
            )
        )

        monkeypatch.setattr(
            "ummfiltered.gui_worker.ensure_ffmpeg_tools",
            lambda: (_ for _ in ()).throw(
                RuntimeError("Unable to download bundled ffmpeg tools for first-run setup.")
            ),
        )

        exit_code = main(["--request-file", str(request_path)])

        captured = json.loads(capsys.readouterr().out.strip())
        assert exit_code == 1
        assert captured["type"] == "error"
        assert captured["code"] == "missing_ffmpeg"
