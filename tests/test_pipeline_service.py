from pathlib import Path

import numpy as np

from ummfiltered.models import (
    DetectionSource,
    FillerSegment,
    PipelineEventKind,
    PipelineStage,
    VideoMetadata,
    Word,
)
from ummfiltered.pipeline import CancellationToken, run_pipeline


class EventCollector:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class TestPipelineService:
    def test_emits_stage_events_in_order_for_no_fillers(self, monkeypatch, tmp_path: Path):
        collector = EventCollector()
        metadata = VideoMetadata(
            codec="h264",
            width=320,
            height=240,
            framerate=30.0,
            bitrate=1_000_000,
            pixel_format="yuv420p",
            duration=2.0,
            audio_codec="aac",
            audio_sample_rate=44100,
            audio_channels=2,
            audio_bitrate=128000,
        )

        monkeypatch.setattr("ummfiltered.pipeline.probe_video", lambda _path: metadata)
        monkeypatch.setattr(
            "ummfiltered.pipeline.extract_audio_pcm",
            lambda _path, sample_rate=16000: (np.zeros(sample_rate, dtype=np.float32), sample_rate),
        )
        monkeypatch.setattr(
            "ummfiltered.pipeline.extract_audio_matrix",
            lambda _path, sample_rate=44100, channel_count=2: (
                np.zeros((sample_rate, channel_count), dtype=np.float32),
                sample_rate,
            ),
        )
        monkeypatch.setattr(
            "ummfiltered.pipeline.transcribe",
            lambda *_args, **_kwargs: [Word("hello", 0.0, 0.5, 0.99)],
        )
        monkeypatch.setattr(
            "ummfiltered.pipeline.extract_room_tone",
            lambda *_args, **_kwargs: np.zeros(1000, dtype=np.float32),
        )
        monkeypatch.setattr("ummfiltered.pipeline.detect_fillers", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            "ummfiltered.pipeline.filter_fillers_by_context",
            lambda fillers, *_args, **_kwargs: fillers,
        )
        monkeypatch.setattr(
            "ummfiltered.pipeline.expand_zero_duration_fillers",
            lambda fillers, _words: fillers,
        )

        result = run_pipeline(
            input_path=tmp_path / "input.mp4",
            output_path=tmp_path / "output.mp4",
            reporter=collector,
        )

        kinds_and_stages = [(event.kind, event.stage) for event in collector.events]
        assert kinds_and_stages[:6] == [
            (PipelineEventKind.STAGE_STARTED, PipelineStage.PROBE),
            (PipelineEventKind.STAGE_COMPLETED, PipelineStage.PROBE),
            (PipelineEventKind.STAGE_STARTED, PipelineStage.EXTRACT_AUDIO),
            (PipelineEventKind.STAGE_COMPLETED, PipelineStage.EXTRACT_AUDIO),
            (PipelineEventKind.STAGE_STARTED, PipelineStage.TRANSCRIBE),
            (PipelineEventKind.STAGE_COMPLETED, PipelineStage.TRANSCRIBE),
        ]
        assert result.finalStatus.value == "no_fillers"

    def test_cancellation_returns_cancelled_result(self, monkeypatch, tmp_path: Path):
        collector = EventCollector()
        token = CancellationToken()
        metadata = VideoMetadata(
            codec="h264",
            width=320,
            height=240,
            framerate=30.0,
            bitrate=1_000_000,
            pixel_format="yuv420p",
            duration=2.0,
            audio_codec="aac",
            audio_sample_rate=44100,
            audio_channels=2,
            audio_bitrate=128000,
        )

        def cancel_after_extract(event) -> None:
            collector.emit(event)
            if event.kind == PipelineEventKind.STAGE_COMPLETED and event.stage == PipelineStage.EXTRACT_AUDIO:
                token.cancel()

        monkeypatch.setattr("ummfiltered.pipeline.probe_video", lambda _path: metadata)
        monkeypatch.setattr(
            "ummfiltered.pipeline.extract_audio_pcm",
            lambda _path, sample_rate=16000: (np.zeros(sample_rate, dtype=np.float32), sample_rate),
        )
        monkeypatch.setattr(
            "ummfiltered.pipeline.extract_audio_matrix",
            lambda _path, sample_rate=44100, channel_count=2: (
                np.zeros((sample_rate, channel_count), dtype=np.float32),
                sample_rate,
            ),
        )

        result = run_pipeline(
            input_path=tmp_path / "input.mp4",
            output_path=tmp_path / "output.mp4",
            reporter=cancel_after_extract,
            cancel_token=token,
        )

        assert result.finalStatus.value == "cancelled"

    def test_dry_run_returns_summary_without_rendering(self, monkeypatch, tmp_path: Path):
        metadata = VideoMetadata(
            codec="h264",
            width=320,
            height=240,
            framerate=30.0,
            bitrate=1_000_000,
            pixel_format="yuv420p",
            duration=2.0,
            audio_codec="aac",
            audio_sample_rate=44100,
            audio_channels=2,
            audio_bitrate=128000,
        )
        filler = FillerSegment(0.4, 0.7, "um", 0.5, DetectionSource.DICTIONARY)

        monkeypatch.setattr("ummfiltered.pipeline.probe_video", lambda _path: metadata)
        monkeypatch.setattr(
            "ummfiltered.pipeline.extract_audio_pcm",
            lambda _path, sample_rate=16000: (np.zeros(sample_rate, dtype=np.float32), sample_rate),
        )
        monkeypatch.setattr(
            "ummfiltered.pipeline.extract_audio_matrix",
            lambda _path, sample_rate=44100, channel_count=2: (
                np.zeros((sample_rate, channel_count), dtype=np.float32),
                sample_rate,
            ),
        )
        monkeypatch.setattr(
            "ummfiltered.pipeline.transcribe",
            lambda *_args, **_kwargs: [Word("um", 0.4, 0.7, 0.5)],
        )
        monkeypatch.setattr(
            "ummfiltered.pipeline.extract_room_tone",
            lambda *_args, **_kwargs: np.zeros(1000, dtype=np.float32),
        )
        monkeypatch.setattr("ummfiltered.pipeline.detect_fillers", lambda *_args, **_kwargs: [filler])
        monkeypatch.setattr(
            "ummfiltered.pipeline.filter_fillers_by_context",
            lambda fillers, *_args, **_kwargs: fillers,
        )
        monkeypatch.setattr(
            "ummfiltered.pipeline.expand_zero_duration_fillers",
            lambda fillers, _words: fillers,
        )

        result = run_pipeline(
            input_path=tmp_path / "input.mp4",
            output_path=tmp_path / "output.mp4",
            dry_run=True,
        )

        assert result.finalStatus.value == "dry_run"
        assert result.removedFillers == 1
