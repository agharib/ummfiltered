from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch
import wave

import numpy as np
import pytest

from ummfiltered.audio import extract_audio_pcm
from ummfiltered.ffmpeg_tools import ensure_ffmpeg_tools
from ummfiltered.render import _extract_segments, add_padding, build_segment_filter, render_video, replace_audio_track
from ummfiltered.models import Segment, TransitionType, VideoMetadata

try:
    ensure_ffmpeg_tools()
    HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
except Exception:
    HAS_FFMPEG = False


class TestBuildSegmentFilter:
    def test_single_segment(self):
        segs = [Segment(0.0, 5.0, TransitionType.HARD, 1.0)]
        vf, af = build_segment_filter(segs)
        assert "0.0" in vf
        assert "5.0" in vf

    def test_multiple_segments(self):
        segs = [
            Segment(0.0, 3.0, TransitionType.HARD, 1.0),
            Segment(4.0, 8.0, TransitionType.HARD, 0.9),
        ]
        vf, af = build_segment_filter(segs)
        assert "3.0" in vf
        assert "4.0" in vf


class TestAddPadding:
    def test_default_padding_does_not_change_segments(self):
        segs = [
            Segment(0.0, 3.0, TransitionType.HARD, 1.0),
            Segment(4.0, 8.0, TransitionType.HARD, 0.9),
        ]
        padded = add_padding(segs, video_duration=8.0)
        assert [(s.start, s.end) for s in padded] == [(0.0, 3.0), (4.0, 8.0)]


class TestRenderVideo:
    @patch("ummfiltered.render._render_concat")
    @patch("ummfiltered.render._render_interpolation_clip")
    @patch("ummfiltered.render._extract_segments")
    @patch("ummfiltered.render.add_padding")
    def test_inserts_interpolation_clip_between_segments(
        self,
        mock_add_padding,
        mock_extract_segments,
        mock_render_interpolation,
        mock_render_concat,
        tmp_path: Path,
    ):
        segments = [
            Segment(0.0, 2.0, TransitionType.HARD, 1.0),
            Segment(3.0, 5.0, TransitionType.INTERPOLATE, 0.4),
        ]
        metadata = VideoMetadata(
            codec="h264",
            width=320,
            height=240,
            framerate=30.0,
            bitrate=1_000_000,
            pixel_format="yuv420p",
            duration=5.0,
            audio_codec="aac",
            audio_sample_rate=44100,
            audio_channels=2,
            audio_bitrate=128000,
        )
        seg0 = tmp_path / "seg0.ts"
        seg1 = tmp_path / "seg1.ts"
        interp = tmp_path / "interp.ts"

        mock_add_padding.return_value = segments
        mock_extract_segments.return_value = [(0, seg0), (1, seg1)]
        mock_render_interpolation.return_value = interp

        render_video(
            input_path=tmp_path / "input.mp4",
            output_path=tmp_path / "output.mp4",
            segments=segments,
            metadata=metadata,
            interpolated_frames={1: [np.zeros((10, 10, 3), dtype=np.uint8)] * 3},
        )

        ordered_files = mock_render_concat.call_args.args[0]
        assert ordered_files == [seg0, interp, seg1]


class TestExtractSegments:
    def test_uses_trim_filters_for_segment_trimming(self, tmp_path: Path):
        segments = [
            Segment(1.25, 2.0, TransitionType.HARD, 1.0),
            Segment(3.0, 4.0, TransitionType.HARD, 1.0),
        ]
        metadata = VideoMetadata(
            codec="h264",
            width=320,
            height=240,
            framerate=30.0,
            bitrate=1_000_000,
            pixel_format="yuv420p",
            duration=5.0,
            audio_codec="aac",
            audio_sample_rate=44100,
            audio_channels=2,
            audio_bitrate=128000,
        )
        seen_commands: list[list[str]] = []

        def fake_run(cmd, capture_output, check):
            seen_commands.append(cmd)

        with patch("ummfiltered.render.subprocess.run", side_effect=fake_run):
            _extract_segments(
                input_path=tmp_path / "input.mp4",
                padded=segments,
                metadata=metadata,
                tmpdir=str(tmp_path),
                crossfade_s=0.04,
            )

        assert seen_commands
        first_cmd = seen_commands[0]
        assert "-filter:v" in first_cmd
        filter_graph = first_cmd[first_cmd.index("-filter:v") + 1]
        assert "trim=start=1.2500:duration=0.7500" in filter_graph
        assert "-an" in first_cmd

    @pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
    def test_renders_video_only_segments_for_separate_audio_pipeline(self, tmp_path: Path):
        ensure_ffmpeg_tools()
        input_path = tmp_path / "sample.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
                "-c:v", "libx264", "-c:a", "aac",
                "-shortest", str(input_path),
            ],
            capture_output=True,
            check=True,
        )
        metadata = VideoMetadata(
            codec="h264",
            width=320,
            height=240,
            framerate=30.0,
            bitrate=1_000_000,
            pixel_format="yuv420p",
            duration=5.0,
            audio_codec="aac",
            audio_sample_rate=44100,
            audio_channels=1,
            audio_bitrate=128000,
        )
        segments = [
            Segment(0.0, 1.0, TransitionType.HARD, 1.0),
            Segment(2.0, 3.0, TransitionType.HARD, 1.0),
        ]

        seg_files = _extract_segments(
            input_path=input_path,
            padded=segments,
            metadata=metadata,
            tmpdir=str(tmp_path),
            crossfade_s=0.04,
        )

        later_path = seg_files[1][1]
        probe = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", str(later_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert '"codec_type": "audio"' not in probe.stdout


class TestReplaceAudioTrack:
    def test_writes_audio_with_source_channel_count(self, tmp_path: Path):
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"video")
        metadata = VideoMetadata(
            codec="h264",
            width=320,
            height=240,
            framerate=30.0,
            bitrate=1_000_000,
            pixel_format="yuv420p",
            duration=1.0,
            audio_codec="aac",
            audio_sample_rate=44100,
            audio_channels=2,
            audio_bitrate=128000,
        )
        samples = np.linspace(-0.2, 0.2, 1000, dtype=np.float32)

        def fake_run(cmd, capture_output, check):
            wav_path = Path(cmd[5])
            with wave.open(str(wav_path), "rb") as wf:
                assert wf.getnchannels() == 2
                assert wf.getframerate() == 44100
            Path(cmd[-1]).write_bytes(b"rendered")

        with patch("ummfiltered.render.subprocess.run", side_effect=fake_run):
            replace_audio_track(video_path, samples, 44100, metadata)

        assert video_path.exists()
