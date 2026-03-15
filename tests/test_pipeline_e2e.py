import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np

from ummfiltered.models import (
    DetectionSource,
    FillerSegment,
    Segment,
    TransitionType,
    VerificationResult,
    VideoMetadata,
    Word,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "test.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest", str(video_path),
        ],
        capture_output=True,
        check=True,
    )
    return video_path


class TestCLIParsing:
    def test_default_output_name(self):
        from ummfiltered.cli import parse_args
        args = parse_args(["video.mp4"])
        assert args.output == Path("video_ummfiltered.mp4")

    def test_custom_output(self):
        from ummfiltered.cli import parse_args
        args = parse_args(["video.mp4", "-o", "clean.mp4"])
        assert args.output == Path("clean.mp4")

    def test_all_flags(self):
        from ummfiltered.cli import parse_args
        args = parse_args([
            "video.mp4", "--interactive", "--aggressive",
            "--cloud", "deepgram", "--model-size", "large",
            "--quality", "lossless", "--dry-run",
            "--fillers", "um,uh,like",
        ])
        assert args.interactive is True
        assert args.aggressive is True
        assert args.cloud == "deepgram"
        assert args.model_size == "large"
        assert args.quality == "lossless"
        assert args.dry_run is True
        assert args.fillers == "um,uh,like"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
class TestVideoProbing:
    def test_probe_sample_video(self, sample_video: Path):
        from ummfiltered.render import probe_video
        meta = probe_video(sample_video)
        assert meta.width == 320
        assert meta.height == 240
        assert meta.framerate == 30.0
        assert meta.duration > 4.0


class TestCLIParsingRefine:
    def test_no_refine_flag(self):
        from ummfiltered.cli import parse_args
        args = parse_args(["video.mp4", "--no-refine"])
        assert args.no_refine is True

    def test_refine_default_on(self):
        from ummfiltered.cli import parse_args
        args = parse_args(["video.mp4"])
        assert args.no_refine is False


class TestPipelineRefinement:
    @patch("ummfiltered.pipeline.shutil.copy2")
    @patch("ummfiltered.pipeline._render_with_audio")
    @patch("ummfiltered.pipeline.verify_output")
    @patch("ummfiltered.pipeline._classify_and_generate_transitions")
    @patch("ummfiltered.pipeline.protect_adjacent_words")
    @patch("ummfiltered.pipeline.find_silence_boundaries")
    @patch("ummfiltered.pipeline.expand_zero_duration_fillers")
    @patch("ummfiltered.pipeline.filter_fillers_by_context")
    @patch("ummfiltered.pipeline.detect_fillers")
    @patch("ummfiltered.pipeline.transcribe")
    @patch("ummfiltered.pipeline.extract_audio_matrix")
    @patch("ummfiltered.pipeline.extract_audio_pcm")
    @patch("ummfiltered.pipeline.probe_video")
    def test_rerenders_when_remaining_fillers_found(
        self,
        mock_probe_video,
        mock_extract_audio_pcm,
        mock_extract_audio_matrix,
        mock_transcribe,
        mock_detect_fillers,
        mock_filter_fillers,
        mock_expand_zero,
        mock_find_silence,
        mock_protect,
        mock_classify,
        mock_verify,
        mock_render_with_audio,
        mock_copy2,
        tmp_path: Path,
    ):
        from ummfiltered.pipeline import run_pipeline

        words = [
            Word("hello", 0.0, 0.5, 0.99),
            Word("um", 0.6, 0.8, 0.40),
            Word("world", 0.9, 1.4, 0.99),
        ]
        filler = FillerSegment(0.6, 0.8, "um", 0.40, DetectionSource.DICTIONARY)
        segments = [
            Segment(0.0, 0.6, TransitionType.HARD, 1.0),
            Segment(0.8, 2.0, TransitionType.HARD, 1.0),
        ]
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
        clean_result = VerificationResult(
            remaining_fillers=[],
            new_fillers=[],
            lost_words=[],
            damaged_words=[],
            audio_discontinuities=[],
            preserved_word_recall=1.0,
            max_missing_run=0,
        )

        mock_probe_video.return_value = metadata
        mock_extract_audio_pcm.side_effect = [
            (np.zeros(16000 * 2, dtype=np.float32), 16000),
            (np.zeros(16000 * 2, dtype=np.float32), 16000),
        ]
        mock_extract_audio_matrix.return_value = (
            np.zeros((44100 * 2, 2), dtype=np.float32),
            44100,
        )
        mock_transcribe.side_effect = [words, words]
        mock_detect_fillers.side_effect = [[filler], []]
        mock_filter_fillers.side_effect = lambda fillers, *_args, **_kwargs: fillers
        mock_expand_zero.side_effect = lambda fillers, _words: fillers
        mock_find_silence.side_effect = lambda samples, sample_rate, start, end, **kwargs: (start, end)
        mock_protect.side_effect = lambda start, end, *_args, **_kwargs: (start, end)
        mock_classify.return_value = (segments, {})
        mock_verify.side_effect = [
            VerificationResult(
                remaining_fillers=[filler],
                new_fillers=[],
                lost_words=[],
                damaged_words=[],
                audio_discontinuities=[],
                preserved_word_recall=0.99,
                max_missing_run=1,
            ),
            clean_result,
        ]

        run_pipeline(
            input_path=tmp_path / "input.mp4",
            output_path=tmp_path / "output.mp4",
            no_refine=False,
        )

        assert mock_render_with_audio.call_count == 2
        assert mock_verify.call_count == 2
        assert mock_copy2.call_count >= 1

    @patch("ummfiltered.pipeline.shutil.copy2")
    @patch("ummfiltered.pipeline._render_with_audio")
    @patch("ummfiltered.pipeline.verify_output")
    @patch("ummfiltered.pipeline._classify_and_generate_transitions")
    @patch("ummfiltered.pipeline.protect_adjacent_words")
    @patch("ummfiltered.pipeline.find_silence_boundaries")
    @patch("ummfiltered.pipeline.expand_zero_duration_fillers")
    @patch("ummfiltered.pipeline.filter_fillers_by_context")
    @patch("ummfiltered.pipeline.detect_fillers")
    @patch("ummfiltered.pipeline.transcribe")
    @patch("ummfiltered.pipeline.extract_audio_matrix")
    @patch("ummfiltered.pipeline.extract_audio_pcm")
    @patch("ummfiltered.pipeline.probe_video")
    def test_keeps_best_earlier_render_when_later_pass_is_worse(
        self,
        mock_probe_video,
        mock_extract_audio_pcm,
        mock_extract_audio_matrix,
        mock_transcribe,
        mock_detect_fillers,
        mock_filter_fillers,
        mock_expand_zero,
        mock_find_silence,
        mock_protect,
        mock_classify,
        mock_verify,
        mock_render_with_audio,
        mock_copy2,
        tmp_path: Path,
    ):
        from ummfiltered.pipeline import run_pipeline

        words = [
            Word("hello", 0.0, 0.5, 0.99),
            Word("um", 0.6, 0.8, 0.40),
            Word("world", 0.9, 1.4, 0.99),
        ]
        filler = FillerSegment(0.6, 0.8, "um", 0.40, DetectionSource.DICTIONARY)
        worse_filler = FillerSegment(1.2, 1.4, "uh", 0.40, DetectionSource.DICTIONARY)
        segments = [
            Segment(0.0, 0.6, TransitionType.HARD, 1.0),
            Segment(0.8, 2.0, TransitionType.HARD, 1.0),
        ]
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

        mock_probe_video.return_value = metadata
        mock_extract_audio_pcm.side_effect = [
            (np.zeros(16000 * 2, dtype=np.float32), 16000),
            (np.zeros(16000 * 2, dtype=np.float32), 16000),
        ]
        mock_extract_audio_matrix.return_value = (
            np.zeros((44100 * 2, 2), dtype=np.float32),
            44100,
        )
        mock_transcribe.side_effect = [words, words]
        mock_detect_fillers.side_effect = [[filler], []]
        mock_filter_fillers.side_effect = lambda fillers, *_args, **_kwargs: fillers
        mock_expand_zero.side_effect = lambda fillers, _words: fillers
        mock_find_silence.side_effect = lambda samples, sample_rate, start, end, **kwargs: (start, end)
        mock_protect.side_effect = lambda start, end, *_args, **_kwargs: (start, end)
        mock_classify.return_value = (segments, {})
        mock_verify.side_effect = [
            VerificationResult(
                remaining_fillers=[filler],
                new_fillers=[],
                lost_words=[],
                damaged_words=[],
                audio_discontinuities=[],
                preserved_word_recall=0.99,
                max_missing_run=1,
            ),
            VerificationResult(
                remaining_fillers=[filler, worse_filler],
                new_fillers=[],
                lost_words=[],
                damaged_words=[],
                audio_discontinuities=[],
                preserved_word_recall=0.98,
                max_missing_run=2,
            ),
        ]

        run_pipeline(
            input_path=tmp_path / "input.mp4",
            output_path=tmp_path / "output.mp4",
            no_refine=False,
        )

        assert mock_render_with_audio.call_count == 2
        assert mock_verify.call_count == 2
        assert mock_copy2.call_count >= 2

    @patch("ummfiltered.pipeline.shutil.copy2")
    @patch("ummfiltered.pipeline._render_with_audio")
    @patch("ummfiltered.pipeline.verify_output")
    @patch("ummfiltered.pipeline._classify_and_generate_transitions")
    @patch("ummfiltered.pipeline.protect_adjacent_words")
    @patch("ummfiltered.pipeline.find_silence_boundaries")
    @patch("ummfiltered.pipeline.expand_zero_duration_fillers")
    @patch("ummfiltered.pipeline.filter_fillers_by_context")
    @patch("ummfiltered.pipeline.detect_fillers")
    @patch("ummfiltered.pipeline.transcribe")
    @patch("ummfiltered.pipeline.extract_audio_matrix")
    @patch("ummfiltered.pipeline.extract_audio_pcm")
    @patch("ummfiltered.pipeline.probe_video")
    def test_stops_when_only_audio_issues_plateau(
        self,
        mock_probe_video,
        mock_extract_audio_pcm,
        mock_extract_audio_matrix,
        mock_transcribe,
        mock_detect_fillers,
        mock_filter_fillers,
        mock_expand_zero,
        mock_find_silence,
        mock_protect,
        mock_classify,
        mock_verify,
        mock_render_with_audio,
        mock_copy2,
        tmp_path: Path,
    ):
        from ummfiltered.pipeline import run_pipeline

        words = [
            Word("hello", 0.0, 0.5, 0.99),
            Word("um", 0.6, 0.8, 0.40),
            Word("world", 0.9, 1.4, 0.99),
        ]
        filler = FillerSegment(0.6, 0.8, "um", 0.40, DetectionSource.DICTIONARY)
        segments = [
            Segment(0.0, 0.6, TransitionType.HARD, 1.0),
            Segment(0.8, 2.0, TransitionType.HARD, 1.0),
        ]
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
        plateau_result = VerificationResult(
            remaining_fillers=[],
            new_fillers=[],
            lost_words=[],
            damaged_words=[],
            audio_discontinuities=[(0.6, 8.0)],
            preserved_word_recall=1.0,
            max_missing_run=0,
        )

        mock_probe_video.return_value = metadata
        mock_extract_audio_pcm.side_effect = [
            (np.zeros(16000 * 2, dtype=np.float32), 16000),
            (np.zeros(16000 * 2, dtype=np.float32), 16000),
        ]
        mock_extract_audio_matrix.return_value = (
            np.zeros((44100 * 2, 2), dtype=np.float32),
            44100,
        )
        mock_transcribe.side_effect = [words, words]
        mock_detect_fillers.side_effect = [[filler], []]
        mock_filter_fillers.side_effect = lambda fillers, *_args, **_kwargs: fillers
        mock_expand_zero.side_effect = lambda fillers, _words: fillers
        mock_find_silence.side_effect = lambda samples, sample_rate, start, end, **kwargs: (start, end)
        mock_protect.side_effect = lambda start, end, *_args, **_kwargs: (start, end)
        mock_classify.return_value = (segments, {})
        mock_verify.side_effect = [plateau_result, plateau_result]

        run_pipeline(
            input_path=tmp_path / "input.mp4",
            output_path=tmp_path / "output.mp4",
            no_refine=False,
        )

        assert mock_render_with_audio.call_count == 2
        assert mock_verify.call_count == 2

    @patch("ummfiltered.pipeline.shutil.copy2")
    @patch("ummfiltered.pipeline._render_with_audio")
    @patch("ummfiltered.pipeline.verify_output")
    @patch("ummfiltered.pipeline._classify_and_generate_transitions")
    @patch("ummfiltered.pipeline.protect_adjacent_words")
    @patch("ummfiltered.pipeline.find_silence_boundaries")
    @patch("ummfiltered.pipeline.expand_zero_duration_fillers")
    @patch("ummfiltered.pipeline.filter_fillers_by_context")
    @patch("ummfiltered.pipeline.detect_fillers")
    @patch("ummfiltered.pipeline.transcribe")
    @patch("ummfiltered.pipeline.extract_audio_matrix")
    @patch("ummfiltered.pipeline.extract_audio_pcm")
    @patch("ummfiltered.pipeline.probe_video")
    def test_does_not_plateau_when_words_are_missing(
        self,
        mock_probe_video,
        mock_extract_audio_pcm,
        mock_extract_audio_matrix,
        mock_transcribe,
        mock_detect_fillers,
        mock_filter_fillers,
        mock_expand_zero,
        mock_find_silence,
        mock_protect,
        mock_classify,
        mock_verify,
        mock_render_with_audio,
        mock_copy2,
        tmp_path: Path,
    ):
        from ummfiltered.pipeline import run_pipeline

        words = [
            Word("hello", 0.0, 0.5, 0.99),
            Word("um", 0.6, 0.8, 0.40),
            Word("world", 0.9, 1.4, 0.99),
        ]
        filler = FillerSegment(0.6, 0.8, "um", 0.40, DetectionSource.DICTIONARY)
        segments = [
            Segment(0.0, 0.6, TransitionType.HARD, 1.0),
            Segment(0.8, 2.0, TransitionType.HARD, 1.0),
        ]
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
        missing_word_result = VerificationResult(
            remaining_fillers=[],
            new_fillers=[],
            lost_words=[Word("hello", 0.0, 0.5, 0.95)],
            damaged_words=[],
            audio_discontinuities=[(0.6, 8.0)],
            preserved_word_recall=0.70,
            max_missing_run=4,
        )
        improved_result = VerificationResult(
            remaining_fillers=[],
            new_fillers=[],
            lost_words=[],
            damaged_words=[],
            audio_discontinuities=[(0.6, 8.0)],
            preserved_word_recall=0.99,
            max_missing_run=0,
        )

        mock_probe_video.return_value = metadata
        mock_extract_audio_pcm.side_effect = [
            (np.zeros(16000 * 2, dtype=np.float32), 16000),
            (np.zeros(16000 * 2, dtype=np.float32), 16000),
        ]
        mock_extract_audio_matrix.return_value = (
            np.zeros((44100 * 2, 2), dtype=np.float32),
            44100,
        )
        mock_transcribe.side_effect = [words, words]
        mock_detect_fillers.side_effect = [[filler], []]
        mock_filter_fillers.side_effect = lambda fillers, *_args, **_kwargs: fillers
        mock_expand_zero.side_effect = lambda fillers, _words: fillers
        mock_find_silence.side_effect = lambda samples, sample_rate, start, end, **kwargs: (start, end)
        mock_protect.side_effect = lambda start, end, *_args, **_kwargs: (start, end)
        mock_classify.return_value = (segments, {})
        mock_verify.side_effect = [missing_word_result, improved_result, improved_result]

        run_pipeline(
            input_path=tmp_path / "input.mp4",
            output_path=tmp_path / "output.mp4",
            no_refine=False,
        )

        assert mock_render_with_audio.call_count == 3
        assert mock_verify.call_count == 3
