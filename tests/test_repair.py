from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import os

from ummfiltered.edit_plan import build_edit_decision_list
from ummfiltered.models import (
    DetectionSource,
    FillerSegment,
    RepairDecision,
    SeamReport,
    SeamReportEntry,
    Segment,
    TransitionType,
    VerificationResult,
    VideoMetadata,
    Word,
)
from ummfiltered.repair import XTTSRepairBackend, merge_repair_decisions, repair_output_audio


class TestMergeRepairDecisions:
    def test_attaches_repair_metadata_to_matching_seam(self):
        report = SeamReport(entries=[
            SeamReportEntry(
                seam_index=0,
                output_time=1.0,
                chosen_strategy="raw",
                before_score=1.2,
                after_score=1.2,
                left_shift_ms=0.0,
                right_shift_ms=0.0,
                duration_ms=0.0,
                accepted=True,
            )
        ])
        decisions = [
            RepairDecision(
                seam_index=0,
                strategy="source_phrase_patch",
                repair_text="from",
                window_start=0.9,
                window_end=1.1,
                before_score=1.2,
                after_score=0.6,
                accepted=True,
                notes="shift=0ms",
            )
        ]

        merged = merge_repair_decisions(report, decisions)

        assert merged.entries[0].repair_strategy == "source_phrase_patch"
        assert merged.entries[0].repair_text == "from"
        assert merged.entries[0].repair_accepted is True


class TestRepairOutputAudio:
    @patch("ummfiltered.repair._candidate_from_xtts", return_value=None)
    @patch("ummfiltered.repair.replace_audio_track")
    @patch("ummfiltered.repair.extract_audio_matrix")
    def test_repairs_lost_word_with_source_phrase_patch(
        self,
        mock_extract_audio_matrix,
        mock_replace_audio_track,
        _mock_xtts_candidate,
        tmp_path: Path,
    ):
        sample_rate = 1000
        source_audio = np.zeros((2200, 2), dtype=np.float32)
        source_audio[:1000] = 0.2
        source_audio[1000:1300] = 0.35
        source_audio[1300:] = -0.15

        output_audio = np.concatenate([source_audio[:1000], source_audio[1300:]], axis=0)
        mock_extract_audio_matrix.return_value = (output_audio, sample_rate)

        segments = [
            Segment(0.0, 1.0, TransitionType.HARD, 1.0),
            Segment(1.3, 2.2, TransitionType.HARD, 1.0),
        ]
        edit_plan = build_edit_decision_list(segments)
        verification = VerificationResult(
            remaining_fillers=[],
            new_fillers=[],
            lost_words=[Word("from", 1.0, 1.28, 0.99)],
            damaged_words=[],
            audio_discontinuities=[],
            seam_report=SeamReport(entries=[
                SeamReportEntry(
                    seam_index=0,
                    output_time=1.0,
                    chosen_strategy="raw",
                    before_score=1.5,
                    after_score=1.5,
                    left_shift_ms=0.0,
                    right_shift_ms=0.0,
                    duration_ms=0.0,
                    accepted=True,
                )
            ]),
        )
        metadata = VideoMetadata(
            codec="h264",
            width=320,
            height=240,
            framerate=30.0,
            bitrate=1_000_000,
            pixel_format="yuv420p",
            duration=2.2,
            audio_codec="aac",
            audio_sample_rate=sample_rate,
            audio_channels=2,
            audio_bitrate=128000,
        )

        decisions = repair_output_audio(
            output_path=tmp_path / "output.mp4",
            source_audio=source_audio,
            sample_rate=sample_rate,
            metadata=metadata,
            edit_plan=edit_plan,
            verification=verification,
        )

        assert decisions
        assert decisions[0].accepted is True
        assert decisions[0].strategy == "source_phrase_patch"
        mock_replace_audio_track.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("ummfiltered.repair._candidate_from_xtts")
    @patch("ummfiltered.repair._candidate_from_source_audio")
    @patch("ummfiltered.repair.extract_audio_matrix")
    def test_does_not_try_xtts_without_experimental_flag(
        self,
        mock_extract_audio_matrix,
        mock_source_candidate,
        mock_xtts_candidate,
        tmp_path: Path,
    ):
        sample_rate = 1000
        source_audio = np.zeros((2000, 2), dtype=np.float32)
        output_audio = np.zeros((1500, 2), dtype=np.float32)
        mock_extract_audio_matrix.return_value = (output_audio, sample_rate)
        mock_source_candidate.return_value = None
        metadata = VideoMetadata(
            codec="h264",
            width=320,
            height=240,
            framerate=30.0,
            bitrate=1_000_000,
            pixel_format="yuv420p",
            duration=2.0,
            audio_codec="aac",
            audio_sample_rate=sample_rate,
            audio_channels=2,
            audio_bitrate=128000,
        )
        verification = VerificationResult(
            remaining_fillers=[],
            new_fillers=[],
            lost_words=[Word("from", 1.0, 1.25, 0.99)],
            damaged_words=[],
            audio_discontinuities=[],
            seam_report=SeamReport(entries=[
                SeamReportEntry(
                    seam_index=0,
                    output_time=1.0,
                    chosen_strategy="raw",
                    before_score=1.5,
                    after_score=1.5,
                    left_shift_ms=0.0,
                    right_shift_ms=0.0,
                    duration_ms=0.0,
                    accepted=True,
                )
            ]),
        )
        edit_plan = build_edit_decision_list([
            Segment(0.0, 1.0, TransitionType.HARD, 1.0),
            Segment(1.3, 2.0, TransitionType.HARD, 1.0),
        ])

        repair_output_audio(
            output_path=tmp_path / "output.mp4",
            source_audio=source_audio,
            sample_rate=sample_rate,
            metadata=metadata,
            edit_plan=edit_plan,
            verification=verification,
        )

        mock_xtts_candidate.assert_not_called()

    @patch.dict(os.environ, {"UMMFILTERED_EXPERIMENTAL_AI_REPAIR": "1"}, clear=True)
    @patch("ummfiltered.repair.replace_audio_track")
    @patch("ummfiltered.repair._candidate_from_xtts")
    @patch("ummfiltered.repair._candidate_from_source_audio")
    @patch("ummfiltered.repair.extract_audio_matrix")
    def test_tries_xtts_only_for_tiny_repairs_when_experimental_flag_is_on(
        self,
        mock_extract_audio_matrix,
        mock_source_candidate,
        mock_xtts_candidate,
        mock_replace_audio_track,
        tmp_path: Path,
    ):
        sample_rate = 1000
        source_audio = np.zeros((2000, 2), dtype=np.float32)
        output_audio = np.zeros((1500, 2), dtype=np.float32)
        mock_extract_audio_matrix.return_value = (output_audio, sample_rate)
        mock_source_candidate.return_value = MagicMock(
            strategy="source_phrase_patch",
            audio=output_audio,
            before_score=0.4,
            after_score=0.8,
            notes="weak",
        )
        mock_xtts_candidate.return_value = MagicMock(
            strategy="xtts_phrase_patch",
            audio=output_audio,
            before_score=0.4,
            after_score=0.3,
            notes="candidate=0",
        )
        metadata = VideoMetadata(
            codec="h264",
            width=320,
            height=240,
            framerate=30.0,
            bitrate=1_000_000,
            pixel_format="yuv420p",
            duration=2.0,
            audio_codec="aac",
            audio_sample_rate=sample_rate,
            audio_channels=2,
            audio_bitrate=128000,
        )
        verification = VerificationResult(
            remaining_fillers=[],
            new_fillers=[],
            lost_words=[Word("from", 1.0, 1.25, 0.99)],
            damaged_words=[],
            audio_discontinuities=[],
            seam_report=SeamReport(entries=[
                SeamReportEntry(
                    seam_index=0,
                    output_time=1.0,
                    chosen_strategy="raw",
                    before_score=1.5,
                    after_score=1.5,
                    left_shift_ms=0.0,
                    right_shift_ms=0.0,
                    duration_ms=0.0,
                    accepted=True,
                )
            ]),
        )
        edit_plan = build_edit_decision_list([
            Segment(0.0, 1.0, TransitionType.HARD, 1.0),
            Segment(1.3, 2.0, TransitionType.HARD, 1.0),
        ])

        decisions = repair_output_audio(
            output_path=tmp_path / "output.mp4",
            source_audio=source_audio,
            sample_rate=sample_rate,
            metadata=metadata,
            edit_plan=edit_plan,
            verification=verification,
        )

        mock_xtts_candidate.assert_called_once()
        assert decisions[0].strategy == "xtts_phrase_patch"
        mock_replace_audio_track.assert_called_once()


class TestXTTSRuntimeProvisioning:
    def test_installs_runtime_when_missing(self):
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = False
        fake_model = MagicMock()
        fake_tts_cls = MagicMock(return_value=fake_model)
        fake_tts_api = MagicMock(TTS=fake_tts_cls)

        with patch("ummfiltered.repair._install_python_dependency") as mock_install_dependency:
            with patch("ummfiltered.repair.importlib.import_module") as mock_import_module:
                mock_import_module.side_effect = [
                    ImportError("missing TTS"),
                    fake_tts_api,
                    fake_torch,
                ]

                backend = XTTSRepairBackend()
                model = backend._ensure_runtime()

        assert model is fake_model
        assert mock_install_dependency.call_args_list[0].args[0] == "torch>=2.2.0"
        assert mock_install_dependency.call_args_list[1].args[0] == "TTS>=0.22.0"
