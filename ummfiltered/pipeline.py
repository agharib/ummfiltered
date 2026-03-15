from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Protocol

import numpy as np
from rich.console import Console

from ummfiltered.audio import (
    assemble_audio_track,
    compute_rms_db,
    extract_audio_matrix,
    extract_audio_pcm,
    extract_room_tone,
    find_silence_boundaries,
    protect_adjacent_words,
)
from ummfiltered.config import (
    CROSSFADE_MS,
    FILLER_MARGIN_END_MS,
    FILLER_MARGIN_START_MS,
    MAX_EXPANSION_MS,
    PHRASE_MARGIN_END_BONUS_MS,
    PHRASE_MARGIN_START_BONUS_MS,
    SILENCE_THRESHOLD_DB,
    compute_adaptive_pause,
)
from ummfiltered.cut_planner import build_keep_segments, classify_transitions
from ummfiltered.detect import (
    detect_fillers,
    expand_zero_duration_fillers,
    filter_fillers_by_context,
)
from ummfiltered.edit_plan import build_edit_decision_list
from ummfiltered.interpolate import interpolate_frames
from ummfiltered.interpolator_tools import ensure_interpolator_backend
from ummfiltered.models import (
    CutAdjustment,
    FillerSegment,
    PipelineEvent,
    PipelineEventKind,
    PipelineFinalStatus,
    PipelineResult,
    PipelineStage,
    Segment,
    TransitionType,
    VideoMetadata,
)
from ummfiltered.repair import merge_repair_decisions, repair_output_audio
from ummfiltered.render import get_frame_at_time, probe_video, render_video, replace_audio_track
from ummfiltered.verify import apply_adjustments, build_reference_contract, rebuild_cuts, verify_output
from ummfiltered.transcribe import transcribe

console = Console()
RECALL_THRESHOLD = 0.98
MAX_MISSING_RUN = 3
SEAM_REGRESSION_MARGIN = 0.15


class ReporterLike(Protocol):
    def emit(self, event: PipelineEvent) -> None: ...


class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class PipelineCancelledError(RuntimeError):
    pass


class RichPipelineReporter:
    _stage_prefix = {
        PipelineStage.PROBE: "  📼",
        PipelineStage.EXTRACT_AUDIO: "  🎙️",
        PipelineStage.TRANSCRIBE: "  🧠",
        PipelineStage.DETECT_FILLERS: "  🔍",
        PipelineStage.PLAN_CUTS: "  ✂️",
        PipelineStage.RENDER: "  🧵",
        PipelineStage.VERIFY: "  🔄",
        PipelineStage.FINAL_CHECK: "  ✅",
    }

    def __init__(self, rich_console: Console | None = None) -> None:
        self.console = rich_console or console

    def emit(self, event: PipelineEvent) -> None:
        prefix = self._stage_prefix.get(event.stage, "  •")
        if event.kind == PipelineEventKind.STAGE_STARTED:
            self.console.print(f"{prefix} [bold]{event.message}[/bold]")
            return
        if event.kind == PipelineEventKind.STAGE_COMPLETED:
            self.console.print(f"{prefix} [dim]{event.message}[/dim] [green]done[/green]")
            return
        if event.kind == PipelineEventKind.INFO:
            self.console.print(f"{prefix} [dim]{event.message}[/dim]")
            return
        if event.kind == PipelineEventKind.WARNING:
            self.console.print(f"{prefix} [yellow]{event.warning or event.message}[/yellow]")
            return
        if event.kind == PipelineEventKind.ERROR:
            self.console.print(f"{prefix} [red]{event.message}[/red]")
            return
        if event.kind == PipelineEventKind.CANCELLED:
            self.console.print(f"{prefix} [yellow]{event.message}[/yellow]")
            return
        if event.kind == PipelineEventKind.RESULT and event.stats:
            removed_fillers = event.stats.get("removedFillers", 0)
            removed_seconds = float(event.stats.get("removedSeconds", 0.0))
            output_path = event.stats.get("outputPath", "")
            if event.stats.get("finalStatus") == PipelineFinalStatus.NO_FILLERS.value:
                self.console.print("  [green bold]✨ No filler words detected! Already clean.[/green bold]\n")
                return
            if event.stats.get("finalStatus") == PipelineFinalStatus.DRY_RUN.value:
                self.console.print(
                    f"  [yellow]Dry run:[/yellow] would remove [bold]{removed_fillers}[/bold] fillers"
                )
                return
            self.console.print()
            self.console.print(f"  [green bold]✨ {removed_fillers} filler words removed[/green bold]")
            self.console.print(
                f"  [green bold]⏱️  {_format_duration(removed_seconds)} of dead air eliminated[/green bold]"
            )
            self.console.print("  [green bold]🔊 Audio gaps closed naturally[/green bold]")
            self.console.print(f"\n  [bold]Saved to[/bold] {output_path}\n")


def _format_duration(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _display_filler_bars(fillers: list[FillerSegment]) -> None:
    counts = Counter(f.word for f in fillers)
    if not counts:
        return

    console.print()
    max_count = max(counts.values())
    max_label_len = max(len(word) for word in counts)
    bar_max = 30

    for word, count in counts.most_common():
        filled = max(1, int(count / max_count * bar_max))
        empty = bar_max - filled
        bar = "\u2588" * filled + "\u2591" * empty
        label = f'[bold cyan]"{word}"[/bold cyan]'.ljust(max_label_len + 26)
        console.print(f"    {label}  {bar}  [bold]{count}[/bold] found")
    console.print()


def interactive_filter(fillers: list[FillerSegment]) -> list[FillerSegment]:
    kept: list[FillerSegment] = []
    for filler in fillers:
        console.print(
            f"\n[cyan]{filler.start:.2f}s - {filler.end:.2f}s[/cyan]: "
            f"[bold]{filler.word}[/bold] "
            f"(confidence: {filler.confidence:.2f}, source: {filler.source.value})"
        )
        response = console.input("[yellow]Remove? [Y/n]: [/yellow]").strip().lower()
        if response in ("", "y", "yes"):
            kept.append(filler)
    return kept


def _emit(
    reporter: ReporterLike | Callable[[PipelineEvent], None] | None,
    *,
    kind: PipelineEventKind,
    stage: PipelineStage | None,
    message: str,
    warning: str | None = None,
    stats: dict[str, int | float | str | bool] | None = None,
) -> None:
    if reporter is None:
        return

    event = PipelineEvent(
        kind=kind,
        stage=stage,
        message=message,
        warning=warning,
        stats=stats,
    )
    emit = getattr(reporter, "emit", None)
    if callable(emit):
        emit(event)
        return
    reporter(event)


def _check_cancel(cancel_token: CancellationToken | None, stage: PipelineStage) -> None:
    if cancel_token is None:
        return
    if cancel_token.is_cancelled():
        raise PipelineCancelledError(f"Cancelled during {stage.value}")


def _warning(
    warnings: list[str],
    reporter: ReporterLike | Callable[[PipelineEvent], None] | None,
    stage: PipelineStage,
    message: str,
) -> None:
    warnings.append(message)
    _emit(
        reporter,
        kind=PipelineEventKind.WARNING,
        stage=stage,
        message=message,
        warning=message,
    )


def _verification_score(result) -> tuple[int, int, int, int]:
    return (
        len(result.remaining_fillers) + len(result.new_fillers),
        len(result.damaged_words),
        len(result.lost_words),
        len(result.audio_discontinuities),
    )


def _smooth_audio_track(
    output_path: Path,
    segments: list[Segment],
    room_tone: np.ndarray,
    native_sr: int,
    metadata: VideoMetadata,
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> None:
    # Compatibility shim for older tests and callers. The primary path now assembles
    # output audio directly from the source track inside `_render_with_audio()`.
    rendered_audio, _ = extract_audio_pcm(output_path, sample_rate=native_sr)
    replace_audio_track(output_path, rendered_audio, native_sr, metadata)


def _build_pause_overrides(fillers: list[FillerSegment]) -> dict[int, float] | None:
    overrides: dict[int, float] = {}
    for filler_index, filler in enumerate(fillers):
        adaptive = compute_adaptive_pause(filler.end - filler.start)
        if adaptive > 0:
            overrides[filler_index] = adaptive
    return overrides or None


def _resolve_pause_overrides(
    fillers: list[FillerSegment],
    pause_ms: float | None,
) -> dict[int, float] | None:
    if pause_ms is None:
        return None
    pause_s = max(0.0, pause_ms / 1000.0)
    if pause_s == 0:
        return None
    return {index: pause_s for index in range(len(fillers))}


def _filler_margin_seconds(filler: FillerSegment) -> tuple[float, float]:
    start_ms = FILLER_MARGIN_START_MS
    end_ms = FILLER_MARGIN_END_MS
    if " " in filler.word:
        start_ms += PHRASE_MARGIN_START_BONUS_MS
        end_ms += PHRASE_MARGIN_END_BONUS_MS
    return start_ms / 1000.0, end_ms / 1000.0


def _classify_and_generate_transitions(
    segments: list[Segment],
    input_path: Path,
    metadata: VideoMetadata,
    interpolator: str,
    reporter: ReporterLike | Callable[[PipelineEvent], None] | None = None,
    warnings: list[str] | None = None,
    cancel_token: CancellationToken | None = None,
) -> tuple[list[Segment], dict[int, list[np.ndarray]]]:
    def get_frame(time_s: float):
        return get_frame_at_time(input_path, time_s, metadata.width, metadata.height)

    _check_cancel(cancel_token, PipelineStage.PLAN_CUTS)
    segments = classify_transitions(segments, get_frame, framerate=metadata.framerate)

    interpolated_frames_map: dict[int, list[np.ndarray]] = {}
    interp_count = sum(1 for segment in segments[1:] if segment.transition_type == TransitionType.INTERPOLATE)
    if interp_count == 0:
        return segments, interpolated_frames_map

    try:
        ensure_interpolator_backend(interpolator, console if isinstance(reporter, RichPipelineReporter) else None)
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError, ValueError):
        for segment in segments[1:]:
            if segment.transition_type == TransitionType.INTERPOLATE:
                segment.transition_type = TransitionType.HARD
        if warnings is not None:
            _warning(
                warnings,
                reporter,
                PipelineStage.PLAN_CUTS,
                "Frame interpolation backend was unavailable. Falling back to hard cuts.",
            )
        return segments, interpolated_frames_map

    for index in range(1, len(segments)):
        _check_cancel(cancel_token, PipelineStage.PLAN_CUTS)
        if segments[index].transition_type != TransitionType.INTERPOLATE:
            continue
        frame_a = get_frame(segments[index - 1].end)
        frame_b = get_frame(segments[index].start)
        try:
            frames = interpolate_frames(frame_a, frame_b, backend=interpolator)
            interpolated_frames_map[index] = frames
        except (FileNotFoundError, subprocess.CalledProcessError):
            segments[index].transition_type = TransitionType.HARD
            if warnings is not None:
                _warning(
                    warnings,
                    reporter,
                    PipelineStage.PLAN_CUTS,
                    "Frame interpolation backend was unavailable. Falling back to hard cuts.",
                )
    return segments, interpolated_frames_map


def _transition_durations(
    interpolated_frames_map: dict[int, list[np.ndarray]],
    framerate: float,
) -> dict[int, float]:
    return {
        index: len(frames) / framerate
        for index, frames in interpolated_frames_map.items()
        if frames
    }


def _write_seam_report(output_path: Path, seam_report) -> None:
    report_path = output_path.with_name(f"{output_path.stem}.ummfiltered-seams.json")
    payload = {
        "median_score": seam_report.median_score,
        "p95_score": seam_report.p95_score,
        "entries": [
            {
                "seam_index": entry.seam_index,
                "output_time": entry.output_time,
                "chosen_strategy": entry.chosen_strategy,
                "before_score": entry.before_score,
                "after_score": entry.after_score,
                "left_shift_ms": entry.left_shift_ms,
                "right_shift_ms": entry.right_shift_ms,
                "duration_ms": entry.duration_ms,
                "accepted": entry.accepted,
                "notes": entry.notes,
                "repair_strategy": entry.repair_strategy,
                "repair_text": entry.repair_text,
                "repair_accepted": entry.repair_accepted,
                "repair_notes": entry.repair_notes,
            }
            for entry in seam_report.entries
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2))


def _render_with_audio(
    input_path: Path,
    output_path: Path,
    segments: list[Segment],
    metadata: VideoMetadata,
    quality: str,
    interpolated_frames_map: dict[int, list[np.ndarray]],
    source_audio: np.ndarray,
    native_sr: int,
    room_tone: np.ndarray,
    preserved_words: list | None = None,
    contract_tokens: list[str] | None = None,
    crossfade_overrides: dict[int, float] | None = None,
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> None:
    edit_plan = build_edit_decision_list(
        segments,
        pause_overrides=pause_overrides,
        transition_durations=transition_durations,
        preserved_words=preserved_words,
        contract_tokens=contract_tokens,
    )
    render_video(
        input_path,
        output_path,
        segments,
        metadata,
        quality,
        interpolated_frames_map,
        pause_ms=0,
        crossfade_overrides=crossfade_overrides,
        pause_overrides=pause_overrides,
        edit_plan=edit_plan,
    )
    assembled_audio, seam_report = assemble_audio_track(
        source_audio,
        native_sr,
        edit_plan,
        room_tone,
    )
    replace_audio_track(output_path, assembled_audio, native_sr, metadata)
    _write_seam_report(output_path, seam_report)


def _result_meets_acceptance(result) -> bool:
    filler_count = len(result.remaining_fillers) + len(result.new_fillers)
    return (
        filler_count == 0
        and result.contract_intact
        and not result.lost_words
        and not result.damaged_words
        and not result.audio_discontinuities
    )


def _transcript_regressed(candidate, reference) -> bool:
    if reference is None:
        return False
    if reference.contract_intact and not candidate.contract_intact:
        return True
    if len(candidate.missing_tokens) > len(reference.missing_tokens):
        return True
    if candidate.preserved_word_recall + 1e-6 < reference.preserved_word_recall - 0.01:
        return True
    if candidate.max_missing_run > reference.max_missing_run + 1:
        return True
    if len(candidate.lost_words) > len(reference.lost_words):
        return True
    if len(candidate.damaged_words) > len(reference.damaged_words):
        return True
    return False


def _is_better_result(candidate, reference) -> bool:
    if reference is None:
        return True
    if _transcript_regressed(candidate, reference):
        return False

    candidate_fillers = len(candidate.remaining_fillers) + len(candidate.new_fillers)
    reference_fillers = len(reference.remaining_fillers) + len(reference.new_fillers)
    candidate_p95 = candidate.seam_report.p95_score if candidate.seam_report else float("inf")
    reference_p95 = reference.seam_report.p95_score if reference.seam_report else float("inf")

    if candidate_fillers < reference_fillers:
        return True
    if candidate_fillers > reference_fillers:
        return False
    if candidate.contract_intact and not reference.contract_intact:
        return True
    if len(candidate.missing_tokens) < len(reference.missing_tokens):
        return True
    if candidate.preserved_word_recall > reference.preserved_word_recall + 0.001:
        return True
    if candidate.max_missing_run < reference.max_missing_run:
        return True
    if candidate_p95 + SEAM_REGRESSION_MARGIN < reference_p95:
        return True
    return False


def _status_message(result) -> str:
    filler_count = len(result.remaining_fillers) + len(result.new_fillers)
    seam_p95 = result.seam_report.p95_score if result.seam_report else 0.0
    return (
        f"{filler_count} fillers, recall {result.preserved_word_recall:.3f}, "
        f"missing tokens {len(result.missing_tokens)}, "
        f"max missing run {result.max_missing_run}, seam p95 {seam_p95:.3f}"
    )


def _should_try_no_pause_variant(
    result,
    pause_ms: float | None,
    pause_overrides: dict[int, float] | None,
) -> bool:
    if pause_ms is not None:
        return False
    if not pause_overrides:
        return False
    if len(pause_overrides) < 5:
        return False
    filler_count = len(result.remaining_fillers) + len(result.new_fillers)
    if filler_count > 0:
        return True
    return len(result.lost_words) >= 3


def _result_stats(result: PipelineResult) -> dict[str, int | float | str | bool]:
    return {
        "outputPath": result.outputPath,
        "removedFillers": result.removedFillers,
        "removedSeconds": result.removedSeconds,
        "finalStatus": result.finalStatus.value,
        "warningCount": len(result.warnings),
    }


def run_pipeline(
    input_path: Path,
    output_path: Path,
    model_size: str = "large",
    cloud: str | None = None,
    aggressive: bool = False,
    interactive: bool = False,
    dry_run: bool = False,
    quality: str = "matched",
    custom_fillers: list[str] | None = None,
    interpolator: str = "ncnn",
    min_confidence: float = 0.15,
    pause_ms: float | None = None,
    no_refine: bool = False,
    reporter: ReporterLike | Callable[[PipelineEvent], None] | None = None,
    cancel_token: CancellationToken | None = None,
) -> PipelineResult:
    warnings: list[str] = []

    try:
        _check_cancel(cancel_token, PipelineStage.PROBE)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_STARTED,
            stage=PipelineStage.PROBE,
            message="Inspecting source video...",
        )
        metadata = probe_video(input_path)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_COMPLETED,
            stage=PipelineStage.PROBE,
            message="Inspecting source video...",
            stats={"duration": metadata.duration},
        )

        _check_cancel(cancel_token, PipelineStage.EXTRACT_AUDIO)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_STARTED,
            stage=PipelineStage.EXTRACT_AUDIO,
            message="Listening to audio...",
        )
        samples, sample_rate = extract_audio_pcm(input_path)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_COMPLETED,
            stage=PipelineStage.EXTRACT_AUDIO,
            message="Listening to audio...",
            stats={"sampleRate": sample_rate},
        )

        _check_cancel(cancel_token, PipelineStage.TRANSCRIBE)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_STARTED,
            stage=PipelineStage.TRANSCRIBE,
            message="Transcribing speech...",
        )
        words = transcribe(str(input_path), model_size=model_size, cloud=cloud)
        source_audio, native_sr = extract_audio_matrix(
            input_path,
            sample_rate=metadata.audio_sample_rate,
            channel_count=metadata.audio_channels,
        )
        room_tone = extract_room_tone(source_audio, native_sr, words=words)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_COMPLETED,
            stage=PipelineStage.TRANSCRIBE,
            message="Transcribing speech...",
            stats={"wordCount": len(words)},
        )

        _check_cancel(cancel_token, PipelineStage.DETECT_FILLERS)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_STARTED,
            stage=PipelineStage.DETECT_FILLERS,
            message="Hunting for filler words...",
        )
        fillers = detect_fillers(words, aggressive=aggressive, custom_fillers=custom_fillers)
        fillers = filter_fillers_by_context(fillers, words, min_confidence=min_confidence)
        fillers = expand_zero_duration_fillers(fillers, words)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_COMPLETED,
            stage=PipelineStage.DETECT_FILLERS,
            message="Hunting for filler words...",
            stats={"fillerCount": len(fillers)},
        )

        if isinstance(reporter, RichPipelineReporter):
            _display_filler_bars(fillers)

        if not fillers:
            result = PipelineResult(
                outputPath=str(output_path),
                removedFillers=0,
                removedSeconds=0.0,
                warnings=warnings,
                finalStatus=PipelineFinalStatus.NO_FILLERS,
            )
            _emit(
                reporter,
                kind=PipelineEventKind.RESULT,
                stage=PipelineStage.DETECT_FILLERS,
                message="No filler words detected.",
                stats=_result_stats(result),
            )
            return result

        if interactive:
            fillers = interactive_filter(fillers)

        if dry_run:
            result = PipelineResult(
                outputPath=str(output_path),
                removedFillers=len(fillers),
                removedSeconds=sum(f.end - f.start for f in fillers),
                warnings=warnings,
                finalStatus=PipelineFinalStatus.DRY_RUN,
            )
            _emit(
                reporter,
                kind=PipelineEventKind.RESULT,
                stage=PipelineStage.DETECT_FILLERS,
                message="Dry run complete.",
                stats=_result_stats(result),
            )
            return result

        _check_cancel(cancel_token, PipelineStage.PLAN_CUTS)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_STARTED,
            stage=PipelineStage.PLAN_CUTS,
            message="Planning cuts...",
        )
        filler_ranges = {(f.start, f.end) for f in fillers}
        non_filler_words = [
            word for word in words
            if not any(fs <= word.start and word.end <= fe for fs, fe in filler_ranges)
        ]
        contract_tokens = build_reference_contract(words, fillers)
        expanded_fillers: list[FillerSegment] = []
        for filler in fillers:
            _check_cancel(cancel_token, PipelineStage.PLAN_CUTS)
            margin_start_s, margin_end_s = _filler_margin_seconds(filler)
            padded_start = max(0.0, filler.start - margin_start_s)
            padded_end = min(metadata.duration, filler.end + margin_end_s)
            new_start, new_end = find_silence_boundaries(
                samples,
                sample_rate,
                padded_start,
                padded_end,
                threshold_db=SILENCE_THRESHOLD_DB,
                max_expansion_ms=MAX_EXPANSION_MS,
            )
            protected = protect_adjacent_words(
                new_start,
                new_end,
                non_filler_words,
                samples,
                sample_rate,
            )
            if protected is None:
                continue
            new_start, new_end = protected
            expanded_fillers.append(
                FillerSegment(
                    start=new_start,
                    end=new_end,
                    word=filler.word,
                    confidence=filler.confidence,
                    source=filler.source,
                )
            )

        segments = build_keep_segments(expanded_fillers, metadata.duration, words=words)
        effective_pause_ms = pause_ms
        pause_overrides = _resolve_pause_overrides(expanded_fillers, effective_pause_ms)
        segments, interpolated_frames_map = _classify_and_generate_transitions(
            segments,
            input_path,
            metadata,
            interpolator,
            reporter=reporter,
            warnings=warnings,
            cancel_token=cancel_token,
        )
        transition_durations = _transition_durations(interpolated_frames_map, metadata.framerate)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_COMPLETED,
            stage=PipelineStage.PLAN_CUTS,
            message="Planning cuts...",
            stats={
                "segmentCount": len(segments),
                "removedFillers": len(expanded_fillers),
            },
        )

        _check_cancel(cancel_token, PipelineStage.RENDER)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_STARTED,
            stage=PipelineStage.RENDER,
            message="Stitching it all together...",
        )
        _render_with_audio(
            input_path,
            output_path,
            segments,
            metadata,
            quality,
            interpolated_frames_map,
            source_audio,
            native_sr,
            room_tone,
            preserved_words=non_filler_words,
            contract_tokens=contract_tokens,
            pause_overrides=pause_overrides,
            transition_durations=transition_durations,
        )
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_COMPLETED,
            stage=PipelineStage.RENDER,
            message="Stitching it all together...",
        )

        final_fillers_for_summary = expanded_fillers
        if not no_refine:
            _emit(
                reporter,
                kind=PipelineEventKind.STAGE_STARTED,
                stage=PipelineStage.VERIFY,
                message="Verifying output...",
            )
            adjustments = {
                index: CutAdjustment(
                    filler=filler,
                    expansion_ms=MAX_EXPANSION_MS,
                    crossfade_ms=CROSSFADE_MS,
                )
                for index, filler in enumerate(expanded_fillers)
            }
            current_fillers = expanded_fillers
            current_segments = segments
            current_pause_overrides = pause_overrides
            current_effective_pause_ms = effective_pause_ms
            current_transition_durations = transition_durations
            best_fillers = current_fillers
            best_result = None
            best_output_path = Path(
                tempfile.NamedTemporaryFile(suffix=output_path.suffix, delete=False).name
            )
            best_report_path = output_path.with_name(f"{output_path.stem}.ummfiltered-seams.json")
            best_report_copy = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)

            try:
                max_passes = 8
                for pass_num in range(1, max_passes + 1):
                    _check_cancel(cancel_token, PipelineStage.VERIFY)
                    current_edit_plan = build_edit_decision_list(
                        current_segments,
                        pause_overrides=current_pause_overrides,
                        transition_durations=current_transition_durations,
                        preserved_words=non_filler_words,
                        contract_tokens=contract_tokens,
                    )
                    result = verify_output(
                        output_path,
                        words,
                        current_fillers,
                        current_segments,
                        model_size=model_size,
                        aggressive=aggressive,
                        min_confidence=min_confidence,
                        pause_overrides=current_pause_overrides,
                        transition_durations=current_transition_durations,
                        edit_plan=current_edit_plan,
                        reference_fillers=fillers,
                    )

                    if pass_num == 1 and _should_try_no_pause_variant(
                        result,
                        current_effective_pause_ms,
                        current_pause_overrides,
                    ):
                        alt_output_path = Path(
                            tempfile.NamedTemporaryFile(suffix=output_path.suffix, delete=False).name
                        )
                        alt_report_path = alt_output_path.with_name(
                            f"{alt_output_path.stem}.ummfiltered-seams.json"
                        )
                        try:
                            alt_pause_overrides = None
                            alt_edit_plan = build_edit_decision_list(
                                current_segments,
                                pause_overrides=alt_pause_overrides,
                                transition_durations=current_transition_durations,
                                preserved_words=non_filler_words,
                                contract_tokens=contract_tokens,
                            )
                            _render_with_audio(
                                input_path,
                                alt_output_path,
                                current_segments,
                                metadata,
                                quality,
                                interpolated_frames_map,
                                source_audio,
                                native_sr,
                                room_tone,
                                preserved_words=non_filler_words,
                                contract_tokens=contract_tokens,
                                crossfade_overrides=None,
                                pause_overrides=alt_pause_overrides,
                                transition_durations=current_transition_durations,
                            )
                            alt_result = verify_output(
                                alt_output_path,
                                words,
                                current_fillers,
                                current_segments,
                                model_size=model_size,
                                aggressive=aggressive,
                                min_confidence=min_confidence,
                                pause_overrides=alt_pause_overrides,
                                transition_durations=current_transition_durations,
                                edit_plan=alt_edit_plan,
                                reference_fillers=fillers,
                            )
                            if _is_better_result(alt_result, result):
                                shutil.copy2(alt_output_path, output_path)
                                if alt_report_path.exists():
                                    shutil.copy2(alt_report_path, best_report_path)
                                current_pause_overrides = alt_pause_overrides
                                current_effective_pause_ms = 0.0
                                current_edit_plan = alt_edit_plan
                                result = alt_result
                                _emit(
                                    reporter,
                                    kind=PipelineEventKind.INFO,
                                    stage=PipelineStage.VERIFY,
                                    message=(
                                        "Adaptive pauses were hurting this file. "
                                        "Retrying the current pass with no inserted pauses."
                                    ),
                                )
                        finally:
                            alt_output_path.unlink(missing_ok=True)
                            alt_report_path.unlink(missing_ok=True)

                    if not _result_meets_acceptance(result) and output_path.exists():
                        repair_backup_path = Path(
                            tempfile.NamedTemporaryFile(suffix=output_path.suffix, delete=False).name
                        )
                        repair_backup_report = Path(
                            tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
                        )
                        try:
                            shutil.copy2(output_path, repair_backup_path)
                            if best_report_path.exists():
                                shutil.copy2(best_report_path, repair_backup_report)
                            repair_decisions = repair_output_audio(
                                output_path=output_path,
                                source_audio=source_audio,
                                sample_rate=native_sr,
                                metadata=metadata,
                                edit_plan=current_edit_plan,
                                verification=result,
                            )
                            if any(decision.accepted for decision in repair_decisions):
                                repaired_result = verify_output(
                                    output_path,
                                    words,
                                    current_fillers,
                                    current_segments,
                                    model_size=model_size,
                                    aggressive=aggressive,
                                    min_confidence=min_confidence,
                                    pause_overrides=current_pause_overrides,
                                    transition_durations=current_transition_durations,
                                    edit_plan=current_edit_plan,
                                    reference_fillers=fillers,
                                )
                                if repaired_result.seam_report is not None:
                                    repaired_result.seam_report = merge_repair_decisions(
                                        repaired_result.seam_report,
                                        repair_decisions,
                                    )
                                    _write_seam_report(output_path, repaired_result.seam_report)
                                if _is_better_result(repaired_result, result):
                                    result = repaired_result
                                    _emit(
                                        reporter,
                                        kind=PipelineEventKind.INFO,
                                        stage=PipelineStage.VERIFY,
                                        message=(
                                            "Applied surgical seam repairs on the roughest cuts "
                                            "and kept the improved version."
                                        ),
                                    )
                                else:
                                    shutil.copy2(repair_backup_path, output_path)
                                    if repair_backup_report.exists():
                                        shutil.copy2(repair_backup_report, best_report_path)
                        finally:
                            repair_backup_path.unlink(missing_ok=True)
                            repair_backup_report.unlink(missing_ok=True)

                    if _is_better_result(result, best_result):
                        best_result = result
                        best_fillers = current_fillers
                        shutil.copy2(output_path, best_output_path)
                        if best_report_path.exists():
                            shutil.copy2(best_report_path, best_report_copy)
                    elif best_result is not None:
                        _emit(
                            reporter,
                            kind=PipelineEventKind.INFO,
                            stage=PipelineStage.VERIFY,
                            message="A later refinement pass regressed. Restoring the best earlier render.",
                        )
                        shutil.copy2(best_output_path, output_path)
                        if best_report_copy.exists():
                            shutil.copy2(best_report_copy, best_report_path)
                        final_fillers_for_summary = best_fillers
                        break

                    if _result_meets_acceptance(result):
                        _emit(
                            reporter,
                            kind=PipelineEventKind.INFO,
                            stage=PipelineStage.VERIFY,
                            message="Verification passed transcript and seam acceptance.",
                        )
                        final_fillers_for_summary = current_fillers
                        break

                    _emit(
                        reporter,
                        kind=PipelineEventKind.INFO,
                        stage=PipelineStage.VERIFY,
                        message=(
                            f"Refinement pass {pass_num} found issues and is rerendering "
                            f"({_status_message(result)})."
                        ),
                        stats={
                            "remainingFillers": len(result.remaining_fillers),
                            "newFillers": len(result.new_fillers),
                            "lostWords": len(result.lost_words),
                            "damagedWords": len(result.damaged_words),
                            "audioIssues": len(result.audio_discontinuities),
                            "preservedRecall": result.preserved_word_recall,
                            "maxMissingRun": result.max_missing_run,
                            "seamP95": (
                                result.seam_report.p95_score if result.seam_report else 0.0
                            ),
                        },
                    )

                    if pass_num == max_passes:
                        final_fillers_for_summary = best_fillers
                        if best_result is not None:
                            shutil.copy2(best_output_path, output_path)
                            if best_report_copy.exists():
                                shutil.copy2(best_report_copy, best_report_path)
                        break

                    apply_adjustments(
                        adjustments,
                        result,
                        segments=current_segments,
                        pause_overrides=current_pause_overrides,
                        transition_durations=current_transition_durations,
                    )
                    current_fillers, crossfade_map = rebuild_cuts(
                        adjustments,
                        samples,
                        sample_rate,
                        non_filler_words=non_filler_words,
                    )
                    current_segments = build_keep_segments(
                        current_fillers,
                        metadata.duration,
                        words=words,
                    )
                    current_pause_overrides = (
                        _resolve_pause_overrides(current_fillers, current_effective_pause_ms)
                    )
                    current_segments, interpolated_frames_map = _classify_and_generate_transitions(
                        current_segments,
                        input_path,
                        metadata,
                        interpolator,
                        reporter=reporter,
                        warnings=warnings,
                        cancel_token=cancel_token,
                    )
                    current_transition_durations = _transition_durations(
                        interpolated_frames_map,
                        metadata.framerate,
                    )

                    _check_cancel(cancel_token, PipelineStage.RENDER)
                    _render_with_audio(
                        input_path,
                        output_path,
                        current_segments,
                        metadata,
                        quality,
                        interpolated_frames_map,
                        source_audio,
                        native_sr,
                        room_tone,
                        preserved_words=non_filler_words,
                        contract_tokens=contract_tokens,
                        crossfade_overrides=crossfade_map,
                        pause_overrides=current_pause_overrides,
                        transition_durations=current_transition_durations,
                    )
                    final_fillers_for_summary = current_fillers
            finally:
                best_output_path.unlink(missing_ok=True)
                best_report_copy.unlink(missing_ok=True)

            _emit(
                reporter,
                kind=PipelineEventKind.STAGE_COMPLETED,
                stage=PipelineStage.VERIFY,
                message="Verifying output...",
            )
        else:
            _emit(
                reporter,
                kind=PipelineEventKind.STAGE_COMPLETED,
                stage=PipelineStage.VERIFY,
                message="Verification skipped.",
                stats={"skipped": True},
            )

        _check_cancel(cancel_token, PipelineStage.FINAL_CHECK)
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_STARTED,
            stage=PipelineStage.FINAL_CHECK,
            message="Running the final quality check...",
        )
        final_words = transcribe(str(output_path), model_size=model_size)
        final_detected_fillers = detect_fillers(final_words, aggressive=aggressive)
        final_detected_fillers = filter_fillers_by_context(
            final_detected_fillers,
            final_words,
            min_confidence=min_confidence,
        )
        output_samples, output_sr = extract_audio_pcm(output_path)
        confirmed_fillers: list[FillerSegment] = []
        for filler in final_detected_fillers:
            start = int(filler.start * output_sr)
            end = int(filler.end * output_sr)
            if start < len(output_samples) and end <= len(output_samples):
                energy = compute_rms_db(output_samples[start:end])
                if energy > -35:
                    confirmed_fillers.append(filler)
        if confirmed_fillers:
            _warning(
                warnings,
                reporter,
                PipelineStage.FINAL_CHECK,
                f"{len(confirmed_fillers)} fillers were still detected in the final output.",
            )
        _emit(
            reporter,
            kind=PipelineEventKind.STAGE_COMPLETED,
            stage=PipelineStage.FINAL_CHECK,
            message="Running the final quality check...",
            stats={"remainingFillers": len(confirmed_fillers)},
        )

        pipeline_result = PipelineResult(
            outputPath=str(output_path),
            removedFillers=len(final_fillers_for_summary),
            removedSeconds=sum(f.end - f.start for f in final_fillers_for_summary),
            warnings=warnings,
            finalStatus=PipelineFinalStatus.SUCCESS,
        )
        _emit(
            reporter,
            kind=PipelineEventKind.RESULT,
            stage=PipelineStage.FINAL_CHECK,
            message="Processing complete.",
            stats=_result_stats(pipeline_result),
        )
        return pipeline_result
    except PipelineCancelledError as exc:
        result = PipelineResult(
            outputPath=str(output_path),
            removedFillers=0,
            removedSeconds=0.0,
            warnings=warnings,
            finalStatus=PipelineFinalStatus.CANCELLED,
        )
        _emit(
            reporter,
            kind=PipelineEventKind.CANCELLED,
            stage=None,
            message=str(exc),
            stats=_result_stats(result),
        )
        return result
