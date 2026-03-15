from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import tempfile

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ummfiltered.audio import compute_rms_db, extract_audio_pcm, extract_room_tone, find_silence_boundaries, find_speech_onset, protect_adjacent_words, smooth_rendered_audio
from ummfiltered.config import CROSSFADE_MS, FILLER_MARGIN_END_MS, FILLER_MARGIN_START_MS, MAX_EXPANSION_MS, PHRASE_MARGIN_END_BONUS_MS, PHRASE_MARGIN_START_BONUS_MS, SILENCE_THRESHOLD_DB, compute_adaptive_pause
from ummfiltered.cut_planner import build_keep_segments, classify_transitions
from ummfiltered.detect import detect_fillers, expand_zero_duration_fillers, filter_fillers_by_context
from ummfiltered.interpolate import interpolate_frames
from ummfiltered.models import CutAdjustment, FillerSegment, Segment, TransitionType
from ummfiltered.render import get_frame_at_time, probe_video, render_video, replace_audio_track
from ummfiltered.verify import apply_adjustments, rebuild_cuts, verify_output
from ummfiltered.transcribe import transcribe

console = Console()


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
    max_label_len = max(len(w) for w in counts)
    bar_max = 30

    for word, count in counts.most_common():
        filled = max(1, int(count / max_count * bar_max))
        empty = bar_max - filled
        bar = "\u2588" * filled + "\u2591" * empty
        label = f'[bold cyan]"{word}"[/bold cyan]'.ljust(max_label_len + 26)
        console.print(f"    {label}  {bar}  [bold]{count}[/bold] found")
    console.print()


def _spinner(description: str):
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )


def _verification_score(result) -> tuple[int, int, int, int]:
    return (
        len(result.remaining_fillers) + len(result.new_fillers),
        len(result.damaged_words),
        len(result.lost_words),
        len(result.audio_discontinuities),
    )


def interactive_filter(fillers: list[FillerSegment]) -> list[FillerSegment]:
    kept: list[FillerSegment] = []
    for f in fillers:
        console.print(f"\n[cyan]{f.start:.2f}s - {f.end:.2f}s[/cyan]: [bold]{f.word}[/bold] (confidence: {f.confidence:.2f}, source: {f.source.value})")
        response = console.input("[yellow]Remove? [Y/n]: [/yellow]").strip().lower()
        if response in ("", "y", "yes"):
            kept.append(f)
    return kept



def _compute_cut_points(
    segments: list[Segment],
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> list[float]:
    cut_points: list[float] = []
    output_offset = 0.0
    for i, seg in enumerate(segments):
        seg_dur = seg.end - seg.start
        if seg_dur < 0.001:
            continue
        output_offset += seg_dur
        pause = pause_overrides.get(i, 0.0) if pause_overrides else 0.0
        output_offset += pause
        if i < len(segments) - 1:
            cut_points.append(output_offset)
            transition_duration = transition_durations.get(i + 1, 0.0) if transition_durations else 0.0
            if transition_duration > 0:
                output_offset += transition_duration
                cut_points.append(output_offset)
    return cut_points


def _smooth_audio_track(
    output_path: Path,
    segments: list[Segment],
    room_tone: np.ndarray,
    native_sr: int,
    metadata: "VideoMetadata",
    pause_overrides: dict[int, float] | None = None,
    transition_durations: dict[int, float] | None = None,
) -> None:
    rendered_audio, _ = extract_audio_pcm(output_path, sample_rate=native_sr)
    cut_points = _compute_cut_points(segments, pause_overrides, transition_durations)
    smoothed = smooth_rendered_audio(rendered_audio, native_sr, cut_points, room_tone)
    replace_audio_track(output_path, smoothed, native_sr, metadata)


def _build_pause_overrides(fillers: list[FillerSegment]) -> dict[int, float] | None:
    overrides: dict[int, float] = {}
    for filler_idx, f in enumerate(fillers):
        adaptive = compute_adaptive_pause(f.end - f.start)
        if adaptive > 0:
            overrides[filler_idx] = adaptive
    return overrides or None


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
    metadata: "VideoMetadata",
    interpolator: str,
) -> tuple[list[Segment], dict[int, list[np.ndarray]]]:
    def get_frame(t: float):
        return get_frame_at_time(input_path, t, metadata.width, metadata.height)

    with _spinner("") as progress:
        progress.add_task("  👀  [bold]Smoothing cadence...[/bold]", total=None)
        segments = classify_transitions(segments, get_frame, framerate=metadata.framerate)
    console.print("  👀  [dim]Smoothing cadence...[/dim] [green]done[/green]")

    interpolated_frames_map: dict[int, list[np.ndarray]] = {}
    interp_count = sum(1 for s in segments[1:] if s.transition_type == TransitionType.INTERPOLATE)
    if interp_count == 0:
        return segments, interpolated_frames_map

    with _spinner("") as progress:
        progress.add_task("  🎞️  [bold]Generating transition frames...[/bold]", total=None)
        for i in range(1, len(segments)):
            if segments[i].transition_type != TransitionType.INTERPOLATE:
                continue

            frame_a = get_frame(segments[i - 1].end)
            frame_b = get_frame(segments[i].start)
            try:
                frames = interpolate_frames(frame_a, frame_b, backend=interpolator)
                interpolated_frames_map[i] = frames
            except FileNotFoundError:
                segments[i].transition_type = TransitionType.HARD
    console.print("  🎞️  [dim]Generating transition frames...[/dim] [green]done[/green]")

    return segments, interpolated_frames_map


def _transition_durations(
    interpolated_frames_map: dict[int, list[np.ndarray]],
    framerate: float,
) -> dict[int, float]:
    return {
        idx: len(frames) / framerate
        for idx, frames in interpolated_frames_map.items()
        if frames
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
) -> None:
    metadata = probe_video(input_path)

    with _spinner("") as progress:
        progress.add_task("  🎙️  [bold]Listening to audio...[/bold]", total=None)
        samples, sample_rate = extract_audio_pcm(input_path)
    console.print("  🎙️  [dim]Listening to audio...[/dim] [green]done[/green]")

    with _spinner("") as progress:
        progress.add_task("  🧠  [bold]Transcribing speech...[/bold]", total=None)
        words = transcribe(str(input_path), model_size=model_size, cloud=cloud)
        native_samples, native_sr = extract_audio_pcm(input_path, sample_rate=metadata.audio_sample_rate)
        room_tone = extract_room_tone(native_samples, native_sr, words=words)
    console.print("  🧠  [dim]Transcribing speech...[/dim] [green]done[/green]")

    with _spinner("") as progress:
        progress.add_task("  🔍  [bold]Hunting for filler words...[/bold]", total=None)
        fillers = detect_fillers(words, aggressive=aggressive, custom_fillers=custom_fillers)
        fillers = filter_fillers_by_context(fillers, words, min_confidence=min_confidence)
        fillers = expand_zero_duration_fillers(fillers, words)
    console.print("  🔍  [dim]Hunting for filler words...[/dim] [green]done[/green]")

    _display_filler_bars(fillers)

    if not fillers:
        console.print("  [green bold]✨ No filler words detected! Already clean.[/green bold]\n")
        return

    if interactive:
        fillers = interactive_filter(fillers)

    if dry_run:
        console.print(f"  [yellow]Dry run:[/yellow] would remove [bold]{len(fillers)}[/bold] fillers")
        return

    with _spinner("") as progress:
        progress.add_task("  ✂️  [bold]De-umming...[/bold]", total=None)
        filler_ranges = {(f.start, f.end) for f in fillers}
        non_filler_words = [
            w for w in words
            if not any(fs <= w.start and w.end <= fe for fs, fe in filler_ranges)
        ]
        expanded_fillers: list[FillerSegment] = []
        for f in fillers:
            margin_start_s, margin_end_s = _filler_margin_seconds(f)
            padded_start = max(0.0, f.start - margin_start_s)
            padded_end = min(metadata.duration, f.end + margin_end_s)
            new_start, new_end = find_silence_boundaries(
                samples, sample_rate, padded_start, padded_end,
                threshold_db=SILENCE_THRESHOLD_DB,
                max_expansion_ms=MAX_EXPANSION_MS,
            )
            protected = protect_adjacent_words(
                new_start, new_end, non_filler_words, samples, sample_rate,
            )
            if protected is None:
                continue
            new_start, new_end = protected
            expanded_fillers.append(FillerSegment(
                start=new_start, end=new_end,
                word=f.word, confidence=f.confidence, source=f.source,
            ))
        segments = build_keep_segments(expanded_fillers, metadata.duration, words=words)
        pause_overrides = _build_pause_overrides(expanded_fillers) if pause_ms is None else None
    console.print("  ✂️  [dim]De-umming...[/dim] [green]done[/green]")

    segments, interpolated_frames_map = _classify_and_generate_transitions(
        segments, input_path, metadata, interpolator,
    )
    transition_durations = _transition_durations(interpolated_frames_map, metadata.framerate)

    with _spinner("") as progress:
        progress.add_task("  🧵  [bold]Stitching it all together...[/bold]", total=None)
        render_video(
            input_path,
            output_path,
            segments,
            metadata,
            quality,
            interpolated_frames_map,
            pause_ms=pause_ms or 0,
            pause_overrides=pause_overrides,
        )
        _smooth_audio_track(
            output_path,
            segments,
            room_tone,
            native_sr,
            metadata,
            pause_overrides,
            transition_durations,
        )
    console.print("  🧵  [dim]Stitching it all together...[/dim] [green]done[/green]")

    final_fillers_for_summary = expanded_fillers
    if not no_refine and not dry_run:
        adjustments = {
            i: CutAdjustment(filler=f, expansion_ms=MAX_EXPANSION_MS, crossfade_ms=CROSSFADE_MS)
            for i, f in enumerate(expanded_fillers)
        }
        current_fillers = expanded_fillers
        current_segments = segments
        current_pause_overrides = pause_overrides
        current_transition_durations = transition_durations
        best_fillers = current_fillers
        best_score: tuple[int, int, int, int] | None = None
        best_output_path = Path(tempfile.NamedTemporaryFile(suffix=output_path.suffix, delete=False).name)

        try:
            max_passes = 8
            for pass_num in range(1, max_passes + 1):
                with _spinner("") as progress:
                    progress.add_task("  🔄  [bold]Verifying output...[/bold]", total=None)
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
                    )

                score = _verification_score(result)
                plateau_without_filler_issues = (
                    best_score is not None
                    and score == best_score
                    and not result.remaining_fillers
                    and not result.new_fillers
                    and not result.lost_words
                    and not result.damaged_words
                )
                if plateau_without_filler_issues:
                    console.print("  🔄  [dim]Verifying output...[/dim] [green]stable plateau reached — keeping current cleanest render[/green]")
                    final_fillers_for_summary = current_fillers
                    break

                if best_score is None or score < best_score:
                    best_score = score
                    best_fillers = current_fillers
                    shutil.copy2(output_path, best_output_path)
                elif score > best_score:
                    console.print("  🔄  [dim]Verifying output...[/dim] [yellow]later pass got worse — keeping best earlier render[/yellow]")
                    shutil.copy2(best_output_path, output_path)
                    final_fillers_for_summary = best_fillers
                    break

                if result.is_clean():
                    console.print("  🔄  [dim]Verifying output...[/dim] [green]✅ clean[/green]")
                    final_fillers_for_summary = current_fillers
                    break

                console.print("  🔄  [dim]Verifying output...[/dim] [yellow]issues found[/yellow]")
                if result.remaining_fillers:
                    console.print(f"     [dim]{len(result.remaining_fillers)} fillers survived — widening cuts[/dim]")
                if result.new_fillers:
                    console.print(f"     [dim]{len(result.new_fillers)} new fillers found — adding cuts[/dim]")
                if result.lost_words:
                    console.print(f"     [dim]{len(result.lost_words)} words looked unsafe — narrowing cuts[/dim]")
                if result.damaged_words:
                    console.print(f"     [dim]{len(result.damaged_words)} clipped words — narrowing cuts[/dim]")
                if result.audio_discontinuities:
                    console.print(f"     [dim]{len(result.audio_discontinuities)} audio jumps — increasing fades[/dim]")

                if pass_num == max_passes:
                    final_fillers_for_summary = best_fillers
                    shutil.copy2(best_output_path, output_path)
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
                current_segments = build_keep_segments(current_fillers, metadata.duration, words=words)
                current_pause_overrides = _build_pause_overrides(current_fillers) if pause_ms is None else None
                current_segments, interpolated_frames_map = _classify_and_generate_transitions(
                    current_segments,
                    input_path,
                    metadata,
                    interpolator,
                )
                current_transition_durations = _transition_durations(interpolated_frames_map, metadata.framerate)

                with _spinner("") as progress:
                    progress.add_task("  🧵  [bold]Re-stitching from original...[/bold]", total=None)
                    render_video(
                        input_path,
                        output_path,
                        current_segments,
                        metadata,
                        quality,
                        interpolated_frames_map,
                        pause_ms=pause_ms or 0,
                        crossfade_overrides=crossfade_map,
                        pause_overrides=current_pause_overrides,
                    )
                    _smooth_audio_track(
                        output_path,
                        current_segments,
                        room_tone,
                        native_sr,
                        metadata,
                        current_pause_overrides,
                        current_transition_durations,
                    )
                console.print("  🧵  [dim]Re-stitching from original...[/dim] [green]done[/green]")
                final_fillers_for_summary = current_fillers
        finally:
            best_output_path.unlink(missing_ok=True)

    if not dry_run:
        with _spinner("") as progress:
            progress.add_task("  🔍  [bold]Final quality check...[/bold]", total=None)
            final_words = transcribe(str(output_path), model_size=model_size)
            final_fillers = detect_fillers(final_words, aggressive=aggressive)
            final_fillers = filter_fillers_by_context(final_fillers, final_words, min_confidence=min_confidence)
            output_samples, output_sr = extract_audio_pcm(output_path)
            real_fillers = []
            for ff in final_fillers:
                s = int(ff.start * output_sr)
                e = int(ff.end * output_sr)
                if s < len(output_samples) and e <= len(output_samples):
                    energy = compute_rms_db(output_samples[s:e])
                    if energy > -35:
                        real_fillers.append((ff, energy))
            final_fillers = [ff for ff, _ in real_fillers]
        if final_fillers:
            console.print(f"  🔍  [dim]Final quality check...[/dim] [yellow]{len(final_fillers)} fillers still detected[/yellow]")
            for ff in final_fillers:
                console.print(f"     [dim]\"{ff.word}\" at {ff.start:.1f}s (conf={ff.confidence:.2f})[/dim]")
        else:
            console.print("  🔍  [dim]Final quality check...[/dim] [green]✅ no fillers detected[/green]")

    total_removed = sum(f.end - f.start for f in final_fillers_for_summary)
    console.print()
    console.print(f"  [green bold]✨ {len(fillers)} filler words removed[/green bold]")
    console.print(f"  [green bold]⏱️  {_format_duration(total_removed)} of dead air eliminated[/green bold]")
    console.print(f"  [green bold]🔊 Audio gaps closed naturally[/green bold]")
    console.print(f"\n  [bold]Saved to[/bold] {output_path}\n")
