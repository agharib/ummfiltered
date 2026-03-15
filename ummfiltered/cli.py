import argparse
import sys
from pathlib import Path

from rich.console import Console

from ummfiltered.ffmpeg_tools import ensure_ffmpeg_tools
from ummfiltered.pipeline import run_pipeline

console = Console()


def _ensure_ffmpeg() -> None:
    try:
        ensure_ffmpeg_tools(console)
    except Exception as exc:
        console.print(f"[red]Error:[/red] Failed to provision ffmpeg/ffprobe: {exc}")
        console.print("Install them manually or check your Python package install permissions.")
        sys.exit(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ummfiltered",
        description="Remove filler words from talking-head videos",
    )
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output video file")
    parser.add_argument("--interactive", action="store_true", help="Review each filler before removing")
    parser.add_argument("--aggressive", action="store_true", help="Also remove contextual/low-confidence fillers")
    parser.add_argument("--cloud", choices=["deepgram"], default=None, help="Use cloud API for transcription")
    parser.add_argument("--model-size", choices=["tiny", "base", "small", "medium", "large"], default="large", help="Whisper model size")
    parser.add_argument("--quality", choices=["lossless", "matched"], default="matched", help="Output quality mode")
    parser.add_argument("--dry-run", action="store_true", help="Show detected fillers without rendering")
    parser.add_argument("--fillers", type=str, default=None, help="Comma-separated custom filler word list")
    parser.add_argument("--interpolator", choices=["ncnn"], default="ncnn", help="Frame interpolation backend")
    parser.add_argument("--min-confidence", type=float, default=0.15, help="Minimum transcription confidence to consider a filler (filters hallucinations)")
    parser.add_argument("--pause-ms", type=float, default=None, help="Fixed micro-pause at each cut in ms. Omit for adaptive pauses (proportional to filler duration). Use 0 to disable pauses entirely.")
    parser.add_argument("--no-refine", action="store_true", help="Skip output verification and refinement passes")
    args = parser.parse_args(argv)

    if args.output is None:
        stem = args.input.stem
        args.output = args.input.with_name(f"{stem}_ummfiltered{args.input.suffix}")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _ensure_ffmpeg()

    if not args.input.exists():
        console.print(f"[red]Error:[/red] Input file not found: {args.input}")
        sys.exit(1)

    console.print(f"[bold]ummfiltered[/bold] — processing {args.input.name}\n")

    custom_fillers = args.fillers.split(",") if args.fillers else None

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        model_size=args.model_size,
        cloud=args.cloud,
        aggressive=args.aggressive,
        interactive=args.interactive,
        dry_run=args.dry_run,
        quality=args.quality,
        custom_fillers=custom_fillers,
        interpolator=args.interpolator,
        min_confidence=args.min_confidence,
        pause_ms=args.pause_ms,
        no_refine=args.no_refine,
    )


if __name__ == "__main__":
    main()
