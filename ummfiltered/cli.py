import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from ummfiltered.pipeline import run_pipeline

console = Console()


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return

    console.print("  [yellow]ffmpeg not found — installing...[/yellow]")
    system = platform.system()

    if system == "Linux":
        subprocess.run(
            ["sudo", "apt-get", "update", "-qq"],
            capture_output=True,
        )
        result = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "-qq", "ffmpeg"],
            capture_output=True,
        )
    elif system == "Darwin":
        result = subprocess.run(
            ["brew", "install", "ffmpeg"],
            capture_output=True,
        )
    else:
        console.print("[red]Error:[/red] Please install ffmpeg manually: https://ffmpeg.org/download.html")
        sys.exit(1)

    if result.returncode != 0 or not shutil.which("ffmpeg"):
        console.print("[red]Error:[/red] Failed to install ffmpeg. Please install it manually: https://ffmpeg.org/download.html")
        sys.exit(1)

    console.print("  [green]ffmpeg installed.[/green]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ummfiltered",
        description="Remove filler words from talking-head videos",
    )
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output video file")
    parser.add_argument("--interactive", action="store_true", help="Review each filler before removing")
    parser.add_argument("--aggressive", action="store_true", help="Also remove contextual/low-confidence fillers")
    parser.add_argument("--cloud", choices=["deepgram", "openai"], default=None, help="Use cloud API for transcription")
    parser.add_argument("--model-size", choices=["tiny", "base", "small", "medium", "large"], default="medium", help="Whisper model size")
    parser.add_argument("--quality", choices=["lossless", "matched"], default="matched", help="Output quality mode")
    parser.add_argument("--dry-run", action="store_true", help="Show detected fillers without rendering")
    parser.add_argument("--fillers", type=str, default=None, help="Comma-separated custom filler word list")
    parser.add_argument("--interpolator", choices=["ncnn", "torch"], default="ncnn", help="Frame interpolation backend")
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
