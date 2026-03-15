<p align="center">
  <img src="assets/logo.png" alt="UmmFiltered" width="600">
</p>

Remove filler words from talking-head videos. Takes a video, detects "um", "uh", "you know", and other fillers, cuts them out, and produces a clean video that sounds like a single continuous take.

## How It Works

1. **Transcribes** the audio using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (runs locally, no API key needed)
2. **Detects** filler words by matching against a built-in dictionary
3. **Refines** cut boundaries by analyzing the audio waveform to find natural silence points
4. **Evaluates** each cut visually — if the speaker moved significantly, it flags the transition for frame interpolation
5. **Renders** the final video with per-segment audio fades and adaptive pauses proportional to each filler's duration
6. **Verifies** the output by re-transcribing it, checking for remaining fillers, lost words, and audio discontinuities — then re-renders with adjusted parameters if needed (up to 3 passes)

## Installation

**System requirements:**
- Python 3.10+
- ffmpeg and ffprobe

If they are not already installed, ummfiltered will provision bundled versions automatically on first run.

```bash
git clone https://github.com/agharib/ummfiltered.git
cd UmmFiltered
pip install -e .
```

For cloud transcription (optional):
```bash
pip install -e ".[cloud]"
```

## Quick Start

```bash
# Basic usage — output goes to my_video_ummfiltered.mp4
ummfiltered my_video.mp4

# Specify output path
ummfiltered my_video.mp4 -o clean_video.mp4

# Preview what would be removed without rendering
ummfiltered my_video.mp4 --dry-run

# Review each filler before removing
ummfiltered my_video.mp4 --interactive
```

If you haven't installed the package, you can also run it directly:
```bash
python -m ummfiltered.cli my_video.mp4
```

## Options

| Flag | Default | Description |
|---|---|---|
| `input` | — | Input video file (required) |
| `-o`, `--output` | `{name}_ummfiltered.{ext}` | Output file path |
| `--model-size` | `large` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large`. Larger = more accurate but slower. |
| `--cloud` | — | Use a cloud API instead of local Whisper. Options: `deepgram` |
| `--aggressive` | off | Also detect contextual fillers like "like", "so", "basically" (words that are only sometimes fillers) |
| `--interactive` | off | Step through each detected filler and confirm before removing |
| `--dry-run` | off | Show the detected fillers table without rendering the output |
| `--quality` | `matched` | `matched` preserves the original bitrate. `lossless` uses CRF 18. |
| `--fillers` | — | Custom comma-separated filler list, e.g. `--fillers "anyway,right,okay"` |
| `--min-confidence` | `0.15` | Minimum transcription confidence to trust a filler detection. Lower catches more but risks false positives. |
| `--pause-ms` | adaptive | Fixed micro-pause override in milliseconds. By default, inserts adaptive pauses proportional to each filler's duration. Set a value like `100` to force a uniform pause, or `0` to disable pauses entirely. |
| `--interpolator` | `ncnn` | Frame interpolation backend. Requires `rife-ncnn-vulkan` on PATH. Falls back to hard cuts if unavailable. |
| `--no-refine` | off | Skip the post-render verification loop. By default, ummfiltered re-transcribes the output to check for remaining fillers and audio issues, then re-renders with adjusted parameters (up to 3 passes). |

## Examples

```bash
# Faster processing with a smaller model (less accurate)
ummfiltered lecture.mp4 --model-size tiny

# Aggressive mode catches "like", "so", "basically", etc.
ummfiltered podcast.mp4 --aggressive

# Custom filler list for a specific speaker's habits
ummfiltered interview.mp4 --fillers "anyway,right,okay,yeah"

# Force a fixed 100ms pause at every cut instead of adaptive pauses
ummfiltered talk.mp4 --pause-ms 100

# Disable pauses entirely for tightest possible cuts
ummfiltered talk.mp4 --pause-ms 0

# Use Deepgram cloud API (requires DEEPGRAM_API_KEY env var)
ummfiltered my_video.mp4 --cloud deepgram
```

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```
