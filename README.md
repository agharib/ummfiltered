<p align="center">
  <img src="assets/logo.png" alt="UmmFiltered" width="600">
</p>

Remove filler words from talking-head videos. UmmFiltered ships as both a CLI and a macOS-first desktop app. It runs locally, detects "um", "uh", "you know", and other fillers, then renders a cleaned video that still feels like one continuous take.

## What You Get

- `ummfiltered` CLI for scripted or terminal-first workflows
- `desktop/` Tauri + React app for drag-and-drop local cleanup on macOS
- Local transcription with `faster-whisper`, no API key required

## How It Works

1. Transcribes the audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
2. Detects fillers against a built-in dictionary plus any custom entries you provide
3. Refines cut points around natural silence
4. Flags visually risky cuts for interpolation when available
5. Renders the cleaned video with fades and adaptive pause smoothing
6. Verifies the output by re-transcribing and re-rendering if needed

## Requirements

### Core

- Python 3.10+
- `ffmpeg` and `ffprobe`

If `ffmpeg` is missing, UmmFiltered will attempt to provision bundled binaries on first run.

### Desktop App Development

The desktop UI is currently macOS-first and expects:

- macOS
- Xcode Command Line Tools
- Node.js and npm
- Rust and Cargo

## Fresh Install On macOS

From a brand-new machine, this is the safest path.

### 1. Clone the repo

```bash
git clone https://github.com/agharib/ummfiltered.git
cd ummfiltered
```

### 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you want the optional cloud transcription extras too:

```bash
python -m pip install -e ".[cloud,dev]"
```

If you want to build the installable desktop app bundle or DMG locally:

```bash
python -m pip install -e ".[dev,packaging]"
```

### 3. Install the macOS desktop toolchain

If these are already installed, you can skip this step.

```bash
xcode-select --install
brew install node rustup-init
rustup-init -y
source "$HOME/.cargo/env"
```

Verify the toolchain:

```bash
python3 --version
node --version
npm --version
rustc --version
cargo --version
```

## CLI Quick Start

```bash
# Basic usage: writes my_video_ummfiltered.mp4 next to the source file
ummfiltered my_video.mp4

# Specify output path
ummfiltered my_video.mp4 -o clean_video.mp4

# Preview what would be removed without rendering
ummfiltered my_video.mp4 --dry-run

# Run directly from the repo without relying on shell PATH
python -m ummfiltered.cli my_video.mp4
```

## Desktop UI Quick Start

Install the frontend dependencies once:

```bash
cd desktop
npm install
```

Then start the app:

```bash
npm run tauri dev
```

The app should open a single-window UI where you can:

1. Drop a video
2. Keep the default `Natural Cleanup` preset or change it
3. Click `Clean My Video`

### Point the UI at your virtualenv Python

The desktop app uses `python3` by default. If your dependencies live in a virtualenv, point Tauri at that interpreter explicitly:

```bash
cd desktop
UMMFILTERED_PYTHON="$(pwd)/../.venv/bin/python" npm run tauri dev
```

That is the most reliable option for a fresh local setup.

## Build A Local macOS App Or DMG

The desktop app can now be bundled with a frozen Python worker, so the packaged app does not require Python on the destination machine.

### Build the frozen desktop worker

```bash
cd desktop
npm install
npm run build:worker
```

That writes the packaged worker bundle into `desktop/resources/macos/worker/`.

### Build an unsigned local `.app` and `.dmg`

```bash
cd desktop
npm run bundle:macos
```

The Tauri outputs land under:

```bash
desktop/src-tauri/target/release/bundle/
```

### Build a signed and notarized public DMG

Set these environment variables first:

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
export APPLE_ID="you@example.com"
export APPLE_TEAM_ID="TEAMID"
export APPLE_APP_PASSWORD="app-specific-password"
```

Then run:

```bash
cd desktop
npm run release:macos
```

The final notarized DMG is written to:

```bash
desktop/release-artifacts/UmmFiltered.dmg
```

## Options

| Flag | Default | Description |
|---|---|---|
| `input` | — | Input video file |
| `-o`, `--output` | `{name}_ummfiltered.{ext}` | Output file path |
| `--model-size` | `large` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--cloud` | — | Use a cloud API instead of local Whisper. Options: `deepgram` |
| `--aggressive` | off | Catch contextual fillers like "like", "so", and "basically" |
| `--interactive` | off | Review each filler before removing it |
| `--dry-run` | off | Show detections without rendering |
| `--quality` | `matched` | `matched` preserves the original bitrate, `lossless` uses CRF 18 |
| `--fillers` | — | Custom comma-separated filler list |
| `--min-confidence` | `0.15` | Minimum transcription confidence for filler detection |
| `--pause-ms` | adaptive | Fixed pause override in milliseconds. `0` disables pauses |
| `--interpolator` | `ncnn` | Frame interpolation backend |
| `--no-refine` | off | Skip the verification loop |

## Examples

```bash
# Faster processing with a smaller model
ummfiltered lecture.mp4 --model-size tiny

# Aggressive mode catches more contextual fillers
ummfiltered podcast.mp4 --aggressive

# Custom filler list for a specific speaker
ummfiltered interview.mp4 --fillers "anyway,right,okay,yeah"

# Force a fixed 100ms pause at every cut
ummfiltered talk.mp4 --pause-ms 100

# Disable pauses entirely
ummfiltered talk.mp4 --pause-ms 0

# Use Deepgram (requires DEEPGRAM_API_KEY)
ummfiltered my_video.mp4 --cloud deepgram
```

## Verify A Fresh Setup

From the repo root:

```bash
python -m pytest -q
```

From the desktop app:

```bash
cd desktop
npm run test:run
npm run build
npm run build:worker
cd src-tauri
cargo check
```

If those commands pass, the Python pipeline, frozen worker build, and local desktop UI are wired correctly.

## Troubleshooting

### `pip` warns that its scripts are not on PATH

That warning is harmless. Keep using `python -m pip ...` and `python -m pytest ...`.

### `cargo` or `rustc` is still missing after install

Load Rust into the current shell:

```bash
source "$HOME/.cargo/env"
```

### The desktop app cannot find the Python worker

Use the virtualenv explicitly:

```bash
cd desktop
UMMFILTERED_PYTHON="$(pwd)/../.venv/bin/python" npm run tauri dev
```

### Tauri window config changes do not show up

Stop the dev app completely and start it again. Hot reload will not pick up `tauri.conf.json` changes.

### PyInstaller is missing when building the packaged worker

Install packaging dependencies into your active Python environment:

```bash
python -m pip install -e ".[packaging]"
```

### The packaged app launches but cannot write helper binaries

The desktop app now stores its runtime helper assets in its app data directory. If you previously had stale test files under `~/.ummfiltered`, remove them and try again.
