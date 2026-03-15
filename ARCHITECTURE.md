# ummfiltered — Complete Technical Reference

This document describes every module, algorithm, data structure, constant, and design decision in the ummfiltered codebase. It is written for an LLM or developer who needs full context to read, modify, or extend this project.

## What ummfiltered Does

ummfiltered is a Python CLI tool that takes a talking-head video as input, detects filler words ("um", "uh", "you know", etc.), removes them, and produces a clean output video that looks like a single continuous take. The tool uses speech-to-text transcription to find fillers, audio analysis to determine precise cut boundaries, computer vision to evaluate cut quality, and ffmpeg to render the final output.

## Project Structure

```
ummfiltered/
  ummfiltered/
    __init__.py
    cli.py           # Argument parsing and entry point
    pipeline.py       # Orchestrates the full processing pipeline
    config.py         # All tunable constants in one place
    models.py         # Data classes: Word, FillerSegment, Segment, VideoMetadata, CutAdjustment, VerificationResult
    transcribe.py     # Speech-to-text via faster-whisper or Deepgram
    detect.py         # Filler word detection (dictionary + contextual)
    audio.py          # PCM extraction, RMS dB, silence boundary finding
    cut_planner.py    # Builds keep-segments from fillers, classifies transitions
    render.py         # ffmpeg rendering: probe, extract, concat
    interpolate.py    # Frame interpolation via rife-ncnn-vulkan (optional)
    verify.py         # Post-render verification and iterative refinement
  tests/
    test_audio.py
    test_cut_planner.py
    test_detect.py
    test_interpolate.py
    test_models.py
    test_pipeline_e2e.py
    test_render.py
    test_transcribe.py
    test_verify.py
  pyproject.toml
```

## Pipeline Flow (pipeline.py)

`run_pipeline()` is the main orchestrator. It executes these stages in order:

### Stage 1: Probe Video
Calls `probe_video(input_path)` to extract `VideoMetadata` (resolution, framerate, bitrate, pixel format, audio codec/sample rate/channels/bitrate, duration) using `ffprobe`.

### Stage 2: Extract Audio
Calls `extract_audio_pcm(input_path)` to produce a 16kHz mono float32 numpy array. This PCM data is used later for silence boundary detection.

### Stage 3: Transcribe
Calls `transcribe(input_path, model_size, cloud)` to get a list of `Word` objects (text, start, end, probability). Default is local faster-whisper with the "medium" model. Cloud option: Deepgram.

### Stage 4: Detect Fillers
Three-step detection:
1. `detect_fillers(words, aggressive, custom_fillers)` — dictionary matching, optionally plus contextual detection.
2. `filter_fillers_by_context(fillers, words, min_confidence)` — removes likely Whisper hallucinations.
3. `expand_zero_duration_fillers(fillers, words)` — handles Whisper artifacts where start==end.

### Stage 5: Silence Boundary Expansion
For each detected filler, `find_silence_boundaries()` walks outward from the filler's edges in 20ms windows, expanding the cut region into surrounding silence (up to 300ms). This ensures the cut happens during a silent moment rather than mid-phoneme.

### Stage 6: Build Keep Segments
`build_keep_segments(expanded_fillers, duration, words)` inverts the filler list into a list of `Segment` objects representing the portions of video to keep. Adjacent fillers with gaps shorter than `MIN_PAUSE_GAP_MS` (300ms) are merged unless the gap contains content words.

### Stage 7: Classify Transitions
`classify_transitions(segments, get_frame, framerate)` compares frames at each cut boundary using SSIM. If the score is >= `SSIM_THRESHOLD` (0.85), the cut is classified as `HARD` (simple concat). Otherwise, it tries shifting the cut point by up to 5 frames in each direction to find a better match. If no match exceeds the threshold, the cut is classified as `INTERPOLATE`.

### Stage 8: Frame Interpolation (Optional)
For `INTERPOLATE` transitions, if `rife-ncnn-vulkan` is available, it generates 3 intermediate frames between the two boundary frames. If the binary is missing, the transition falls back to `HARD`.

### Stage 9: Render
`render_video()` produces the final output. For multi-segment videos (the common case), it:
1. Adds 50ms padding around each cut via `add_padding()`.
2. Extracts each segment as a `.ts` file with per-segment audio fades (40ms in/out).
3. Concatenates all `.ts` files using ffmpeg's concat demuxer.
4. Re-encodes with quality matched to the source (same bitrate, framerate, pixel format, audio settings).

### Stage 10: Verify and Refine (default ON, skip with `--no-refine`)
After the initial render, `verify_output()` re-transcribes the output video and runs three checks:

1. **Remaining filler detection**: Re-runs filler detection on the output. Fillers are classified as "remaining" (matched to a known filler from pass 1 by position) or "new" (found only in the output). Known fillers that survived get their expansion widened by 150ms. New fillers are added to the cut list at normal confidence thresholds — this protects words like "I'm" from being misidentified.

2. **Word integrity check**: Compares expected words (original transcript minus intentionally cut fillers) against actual output words using LCS alignment with fuzzy matching. Only flags words near cut boundaries as "lost". Detects truncated words at cut points (e.g. "orld" for "world") as "damaged". Damaged words trigger narrowing the responsible cut by 100ms.

3. **Audio smoothness check**: At each cut point, compares RMS dB in 50ms windows before and after the cut. Flags volume jumps > 6dB and DC offset jumps > 0.05. Discontinuities trigger increasing crossfade for that specific cut by 40ms.

If any issues are found, `apply_adjustments()` modifies per-filler `CutAdjustment` parameters and `rebuild_cuts()` re-expands boundaries. The video is re-rendered from the **original source** (never re-encoding output) with the adjusted parameters. This repeats for up to 3 total passes.

---

## Module Details

### models.py

Six dataclasses and two enums:

- **`Word`**: `text: str`, `start: float`, `end: float`, `probability: float`. Output of transcription. `text` is always lowercase with punctuation stripped. `probability` is the transcriber's confidence (0.0-1.0).

- **`FillerSegment`**: `start: float`, `end: float`, `word: str`, `confidence: float`, `source: DetectionSource`. Represents a detected filler. `word` can be multi-word (e.g. "you know"). `source` is either `DICTIONARY` or `CONTEXTUAL`.

- **`Segment`**: `start: float`, `end: float`, `transition_type: TransitionType`, `visual_gap_score: float`. Represents a keep-region of the video. `visual_gap_score` is the SSIM score at the cut boundary preceding this segment.

- **`VideoMetadata`**: All ffprobe-derived properties of the input video needed for encoding the output.

- **`TransitionType`**: Enum with `HARD` and `INTERPOLATE`.

- **`DetectionSource`**: Enum with `DICTIONARY` and `CONTEXTUAL`.

- **`CutAdjustment`**: `filler: FillerSegment`, `expansion_ms: float`, `crossfade_ms: float`, `skip: bool`. Per-filler adjustment parameters used by the refinement loop. `expansion_ms` starts at `MAX_EXPANSION_MS` (300) and grows by 150 per pass. `crossfade_ms` starts at `CROSSFADE_MS` (40) and grows by 40 per pass. `skip` is set to True if cutting a filler damages adjacent words.

- **`VerificationResult`**: `remaining_fillers`, `new_fillers`, `lost_words`, `damaged_words`, `audio_discontinuities`. Output of the verification pass. `is_clean()` returns True when all lists are empty.

### config.py

All tunable constants:

| Constant | Value | Purpose |
|---|---|---|
| `DEFAULT_FILLERS` | `["um", "uh", "er", "ah", "hm", "hmm", "mm", "mhm"]` | Single-word fillers for dictionary detection |
| `DEFAULT_FILLER_PHRASES` | `["you know", "i mean", "kind of", "sort of"]` | Multi-word filler phrases |
| `CONTEXTUAL_FILLERS` | `["like", "so", "basically", "right", "actually", "literally"]` | Words that are only fillers in certain contexts (aggressive mode) |
| `SILENCE_THRESHOLD_DB` | `-40` | RMS dB below which audio is considered silence |
| `MAX_EXPANSION_MS` | `300` | Maximum ms to expand a filler's cut region into surrounding silence |
| `MIN_PAUSE_GAP_MS` | `300` | Minimum gap between fillers to keep as a separate segment |
| `MAX_FILLER_DURATION_MS` | `500` | Maximum duration for contextual filler detection |
| `SSIM_THRESHOLD` | `0.85` | SSIM score above which a hard cut is acceptable |
| `FRAME_SEARCH_WINDOW` | `5` | Frames to search in each direction for better cut alignment |
| `CROSSFADE_MS` | `40` | Duration of per-segment audio fade in/out |
| `PADDING_MS` | `50` | Padding added around cut boundaries |
| `MIN_CONFIDENCE` | `0.15` | Default minimum transcription confidence for filler filtering |
| `INTERPOLATION_FRAMES` | `3` | Number of intermediate frames to generate for interpolated transitions |
| `WHISPER_FILLER_PROMPT` | `"Um, uh, so like, you know..."` | Initial prompt fed to Whisper to bias it toward detecting fillers |

### transcribe.py

**`_clean_word(text)`**: Strips punctuation, lowercases. This is critical because Whisper sometimes returns "Um," or "Uh." and the dictionary matcher needs clean text.

**`build_whisper_params(model_size)`**: Returns a dict with:
- `initial_prompt`: The filler prompt that biases Whisper toward transcribing fillers rather than suppressing them.
- `suppress_tokens`: Empty list (don't suppress any tokens — the default suppresses "um"/"uh").
- `suppress_blank`: False.
- `vad_filter`: True (uses Silero VAD to skip silence, improving speed and accuracy).
- `word_timestamps`: True (needed for per-word timing).

**`transcribe_local(audio_path, model_size)`**: Uses faster-whisper. Iterates over segments and their words, cleaning each word and building `Word` objects.

**`transcribe_cloud_deepgram(audio_path)`**: Uses Deepgram's nova-2 model with `filler_words=True` and `punctuate=False`.

**`transcribe(audio_path, model_size, cloud)`**: Dispatcher. Routes to local or Deepgram based on `cloud` parameter.

### detect.py

**`detect_fillers_dictionary(words, custom_fillers)`**: Scans the word list for matches against `DEFAULT_FILLERS` (single words) and `DEFAULT_FILLER_PHRASES` (multi-word). Multi-word phrases are matched by checking consecutive words. When a phrase matches, subsequent words in the phrase are skipped. If `custom_fillers` is provided, it replaces the default single-word list and disables phrase matching.

**`detect_fillers_contextual(words)`**: Flags words from `CONTEXTUAL_FILLERS` that have low confidence (< 0.5), short duration (< 500ms), and are surrounded by pauses (gap > 150ms on at least one side). These are words like "like", "so", "basically" that are fillers only in certain speech patterns.

**`detect_fillers(words, aggressive, custom_fillers)`**: Combines dictionary and contextual detection. In non-aggressive mode, only dictionary fillers are returned. In aggressive mode, contextual fillers are added (deduplicated by time range).

**`filter_fillers_by_context(fillers, words, min_confidence)`**: Post-processing filter that removes likely Whisper hallucinations. For fillers with confidence < `min_confidence` (default 0.15):
- If the filler is a single word from `DEFAULT_FILLERS` AND is surrounded by pauses (gap > 150ms), it is kept (likely a real filler that Whisper was uncertain about).
- Otherwise, it is removed (likely a hallucinated word).
- Fillers with confidence >= `min_confidence` are always kept.

**`expand_zero_duration_fillers(fillers, words)`**: Handles a Whisper artifact where some fillers are reported with `start == end` (zero duration). These are real fillers where Whisper detected the word but couldn't determine its boundaries. The function expands the region to fill the gap between the previous and next word, capped at 0.5s in each direction. If the expanded region is still too short (< 50ms), the filler is dropped.

### audio.py

**`compute_rms_db(samples)`**: Computes RMS in decibels for a numpy audio array. Returns -100.0 for pure silence (avoids log(0)).

**`extract_audio_pcm(video_path, sample_rate=16000)`**: Extracts audio from video to a temporary .wav file using ffmpeg, then reads it as float32 numpy array normalized to [-1, 1]. Returns (samples, sample_rate).

**`find_silence_boundaries(samples, sample_rate, filler_start, filler_end, ...)`**: Starting from the filler's edges, walks outward in 20ms windows. For each window, if RMS is below `threshold_db` (-40 dB), the boundary expands. Stops when it hits non-silent audio or reaches `max_expansion_ms` (300ms). This ensures cuts happen at silence boundaries rather than clipping speech.

**`extract_room_tone(samples, sample_rate, ...)`**: Finds the quietest segment of audio in the file. Currently defined but not used in the pipeline (was planned for room tone filling at cut points).

### cut_planner.py

**`compute_frame_similarity(frame_a, frame_b)`**: SSIM comparison between two frames. Handles both grayscale and RGB (uses `channel_axis=2` for RGB).

**`_gap_has_content_words(gap_start, gap_end, words)`**: Returns True if any word falls entirely within the given time range. Used to prevent merging segments when real speech exists in the gap.

**`build_keep_segments(fillers, video_duration, words)`**: The core segment builder. Inverts the filler list: everything between fillers becomes a keep-segment. Adjacent fillers with gaps < 300ms are merged (the gap is cut) UNLESS the gap contains content words. Returns a list of `Segment` objects spanning the entire video minus the fillers.

**`classify_transitions(segments, get_frame_at, ssim_threshold, search_window, framerate)`**: For each cut boundary (between consecutive segments), extracts the last frame of the previous segment and the first frame of the next. Computes SSIM. If >= 0.85, the cut is visually seamless → `HARD`. If below threshold, tries shifting the boundary by up to 5 frames (in three directions: extend previous, retract next, or both) to find a better match. If a good match is found, the segment boundaries are adjusted and it becomes `HARD`. If no match exceeds threshold, it becomes `INTERPOLATE`. The search direction combinations are `(+offset, 0)`, `(0, -offset)`, `(+offset, -offset)` — meaning it can extend the previous segment's end forward, pull the next segment's start backward, or both.

### render.py

**`probe_video(video_path)`**: Calls ffprobe with JSON output. Extracts video and audio stream info. Parses framerate from the `r_frame_rate` fraction (e.g. "30000/1001"). Returns `VideoMetadata`.

**`get_frame_at_time(video_path, time_s, width, height)`**: Extracts a single frame at the given timestamp using ffmpeg. Returns an RGB numpy array. Uses `-ss` before `-i` for fast seeking.

**`build_segment_filter(segments)`**: Builds ffmpeg `select`/`aselect` filter expressions for a simple single-pass approach. Only used when there is 1 or fewer segments (no cuts needed).

**`add_padding(segments, video_duration, padding_ms=50)`**: Adds 50ms padding around each cut point. For segment i (not the first), start is pulled back by padding_ms. For segment i (not the last), end is pushed forward by padding_ms. Padding is clamped to avoid overlapping with adjacent segments. This creates a small overlap that makes cuts sound more natural.

**`_extract_segments(input_path, padded, metadata, tmpdir, crossfade_s, pause_ms)`**: Extracts each segment as an individual `.ts` file. Each segment gets:
- Audio fade-in (40ms) and fade-out (40ms) applied via ffmpeg `afade` filter.
- Optional micro-pause at the end (frozen last frame + silence padding) if `pause_ms > 0`.
- Encoded with libx264 fast preset and AAC audio.

**`_render_concat(seg_files, output_path, metadata, quality, tmpdir)`**: Writes a concat list file and uses ffmpeg's concat demuxer to join all segment files. Re-encodes with `_add_encoding_args`.

**`render_video(input_path, output_path, segments, metadata, quality, interpolated_frames, pause_ms, crossfade_overrides)`**: Main render entry point. For single segments, uses the simple `select`/`aselect` filter approach. For multiple segments, calls `add_padding`, `_extract_segments`, and `_render_concat`. The optional `crossfade_overrides` dict maps segment indices to per-segment crossfade durations (in seconds), used by the refinement loop to increase crossfade at specific cut points.

**`_add_encoding_args(cmd, metadata, quality)`**: Appends encoding arguments to an ffmpeg command. In "matched" quality mode, uses the source bitrate (or CRF 18 if bitrate is unknown). Always matches framerate, pixel format, and audio settings to the source.

### interpolate.py

**`save_frame_png(frame, directory, filename)`**: Saves a numpy RGB array as PNG using Pillow.

**`load_frame_png(path)`**: Loads a PNG as numpy array.

**`build_ncnn_command(input_dir, output_dir, num_frames)`**: Builds the rife-ncnn-vulkan command line.

**`interpolate_frames_ncnn(frame_a, frame_b, num_frames=3)`**: Saves two frames as PNGs, runs rife-ncnn-vulkan to generate intermediate frames, loads the results. Requires the `rife-ncnn-vulkan` binary to be on PATH.

**`interpolate_frames(frame_a, frame_b, num_frames, backend)`**: Dispatcher. Currently only supports "ncnn" backend. In the pipeline, if the binary is missing, the FileNotFoundError is caught and the transition falls back to HARD.

### verify.py

Post-render verification and iterative refinement. All functions operate on timestamps and word lists — no re-encoding happens here.

**`build_segment_map(segments)`**: Converts the keep-segment list into a mapping of `(orig_start, orig_end, output_offset)` tuples. Used to translate output timestamps back to original video timeline.

**`map_output_to_original(output_time, seg_map)`**: Given a timestamp in the output video, returns the corresponding timestamp in the original video by walking the segment map.

**`check_remaining_fillers(output_words, original_fillers, segments, ...)`**: Re-runs filler detection on the output transcription. For each detected filler, uses the segment map to determine if it corresponds to a known filler from pass 1 (position-based matching within 1s tolerance) or is newly detected. Returns `(remaining, new)`.

**`check_word_integrity(original_words, output_words, cut_fillers, segments)`**: Builds an expected word list (original minus intentionally cut fillers), then aligns it against the output word list using LCS with fuzzy matching. Lost words are only flagged if they fall near a cut boundary (within 0.5s of a filler edge). Damaged words are detected by checking output words at cut boundaries for truncation patterns (prefix/suffix of expected word). Returns `(lost, damaged)`.

**`check_audio_smoothness(samples, sample_rate, cut_points, ...)`**: For each cut point, extracts 50ms windows before and after, computes RMS dB difference and DC offset jump. Flags discontinuities exceeding thresholds (6dB volume, 0.05 DC offset).

**`verify_output(output_path, original_words, original_fillers, segments, ...)`**: Orchestrates all three checks. Re-transcribes the output, then runs remaining filler detection, word integrity check, and audio smoothness check. Returns a `VerificationResult`.

**`apply_adjustments(adjustments, result, segments)`**: Modifies per-filler `CutAdjustment` parameters based on verification results. Widens expansion (+150ms) for remaining fillers, reduces expansion (-100ms) for damaged words, increases crossfade (+40ms) for audio discontinuities, and adds new fillers to the adjustment map.

**`rebuild_cuts(adjustments, samples, sample_rate)`**: Re-expands filler boundaries using the adjusted parameters. Skips fillers marked with `skip=True`. Returns the new filler list and a per-filler crossfade map.

### cli.py

Argument parsing with argparse. Entry point is `main()`. Default output filename is `{input_stem}_ummfiltered.{ext}`.

### pipeline.py

`display_fillers()`: Renders a Rich table of detected fillers.

`interactive_filter()`: Prompts the user for each filler, asking whether to remove it. Returns the confirmed list.

`run_pipeline()`: Full orchestration as described in the Pipeline Flow section above.

---

## Design Decisions and Rationale

### Why segment-based concat instead of single-pass filters
A single-pass approach using `select`/`aselect` filters works for simple cases but can't apply per-segment audio fades. The segment-based approach (extract each segment as a .ts file, then concat) allows independent processing of each segment with its own fade in/out, padding, and optional pause insertion.

### Why 40ms audio fades (CROSSFADE_MS)
40ms is short enough to be imperceptible as a "fade effect" but long enough to prevent audio clicks/pops at cut boundaries. Without these fades, abrupt audio cuts produce audible artifacts.

### Why 50ms padding (PADDING_MS)
When a filler is removed, the preceding and following speech may sound unnatural if cut precisely at the filler boundary. Adding 50ms of the surrounding audio on each side creates a small buffer that makes the transition sound more natural — you get a tiny bit of the natural room tone and speech cadence.

### Why silence boundary expansion
Whisper's word timestamps are approximate (often off by 50-200ms). Cutting exactly at Whisper's boundaries can clip the beginning or end of adjacent words. By expanding into surrounding silence, the cuts happen at natural pauses in the audio, avoiding clipped speech.

### Why SSIM-based transition classification
When a filler is removed from a talking-head video, the speaker's head/body position may have changed between the end of the preceding speech and the start of the following speech. If the visual difference is small (high SSIM), a simple hard cut looks seamless. If the difference is large (low SSIM), the cut is jarring and frame interpolation is needed to smooth the transition.

### Why the 5-frame search window
Sometimes the optimal cut point isn't exactly where Whisper said the filler boundary was. By searching a few frames in each direction, we can find a moment where the speaker's position is more similar, making the hard cut less noticeable.

### Why the confidence-based hallucination filter
Whisper sometimes hallucinates filler words that weren't spoken, especially at low confidence. The `filter_fillers_by_context` function uses a heuristic: low-confidence fillers that are surrounded by pauses are likely real (the speaker paused, said "uh", paused again), while low-confidence fillers embedded in fluid speech are likely hallucinations.

### Why zero-duration filler expansion
Whisper sometimes reports fillers with identical start and end times. These are real fillers where Whisper detected the word but couldn't determine boundaries. Expanding them to fill the gap between surrounding words captures the actual filler audio.

### Why .ts container format for intermediate segments
MPEG-TS (.ts) is designed for concatenation — it doesn't require a global header, so files can be joined without re-muxing. This makes the concat demuxer approach reliable.

---

## Testing

77 tests across 9 test files. Tests use pytest with plain assertions (no mocks for core logic). Tests cover:
- Audio RMS computation and silence boundary detection
- Dictionary and contextual filler detection
- Filler confidence filtering and zero-duration expansion
- Keep-segment building and gap merging
- Frame similarity computation
- CLI argument parsing (including `--no-refine` flag)
- Video probing (integration test requiring ffmpeg)
- Segment filter generation
- Whisper parameter building and word cleaning
- Frame interpolation command building
- Segment map building and output-to-original time mapping
- Remaining filler detection (known vs new classification)
- Word integrity checking (LCS alignment, lost/damaged word detection)
- Audio smoothness checking (volume jumps, DC offset detection)
- Verification orchestration
- Cut adjustment application and rebuilding
- CutAdjustment and VerificationResult data classes

Tests are run with: `python -m pytest tests/ -v`

---

## Dependencies

**Required:**
- Python >= 3.10
- ffmpeg and ffprobe (system binaries)
- faster-whisper >= 1.0.0
- rich >= 13.0.0
- scikit-image >= 0.22.0
- numpy >= 1.24.0
- Pillow >= 10.0.0

**Optional:**
- rife-ncnn-vulkan (system binary, for frame interpolation)
- deepgram-sdk >= 3.0.0 (for cloud transcription)
- torch >= 2.0.0 (for GPU-accelerated Whisper)

---

## CLI Options Reference

| Flag | Default | Description |
|---|---|---|
| `input` | (required) | Input video file path |
| `-o`, `--output` | `{stem}_ummfiltered.{ext}` | Output video file path |
| `--model-size` | `medium` | Whisper model: tiny, base, small, medium, large |
| `--cloud` | None | Cloud transcription: deepgram |
| `--aggressive` | False | Include contextual fillers (like, so, basically, etc.) |
| `--interactive` | False | Review each filler before removing |
| `--dry-run` | False | Show detected fillers without rendering |
| `--quality` | `matched` | Output quality: matched (same bitrate) or lossless |
| `--fillers` | None | Comma-separated custom filler word list |
| `--interpolator` | `ncnn` | Frame interpolation backend |
| `--min-confidence` | `0.15` | Minimum transcription confidence threshold |
| `--pause-ms` | `0` | Insert micro-pause (frozen frame + silence) at each cut, in ms |
| `--no-refine` | False | Skip the post-render verification and refinement loop |
