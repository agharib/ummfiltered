import json
import subprocess
import tempfile
from pathlib import Path
import wave

import numpy as np
from PIL import Image

from ummfiltered.config import CROSSFADE_MS, PADDING_MS
from ummfiltered.ffmpeg_tools import ffmpeg_cmd, ffprobe_cmd
from ummfiltered.models import EditDecisionList, Segment, TransitionType, VideoMetadata


def probe_video(video_path: Path) -> VideoMetadata:
    result = subprocess.run(
        ffprobe_cmd(
            "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path,
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)

    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio_stream = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    framerate = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    return VideoMetadata(
        codec=video_stream["codec_name"],
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        framerate=framerate,
        bitrate=int(video_stream.get("bit_rate", data["format"].get("bit_rate", 0))),
        pixel_format=video_stream.get("pix_fmt", "yuv420p"),
        duration=float(data["format"]["duration"]),
        audio_codec=audio_stream["codec_name"] if audio_stream else "aac",
        audio_sample_rate=int(audio_stream["sample_rate"]) if audio_stream else 44100,
        audio_channels=int(audio_stream["channels"]) if audio_stream else 2,
        audio_bitrate=int(audio_stream.get("bit_rate", 128000)) if audio_stream else 128000,
    )


def get_frame_at_time(video_path: Path, time_s: float, width: int, height: int) -> np.ndarray:
    time_s = max(0.0, time_s)
    result = subprocess.run(
        ffmpeg_cmd(
            "-ss", time_s, "-i", video_path,
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ),
        capture_output=True,
        check=True,
    )
    expected_size = height * width * 3
    if len(result.stdout) < expected_size:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def build_segment_filter(segments: list[Segment]) -> tuple[str, str]:
    video_parts = []
    audio_parts = []
    for i, seg in enumerate(segments):
        video_parts.append(f"between(t,{seg.start},{seg.end})")
        audio_parts.append(f"between(t,{seg.start},{seg.end})")

    vf = f"select='{'+'.join(video_parts)}',setpts=N/FRAME_RATE/TB"
    af = f"aselect='{'+'.join(audio_parts)}',asetpts=N/SR/TB"
    return vf, af


def add_padding(segments: list[Segment], video_duration: float, padding_ms: float = PADDING_MS) -> list[Segment]:
    padding_s = padding_ms / 1000.0
    padded = []
    for i, seg in enumerate(segments):
        start = seg.start
        end = seg.end

        if i > 0:
            start = max(seg.start - padding_s, padded[-1].end if padded else 0.0)
        if i < len(segments) - 1:
            end = min(seg.end + padding_s, video_duration)

        padded.append(Segment(
            start=start,
            end=end,
            transition_type=seg.transition_type,
            visual_gap_score=seg.visual_gap_score,
        ))
    return padded


def _extract_segments(
    input_path: Path,
    padded: list[Segment],
    metadata: VideoMetadata,
    tmpdir: str,
    crossfade_s: float,
    pause_ms: float = 0,
    crossfade_overrides: dict[int, float] | None = None,
    pause_overrides: dict[int, float] | None = None,
) -> list[tuple[int, Path]]:
    pause_s = pause_ms / 1000.0
    seg_files: list[tuple[int, Path]] = []
    for i, seg in enumerate(padded):
        duration = seg.end - seg.start
        if duration < 0.001:
            continue
        seg_path = Path(tmpdir) / f"seg_{i:04d}.mp4"
        seg_files.append((i, seg_path))

        vf_filters = []

        seg_pause_s = pause_overrides.get(i, pause_s) if pause_overrides else pause_s
        if seg_pause_s > 0 and i < len(padded) - 1:
            vf_filters.append(f"tpad=stop_duration={seg_pause_s}:stop_mode=clone")

        video_filter = ",".join([
            f"trim=start={seg.start:.4f}:duration={duration:.4f}",
            "setpts=PTS-STARTPTS",
            *vf_filters,
        ])

        cmd = ffmpeg_cmd(
            "-y",
            "-i", input_path,
            "-filter:v", video_filter,
            "-an",
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", metadata.pixel_format,
            "-r", str(metadata.framerate),
            seg_path,
        )
        subprocess.run(cmd, capture_output=True, check=True)
    return seg_files


def _render_concat(seg_files: list[Path], output_path: Path, metadata: VideoMetadata, quality: str, tmpdir: str) -> None:
    concat_list = Path(tmpdir) / "concat.txt"
    with open(concat_list, "w") as f:
        for seg_path in seg_files:
            f.write(f"file '{seg_path}'\n")

    cmd = ffmpeg_cmd(
        "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
    )
    _add_encoding_args(cmd, metadata, quality, include_audio=False)
    cmd.append(str(output_path))
    subprocess.run(cmd, capture_output=True, check=True)


def _render_interpolation_clip(
    frames: list[np.ndarray],
    metadata: VideoMetadata,
    tmpdir: str,
    clip_idx: int,
) -> Path:
    frame_dir = Path(tmpdir) / f"interp_{clip_idx:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_dir / f"{frame_idx:04d}.png")

    clip_path = Path(tmpdir) / f"interp_{clip_idx:04d}.mp4"
    duration_s = len(frames) / metadata.framerate
    cmd = ffmpeg_cmd(
        "-y",
        "-framerate", metadata.framerate,
        "-i", frame_dir / "%04d.png",
        "-c:v", "libx264", "-preset", "fast",
        "-pix_fmt", metadata.pixel_format,
        "-r", str(metadata.framerate),
        "-t", f"{duration_s:.4f}",
        "-an",
        clip_path,
    )
    subprocess.run(cmd, capture_output=True, check=True)
    return clip_path


def render_video(
    input_path: Path,
    output_path: Path,
    segments: list[Segment],
    metadata: VideoMetadata,
    quality: str = "matched",
    interpolated_frames: dict[int, list[np.ndarray]] | None = None,
    pause_ms: float = 0,
    crossfade_overrides: dict[int, float] | None = None,
    pause_overrides: dict[int, float] | None = None,
    edit_plan: EditDecisionList | None = None,
) -> None:
    if edit_plan is not None:
        segments = [
            Segment(
                start=decision.source_start,
                end=decision.source_end,
                transition_type=decision.transition_type,
                visual_gap_score=1.0,
            )
            for decision in edit_plan.decisions
        ]
        pause_overrides = {
            decision.index: decision.pause_after
            for decision in edit_plan.decisions
            if decision.pause_after > 0
        } or None

    padded = add_padding(segments, metadata.duration)
    crossfade_s = CROSSFADE_MS / 1000.0

    with tempfile.TemporaryDirectory() as tmpdir:
        seg_files = _extract_segments(
            input_path, padded, metadata, tmpdir,
            crossfade_s=crossfade_s,
            pause_ms=pause_ms,
            crossfade_overrides=crossfade_overrides,
            pause_overrides=pause_overrides,
        )
        ordered_files: list[Path] = []
        for seg_idx, seg_path in seg_files:
            ordered_files.append(seg_path)
            frames = interpolated_frames.get(seg_idx + 1) if interpolated_frames else None
            if frames:
                ordered_files.append(_render_interpolation_clip(frames, metadata, tmpdir, seg_idx + 1))
        _render_concat(ordered_files, output_path, metadata, quality, tmpdir)


def replace_audio_track(
    video_path: Path,
    audio_samples: np.ndarray,
    sample_rate: int,
    metadata: VideoMetadata,
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    clipped_audio = np.clip(audio_samples, -1.0, 1.0)
    if clipped_audio.ndim == 1:
        channel_count = max(1, metadata.audio_channels)
        if channel_count == 1:
            audio_matrix = clipped_audio[:, None]
        else:
            audio_matrix = np.repeat(clipped_audio[:, None], channel_count, axis=1)
    else:
        audio_matrix = clipped_audio
        channel_count = audio_matrix.shape[1]

    audio_int16 = (audio_matrix * 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(channel_count)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    tmp_output = str(video_path) + ".tmp.mp4"
    subprocess.run(
        ffmpeg_cmd(
            "-y",
            "-i", video_path, "-i", wav_path,
            "-c:v", "copy", "-c:a", "aac",
            "-ar", metadata.audio_sample_rate,
            "-ac", channel_count,
            "-b:a", metadata.audio_bitrate,
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            tmp_output,
        ),
        capture_output=True,
        check=True,
    )

    Path(wav_path).unlink()
    Path(tmp_output).replace(video_path)


def _add_encoding_args(cmd: list[str], metadata: VideoMetadata, quality: str, include_audio: bool = True) -> None:
    if quality == "matched" and metadata.bitrate > 0:
        cmd.extend(["-b:v", str(metadata.bitrate)])
    elif quality == "matched":
        cmd.extend(["-crf", "18"])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", "medium",
        "-pix_fmt", metadata.pixel_format,
        "-r", str(metadata.framerate),
    ])
    if include_audio:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", str(metadata.audio_bitrate),
            "-ar", str(metadata.audio_sample_rate),
            "-ac", str(metadata.audio_channels),
        ])
