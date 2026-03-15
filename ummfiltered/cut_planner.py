import numpy as np
from skimage.metrics import structural_similarity as ssim

from ummfiltered.config import MIN_PAUSE_GAP_MS, SSIM_THRESHOLD
from ummfiltered.models import FillerSegment, Segment, TransitionType


def compute_frame_similarity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    if frame_a.ndim == 3:
        return ssim(frame_a, frame_b, channel_axis=2, data_range=255)
    return ssim(frame_a, frame_b, data_range=255)


def _gap_has_content_words(gap_start: float, gap_end: float, words: list) -> bool:
    return any(w.start >= gap_start and w.end <= gap_end for w in words)


def build_keep_segments(
    fillers: list[FillerSegment],
    video_duration: float,
    words: list | None = None,
) -> list[Segment]:
    if not fillers:
        return [Segment(start=0.0, end=video_duration, transition_type=TransitionType.HARD, visual_gap_score=1.0)]

    sorted_fillers = sorted(fillers, key=lambda f: f.start)
    segments: list[Segment] = []
    current_start = 0.0

    min_gap = MIN_PAUSE_GAP_MS / 1000.0

    for filler in sorted_fillers:
        gap = filler.start - current_start
        has_words = words is not None and _gap_has_content_words(current_start, filler.start, words)
        if gap >= min_gap or has_words:
            segments.append(Segment(
                start=current_start,
                end=filler.start,
                transition_type=TransitionType.HARD,
                visual_gap_score=1.0,
            ))
        current_start = filler.end

    if current_start < video_duration:
        segments.append(Segment(
            start=current_start,
            end=video_duration,
            transition_type=TransitionType.HARD,
            visual_gap_score=1.0,
        ))

    return segments


def classify_transitions(
    segments: list[Segment],
    get_frame_at: callable,
    ssim_threshold: float = SSIM_THRESHOLD,
    search_window: int = 5,
    framerate: float = 30.0,
) -> list[Segment]:
    for i in range(1, len(segments)):
        prev_end_time = segments[i - 1].end
        curr_start_time = segments[i].start

        frame_a = get_frame_at(prev_end_time)
        frame_b = get_frame_at(curr_start_time)
        score = compute_frame_similarity(frame_a, frame_b)

        if score >= ssim_threshold:
            segments[i].transition_type = TransitionType.HARD
            segments[i].visual_gap_score = score
            continue

        best_score = score
        frame_duration = 1.0 / framerate
        best_prev_end = prev_end_time
        best_curr_start = curr_start_time
        for offset in range(1, search_window + 1):
            for da, db in [(offset, 0), (0, offset), (offset, offset)]:
                t_a = max(segments[i - 1].start, prev_end_time - da * frame_duration)
                t_b = min(segments[i].end, curr_start_time + db * frame_duration)
                if t_a >= prev_end_time or t_b <= curr_start_time:
                    continue
                fa = get_frame_at(t_a)
                fb = get_frame_at(t_b)
                s = compute_frame_similarity(fa, fb)
                if s > best_score:
                    best_score = s
                    best_prev_end = t_a
                    best_curr_start = t_b

        segments[i - 1].end = best_prev_end
        segments[i].start = best_curr_start

        if best_score >= ssim_threshold:
            segments[i].transition_type = TransitionType.HARD
        else:
            segments[i].transition_type = TransitionType.INTERPOLATE

        segments[i].visual_gap_score = best_score

    return segments
