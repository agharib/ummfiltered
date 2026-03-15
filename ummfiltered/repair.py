from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Protocol
import wave

import numpy as np

from ummfiltered.audio import (
    compute_rms_db,
    downmix_audio,
    extract_audio_matrix,
    measure_cut_naturalness,
)
from ummfiltered.edit_plan import cut_points
from ummfiltered.models import EditDecisionList, RepairDecision, SeamReport, VerificationResult, VideoMetadata, Word
from ummfiltered.render import replace_audio_track


REPAIR_BLEND_MS = 20
MAX_REPAIR_WORDS = 3
MAX_REPAIR_DURATION_S = 0.9
REPAIR_WINDOW_MIN_S = 0.18
REPAIR_SEAM_THRESHOLD = 0.8
REPAIR_NEAR_BOUNDARY_S = 0.35
REPAIR_CANDIDATE_SPEEDS = (0.96, 1.0, 1.04)


@dataclass
class _RepairTarget:
    seam_index: int
    words: list[Word]
    repair_text: str
    source_start: float
    source_end: float
    output_boundary: float
    before_score: float


@dataclass
class _PatchCandidate:
    strategy: str
    audio: np.ndarray
    before_score: float
    after_score: float
    notes: str = ""


class RepairBackend(Protocol):
    def synthesize_candidates(
        self,
        text: str,
        speaker_reference: np.ndarray,
        sample_rate: int,
        channel_count: int,
    ) -> list[np.ndarray]: ...


def _install_python_dependency(requirement: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", requirement],
        capture_output=True,
        text=True,
        check=True,
    )


class XTTSRepairBackend:
    def __init__(self) -> None:
        self._tts_cls = None
        self._torch = None
        self._model = None

    def _ensure_runtime(self):
        if self._model is not None:
            return self._model
        try:
            tts_module = importlib.import_module("TTS.api")
        except ImportError:
            _install_python_dependency("torch>=2.2.0")
            _install_python_dependency("TTS>=0.22.0")
            tts_module = importlib.import_module("TTS.api")

        try:
            self._torch = importlib.import_module("torch")
        except ImportError:
            _install_python_dependency("torch>=2.2.0")
            self._torch = importlib.import_module("torch")

        self._tts_cls = tts_module.TTS
        model = self._tts_cls("tts_models/multilingual/multi-dataset/xtts_v2")
        device = "cpu"
        if hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
            device = "mps"
        if device != "cpu":
            model = model.to(device)
        self._model = model
        return self._model

    def synthesize_candidates(
        self,
        text: str,
        speaker_reference: np.ndarray,
        sample_rate: int,
        channel_count: int,
    ) -> list[np.ndarray]:
        model = self._ensure_runtime()
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            speaker_path = tmpdir / "speaker.wav"
            _write_wav(speaker_path, speaker_reference, sample_rate)
            candidates: list[np.ndarray] = []
            signature = inspect.signature(model.tts_to_file)
            for index, speed in enumerate(REPAIR_CANDIDATE_SPEEDS):
                candidate_path = tmpdir / f"candidate_{index}.wav"
                kwargs = {
                    "text": text,
                    "speaker_wav": str(speaker_path),
                    "language": "en",
                    "file_path": str(candidate_path),
                }
                if "speed" in signature.parameters:
                    kwargs["speed"] = speed
                model.tts_to_file(**kwargs)
                generated, _ = extract_audio_matrix(
                    candidate_path,
                    sample_rate=sample_rate,
                    channel_count=channel_count,
                )
                candidates.append(generated.astype(np.float32))
        return candidates


_XTTS_BACKEND: XTTSRepairBackend | None = None


def _xtts_backend() -> XTTSRepairBackend:
    global _XTTS_BACKEND
    if _XTTS_BACKEND is None:
        _XTTS_BACKEND = XTTSRepairBackend()
    return _XTTS_BACKEND


def _as_audio_matrix(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples[:, None].astype(np.float32)
    return samples.astype(np.float32)


def _fit_audio_length(samples: np.ndarray, target_length: int, channel_count: int) -> np.ndarray:
    matrix = _as_audio_matrix(samples)
    if matrix.shape[1] == 1 and channel_count > 1:
        matrix = np.repeat(matrix, channel_count, axis=1)
    elif matrix.shape[1] > channel_count:
        matrix = matrix[:, :channel_count]

    if target_length <= 0:
        return np.zeros((0, channel_count), dtype=np.float32)
    if len(matrix) == target_length:
        return matrix.astype(np.float32)
    if len(matrix) == 0:
        return np.zeros((target_length, channel_count), dtype=np.float32)

    source_positions = np.linspace(0.0, 1.0, len(matrix), endpoint=True)
    target_positions = np.linspace(0.0, 1.0, target_length, endpoint=True)
    channels = [
        np.interp(target_positions, source_positions, matrix[:, channel])
        for channel in range(channel_count)
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    matrix = _as_audio_matrix(audio)
    clipped = np.clip(matrix, -1.0, 1.0)
    audio_int16 = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(matrix.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def _nearest_cut_index(word: Word, edit_plan: EditDecisionList) -> int | None:
    if len(edit_plan.decisions) < 2:
        return None
    midpoint = (word.start + word.end) / 2.0
    candidates = [
        (abs(decision.source_end - midpoint), decision.index)
        for decision in edit_plan.decisions[:-1]
    ]
    distance, cut_index = min(candidates, default=(float("inf"), None))
    if cut_index is None or distance > REPAIR_NEAR_BOUNDARY_S:
        return None
    return cut_index


def _select_repair_targets(
    result: VerificationResult,
    edit_plan: EditDecisionList,
) -> list[_RepairTarget]:
    if len(edit_plan.decisions) < 2:
        return []

    cut_map = cut_points(edit_plan)
    by_seam: dict[int, list[Word]] = {}
    for word in result.missing_words:
        cut_index = _nearest_cut_index(word, edit_plan)
        if cut_index is None:
            continue
        by_seam.setdefault(cut_index, []).append(word)
    for word in result.lost_words:
        cut_index = _nearest_cut_index(word, edit_plan)
        if cut_index is None:
            continue
        by_seam.setdefault(cut_index, []).append(word)
    for word, cut_index in result.damaged_words:
        if cut_index not in cut_map:
            continue
        by_seam.setdefault(cut_index, []).append(word)

    seam_scores = result.seam_report.scores if result.seam_report else []
    if seam_scores:
        cutoff_index = max(0, int(round((len(sorted(seam_scores)) - 1) * 0.9)))
        worst_threshold = sorted(seam_scores)[cutoff_index]
    else:
        worst_threshold = float("inf")

    targets: list[_RepairTarget] = []
    for seam_index, words in sorted(by_seam.items()):
        unique_words = sorted(
            {(word.text, word.start, word.end): word for word in words}.values(),
            key=lambda word: (word.start, word.end),
        )
        if not unique_words or len(unique_words) > MAX_REPAIR_WORDS:
            continue
        source_start = max(0.0, unique_words[0].start - 0.02)
        source_end = unique_words[-1].end + 0.02
        if source_end - source_start > MAX_REPAIR_DURATION_S:
            continue
        repair_text = " ".join(word.text for word in unique_words).strip()
        if not repair_text:
            continue
        before_score = 0.0
        if result.seam_report and seam_index < len(result.seam_report.entries):
            before_score = result.seam_report.entries[seam_index].after_score
            if before_score < REPAIR_SEAM_THRESHOLD and before_score < worst_threshold and not result.lost_words:
                continue
        output_boundary = cut_map.get(seam_index)
        if output_boundary is None:
            continue
        targets.append(
            _RepairTarget(
                seam_index=seam_index,
                words=unique_words,
                repair_text=repair_text,
                source_start=source_start,
                source_end=source_end,
                output_boundary=output_boundary,
                before_score=before_score,
            )
        )
    return targets


def _repair_window(
    target: _RepairTarget,
    edit_plan: EditDecisionList,
    sample_rate: int,
    output_length: int,
) -> tuple[int, int]:
    boundary_source = edit_plan.decisions[target.seam_index].source_end
    word_center = (target.words[0].start + target.words[-1].end) / 2.0
    output_center = target.output_boundary + (word_center - boundary_source)
    duration_s = max(
        REPAIR_WINDOW_MIN_S,
        min(MAX_REPAIR_DURATION_S, target.source_end - target.source_start),
    )
    target_length = max(1, int(round(duration_s * sample_rate)))
    center_sample = int(round(output_center * sample_rate))
    start_sample = center_sample - target_length // 2
    end_sample = start_sample + target_length
    if start_sample < 0:
        end_sample = min(output_length, end_sample - start_sample)
        start_sample = 0
    if end_sample > output_length:
        start_sample = max(0, start_sample - (end_sample - output_length))
        end_sample = output_length
    if end_sample <= start_sample:
        end_sample = min(output_length, start_sample + target_length)
    return start_sample, end_sample


def _repair_score(
    samples: np.ndarray,
    sample_rate: int,
    seam_sample: int,
    start_sample: int,
    end_sample: int,
) -> float:
    matrix = _as_audio_matrix(samples)
    points = [
        seam_sample,
        max(1, min(start_sample + 1, len(matrix) - 1)),
        max(1, min(end_sample - 1, len(matrix) - 1)),
    ]
    scores = [
        measure_cut_naturalness(matrix, sample_rate, point / sample_rate).score
        for point in points
        if 0 < point < len(matrix)
    ]
    return float(sum(scores) / max(1, len(scores)))


def _apply_patch_window(
    base_audio: np.ndarray,
    patch_audio: np.ndarray,
    start_sample: int,
    end_sample: int,
    blend_samples: int,
) -> np.ndarray:
    base = _as_audio_matrix(base_audio).copy()
    channel_count = base.shape[1]
    target_length = max(0, end_sample - start_sample)
    patch = _fit_audio_length(patch_audio, target_length, channel_count)
    if target_length == 0:
        return base.astype(np.float32)

    blend = min(blend_samples, target_length // 2)
    if blend == 0:
        base[start_sample:end_sample] = patch
        return base.astype(np.float32)

    fade = np.linspace(0.0, 1.0, blend, dtype=np.float32)[:, None]
    base[start_sample:start_sample + blend] = (
        base[start_sample:start_sample + blend] * (1.0 - fade)
        + patch[:blend] * fade
    )
    if target_length > blend * 2:
        base[start_sample + blend:end_sample - blend] = patch[blend:target_length - blend]
    tail_fade = np.linspace(1.0, 0.0, blend, dtype=np.float32)[:, None]
    base[end_sample - blend:end_sample] = (
        patch[target_length - blend:] * tail_fade
        + base[end_sample - blend:end_sample] * (1.0 - tail_fade)
    )
    return base.astype(np.float32)


def _speaker_reference_audio(
    source_audio: np.ndarray,
    sample_rate: int,
    target: _RepairTarget,
) -> np.ndarray:
    total_duration = len(_as_audio_matrix(source_audio)) / sample_rate
    start = max(0.0, target.source_start - 4.0)
    end = min(total_duration, target.source_end + 4.0)
    if end - start < 2.0:
        start = max(0.0, target.source_start - 1.0)
        end = min(total_duration, target.source_end + 1.0)
    start_sample = int(round(start * sample_rate))
    end_sample = int(round(end * sample_rate))
    return _as_audio_matrix(source_audio)[start_sample:end_sample]


def _candidate_from_source_audio(
    output_audio: np.ndarray,
    source_audio: np.ndarray,
    sample_rate: int,
    target: _RepairTarget,
    edit_plan: EditDecisionList,
) -> _PatchCandidate | None:
    source_matrix = _as_audio_matrix(source_audio)
    start_sample, end_sample = _repair_window(target, edit_plan, sample_rate, len(_as_audio_matrix(output_audio)))
    source_start_sample = max(0, int(round(target.source_start * sample_rate)))
    source_end_sample = min(len(source_matrix), int(round(target.source_end * sample_rate)))
    phrase = source_matrix[source_start_sample:source_end_sample]
    if len(phrase) == 0:
        return None
    seam_sample = int(round(target.output_boundary * sample_rate))
    before_score = _repair_score(output_audio, sample_rate, seam_sample, start_sample, end_sample)
    blend_samples = max(1, int(round(REPAIR_BLEND_MS / 1000.0 * sample_rate)))
    best_candidate: _PatchCandidate | None = None
    for shift_ms in (-20, 0, 20):
        shift_samples = int(round(shift_ms / 1000.0 * sample_rate))
        shifted_start = max(0, min(start_sample + shift_samples, len(_as_audio_matrix(output_audio)) - 1))
        shifted_end = min(len(_as_audio_matrix(output_audio)), shifted_start + max(1, end_sample - start_sample))
        if shifted_end - shifted_start < 4:
            continue
        patched = _apply_patch_window(output_audio, phrase, shifted_start, shifted_end, blend_samples)
        after_score = _repair_score(patched, sample_rate, seam_sample, shifted_start, shifted_end)
        candidate = _PatchCandidate(
            strategy="source_phrase_patch",
            audio=patched,
            before_score=before_score,
            after_score=after_score,
            notes=f"shift={shift_ms}ms",
        )
        if best_candidate is None or candidate.after_score < best_candidate.after_score:
            best_candidate = candidate
    return best_candidate


def _candidate_from_xtts(
    output_audio: np.ndarray,
    source_audio: np.ndarray,
    sample_rate: int,
    target: _RepairTarget,
    edit_plan: EditDecisionList,
    backend: RepairBackend,
) -> _PatchCandidate | None:
    start_sample, end_sample = _repair_window(target, edit_plan, sample_rate, len(_as_audio_matrix(output_audio)))
    target_length = end_sample - start_sample
    if target_length <= 0:
        return None
    channel_count = _as_audio_matrix(output_audio).shape[1]
    speaker_reference = _speaker_reference_audio(source_audio, sample_rate, target)
    if len(speaker_reference) == 0:
        return None
    seam_sample = int(round(target.output_boundary * sample_rate))
    before_score = _repair_score(output_audio, sample_rate, seam_sample, start_sample, end_sample)
    blend_samples = max(1, int(round(REPAIR_BLEND_MS / 1000.0 * sample_rate)))
    best_candidate: _PatchCandidate | None = None
    try:
        synth_candidates = backend.synthesize_candidates(
            target.repair_text,
            speaker_reference,
            sample_rate,
            channel_count,
        )
    except Exception as exc:
        return _PatchCandidate(
            strategy="xtts_fallback",
            audio=_as_audio_matrix(output_audio),
            before_score=before_score,
            after_score=before_score,
            notes=f"backend unavailable: {exc}",
        )

    for index, synth in enumerate(synth_candidates):
        patched = _apply_patch_window(output_audio, synth, start_sample, end_sample, blend_samples)
        after_score = _repair_score(patched, sample_rate, seam_sample, start_sample, end_sample)
        candidate = _PatchCandidate(
            strategy="xtts_phrase_patch",
            audio=patched,
            before_score=before_score,
            after_score=after_score,
            notes=f"candidate={index}",
        )
        if best_candidate is None or candidate.after_score < best_candidate.after_score:
            best_candidate = candidate
    return best_candidate


def merge_repair_decisions(
    seam_report: SeamReport,
    repair_decisions: list[RepairDecision],
) -> SeamReport:
    by_seam = {decision.seam_index: decision for decision in repair_decisions}
    for entry in seam_report.entries:
        decision = by_seam.get(entry.seam_index)
        if decision is None:
            continue
        entry.repair_strategy = decision.strategy
        entry.repair_text = decision.repair_text
        entry.repair_accepted = decision.accepted
        entry.repair_notes = decision.notes
    return seam_report


def repair_output_audio(
    *,
    output_path: Path,
    source_audio: np.ndarray,
    sample_rate: int,
    metadata: VideoMetadata,
    edit_plan: EditDecisionList,
    verification: VerificationResult,
) -> list[RepairDecision]:
    targets = _select_repair_targets(verification, edit_plan)
    if not targets:
        return []

    output_audio, _ = extract_audio_matrix(
        output_path,
        sample_rate=sample_rate,
        channel_count=metadata.audio_channels,
    )
    repaired_audio = _as_audio_matrix(output_audio)
    decisions: list[RepairDecision] = []

    for target in targets:
        source_candidate = _candidate_from_source_audio(
            repaired_audio,
            source_audio,
            sample_rate,
            target,
            edit_plan,
        )
        candidates = [candidate for candidate in [source_candidate] if candidate is not None]

        needs_xtts_fallback = (
            source_candidate is None
            or source_candidate.after_score > source_candidate.before_score + 0.05
        )
        if needs_xtts_fallback:
            xtts_candidate = _candidate_from_xtts(
                repaired_audio,
                source_audio,
                sample_rate,
                target,
                edit_plan,
                _xtts_backend(),
            )
            if xtts_candidate is not None:
                candidates.append(xtts_candidate)

        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: candidate.after_score)
        accepted = best.after_score <= best.before_score + 0.1
        if accepted and best.strategy != "xtts_fallback":
            repaired_audio = _as_audio_matrix(best.audio)
        decisions.append(
            RepairDecision(
                seam_index=target.seam_index,
                strategy=best.strategy,
                repair_text=target.repair_text,
                window_start=_repair_window(target, edit_plan, sample_rate, len(repaired_audio))[0] / sample_rate,
                window_end=_repair_window(target, edit_plan, sample_rate, len(repaired_audio))[1] / sample_rate,
                before_score=best.before_score,
                after_score=best.after_score,
                accepted=accepted and best.strategy != "xtts_fallback",
                notes=best.notes,
            )
        )

    if any(decision.accepted for decision in decisions):
        replace_audio_track(output_path, repaired_audio, sample_rate, metadata)
    return decisions
