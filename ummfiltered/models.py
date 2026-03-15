from dataclasses import dataclass, field
from enum import Enum


class TransitionType(Enum):
    HARD = "hard"
    INTERPOLATE = "interpolate"


class DetectionSource(Enum):
    DICTIONARY = "dictionary"
    CONTEXTUAL = "contextual"


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass
class FillerSegment:
    start: float
    end: float
    word: str
    confidence: float
    source: DetectionSource


@dataclass
class Segment:
    start: float
    end: float
    transition_type: TransitionType
    visual_gap_score: float


@dataclass
class RemovedGap:
    start: float
    end: float
    reason: str = "filler"


@dataclass
class EditDecision:
    index: int
    source_start: float
    source_end: float
    output_start: float
    output_end: float
    transition_type: TransitionType
    pause_after: float = 0.0
    transition_duration_after: float = 0.0
    removed_gap_after: RemovedGap | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.source_end - self.source_start)


@dataclass
class EditDecisionList:
    decisions: list[EditDecision]
    total_output_duration: float

    def cut_points(self) -> list[float]:
        return [decision.output_end for decision in self.decisions[:-1]]


@dataclass
class VideoMetadata:
    codec: str
    width: int
    height: int
    framerate: float
    bitrate: int
    pixel_format: str
    duration: float
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int
    audio_bitrate: int


@dataclass
class CutAdjustment:
    filler: FillerSegment
    expansion_ms: float
    crossfade_ms: float
    skip: bool = False


@dataclass
class VerificationResult:
    remaining_fillers: list[FillerSegment]
    new_fillers: list[FillerSegment]
    lost_words: list[Word]
    damaged_words: list[tuple[Word, int]]
    audio_discontinuities: list[tuple[float, float]]
    preserved_word_recall: float = 1.0
    max_missing_run: int = 0
    seam_report: "SeamReport | None" = None

    def is_clean(self) -> bool:
        return not any([
            self.remaining_fillers,
            self.new_fillers,
            self.lost_words,
            self.damaged_words,
            self.audio_discontinuities,
        ])

    def needs_rerender(self) -> bool:
        return not self.is_clean()


@dataclass
class SeamCandidate:
    strategy: str
    score: float
    before_score: float
    left_shift_ms: float = 0.0
    right_shift_ms: float = 0.0
    duration_ms: float = 0.0
    accepted: bool = True
    notes: str = ""


@dataclass
class SeamReportEntry:
    seam_index: int
    output_time: float
    chosen_strategy: str
    before_score: float
    after_score: float
    left_shift_ms: float
    right_shift_ms: float
    duration_ms: float
    accepted: bool
    notes: str = ""


@dataclass
class SeamReport:
    entries: list[SeamReportEntry] = field(default_factory=list)

    @property
    def scores(self) -> list[float]:
        return [entry.after_score for entry in self.entries]

    @property
    def median_score(self) -> float:
        if not self.entries:
            return 0.0
        ordered = sorted(self.scores)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    @property
    def p95_score(self) -> float:
        if not self.entries:
            return 0.0
        ordered = sorted(self.scores)
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * 0.95))))
        return ordered[idx]
