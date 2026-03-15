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
    lead_padding: float = 0.0
    trail_padding: float = 0.0
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
    contract_tokens: list[str] = field(default_factory=list)

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
    missing_words: list[Word] = field(default_factory=list)
    output_words: list[Word] = field(default_factory=list)
    contract_tokens: list[str] = field(default_factory=list)
    missing_tokens: list[str] = field(default_factory=list)
    contract_intact: bool | None = None

    def __post_init__(self) -> None:
        if self.contract_intact is None:
            self.contract_intact = (
                not self.missing_tokens
                and not self.missing_words
                and not self.lost_words
                and not self.damaged_words
                and self.max_missing_run == 0
            )

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
    repair_strategy: str | None = None
    repair_text: str | None = None
    repair_accepted: bool = False
    repair_notes: str = ""


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


@dataclass
class RepairCandidate:
    seam_index: int
    strategy: str
    repair_text: str
    window_start: float
    window_end: float
    before_score: float
    after_score: float
    accepted: bool
    notes: str = ""


@dataclass
class RepairDecision:
    seam_index: int
    strategy: str
    repair_text: str
    window_start: float
    window_end: float
    before_score: float
    after_score: float
    accepted: bool
    notes: str = ""


class PipelineStage(str, Enum):
    PROBE = "probe"
    EXTRACT_AUDIO = "extract_audio"
    TRANSCRIBE = "transcribe"
    DETECT_FILLERS = "detect_fillers"
    PLAN_CUTS = "plan_cuts"
    RENDER = "render"
    VERIFY = "verify"
    FINAL_CHECK = "final_check"


class PipelineEventKind(str, Enum):
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    INFO = "info"
    WARNING = "warning"
    RESULT = "result"
    ERROR = "error"
    CANCELLED = "cancelled"


class PipelineFinalStatus(str, Enum):
    SUCCESS = "success"
    NO_FILLERS = "no_fillers"
    DRY_RUN = "dry_run"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class PipelineEvent:
    kind: PipelineEventKind
    stage: PipelineStage | None
    message: str
    warning: str | None = None
    stats: dict[str, int | float | str | bool] | None = None


@dataclass
class PipelineResult:
    outputPath: str
    removedFillers: int
    removedSeconds: float
    warnings: list[str] = field(default_factory=list)
    finalStatus: PipelineFinalStatus = PipelineFinalStatus.SUCCESS


class GuiPreset(str, Enum):
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


@dataclass
class GuiOverrides:
    modelSize: str | None = None
    quality: str | None = None
    minConfidence: float | None = None
    customFillers: list[str] | None = None
    fixedPauseMs: float | None = None


@dataclass
class GuiProcessRequest:
    inputPath: str
    outputPath: str | None
    preset: GuiPreset
    aggressive: bool
    verifyPass: bool
    naturalPauses: bool
    overrides: GuiOverrides = field(default_factory=GuiOverrides)
