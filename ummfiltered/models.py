from dataclasses import dataclass
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

    def needs_rerender(self) -> bool:
        return bool(self.damaged_words)
