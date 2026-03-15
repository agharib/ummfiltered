DEFAULT_FILLERS = [
    "um", "uh", "er", "ah", "hm", "hmm", "mm", "mhm",
]

DEFAULT_FILLER_PHRASES = [
    "you know", "i mean", "kind of", "sort of",
]

CONTEXTUAL_FILLERS = [
    "like", "so", "basically", "right", "actually", "literally",
]

SILENCE_THRESHOLD_DB = -40
MAX_EXPANSION_MS = 300
MIN_PAUSE_GAP_MS = 300
MAX_FILLER_DURATION_MS = 500
SSIM_THRESHOLD = 0.85
FRAME_SEARCH_WINDOW = 5
CROSSFADE_MS = 15
PADDING_MS = 0
FILLER_MARGIN_START_MS = 50
FILLER_MARGIN_END_MS = 150
MIN_CONFIDENCE = 0.15
INTERPOLATION_FRAMES = 3

WHISPER_FILLER_PROMPT = "Um, uh, so like, you know, basically, I mean, er, ah, hmm..."


def compute_adaptive_pause(filler_duration_s: float) -> float:
    return max(0.0, filler_duration_s * 0.3 - 0.05)

