from ummfiltered.config import (
    CONTEXTUAL_FILLERS,
    DEFAULT_FILLER_PHRASES,
    DEFAULT_FILLERS,
    MAX_FILLER_DURATION_MS,
    MIN_CONFIDENCE,
    MIN_PAUSE_GAP_MS,
)
from ummfiltered.models import DetectionSource, FillerSegment, Word

PAUSE_THRESHOLD_S = 0.15


def detect_fillers_dictionary(
    words: list[Word],
    custom_fillers: list[str] | None = None,
) -> list[FillerSegment]:
    if custom_fillers is not None:
        single_fillers = set(custom_fillers)
        phrase_fillers: list[list[str]] = []
    else:
        single_fillers = set(DEFAULT_FILLERS)
        phrase_fillers = [p.split() for p in DEFAULT_FILLER_PHRASES]

    fillers: list[FillerSegment] = []
    skip_until = -1

    for i, word in enumerate(words):
        if i <= skip_until:
            continue

        matched_phrase = False
        for phrase_words in phrase_fillers:
            if i + len(phrase_words) <= len(words):
                candidate = [words[i + j].text for j in range(len(phrase_words))]
                if candidate == phrase_words:
                    fillers.append(FillerSegment(
                        start=words[i].start,
                        end=words[i + len(phrase_words) - 1].end,
                        word=" ".join(phrase_words),
                        confidence=min(words[i + j].probability for j in range(len(phrase_words))),
                        source=DetectionSource.DICTIONARY,
                    ))
                    skip_until = i + len(phrase_words) - 1
                    matched_phrase = True
                    break

        if not matched_phrase and word.text in single_fillers:
            fillers.append(FillerSegment(
                start=word.start,
                end=word.end,
                word=word.text,
                confidence=word.probability,
                source=DetectionSource.DICTIONARY,
            ))

    return fillers


def detect_fillers_contextual(words: list[Word]) -> list[FillerSegment]:
    contextual_set = set(CONTEXTUAL_FILLERS)
    min_gap_s = MIN_PAUSE_GAP_MS / 1000.0
    max_dur_s = MAX_FILLER_DURATION_MS / 1000.0
    fillers: list[FillerSegment] = []

    for i, word in enumerate(words):
        if word.text not in contextual_set:
            continue

        duration = word.end - word.start
        if duration > max_dur_s:
            continue

        if word.probability > 0.5:
            continue

        gap_before = word.start - words[i - 1].end if i > 0 else float("inf")
        gap_after = words[i + 1].start - word.end if i < len(words) - 1 else float("inf")

        if gap_before > min_gap_s or gap_after > min_gap_s:
            fillers.append(FillerSegment(
                start=word.start,
                end=word.end,
                word=word.text,
                confidence=word.probability,
                source=DetectionSource.CONTEXTUAL,
            ))

    return fillers


def detect_fillers(
    words: list[Word],
    aggressive: bool = False,
    custom_fillers: list[str] | None = None,
) -> list[FillerSegment]:
    fillers = detect_fillers_dictionary(words, custom_fillers)

    if aggressive:
        dict_ranges = {(f.start, f.end) for f in fillers}
        contextual = detect_fillers_contextual(words)
        for c in contextual:
            if (c.start, c.end) not in dict_ranges:
                fillers.append(c)

    fillers.sort(key=lambda f: f.start)
    return fillers


def filter_fillers_by_context(
    fillers: list[FillerSegment],
    words: list[Word],
    min_confidence: float = MIN_CONFIDENCE,
) -> list[FillerSegment]:
    single_fillers = set(DEFAULT_FILLERS)
    kept: list[FillerSegment] = []

    for f in fillers:
        if f.confidence >= min_confidence:
            kept.append(f)
            continue

        if f.word not in single_fillers:
            continue

        gap_before = float("inf")
        gap_after = float("inf")
        for w in words:
            if w.end <= f.start and w.start < f.start:
                gap_before = min(gap_before, f.start - w.end)
            if w.start >= f.end and w.end > f.end:
                gap_after = min(gap_after, w.start - f.end)

        if gap_before > PAUSE_THRESHOLD_S or gap_after > PAUSE_THRESHOLD_S:
            kept.append(f)

    kept.sort(key=lambda filler: filler.start)
    return kept


MIN_FILLER_DURATION_S = 0.05


def expand_zero_duration_fillers(
    fillers: list[FillerSegment],
    words: list[Word],
) -> list[FillerSegment]:
    result: list[FillerSegment] = []
    for f in fillers:
        if f.end - f.start >= MIN_FILLER_DURATION_S:
            result.append(f)
            continue

        prev_end = 0.0
        next_start = float("inf")
        for w in words:
            if w.end <= f.start and w.text != f.word:
                prev_end = max(prev_end, w.end)
            if w.start >= f.end and w.text != f.word and w.start < next_start:
                next_start = w.start

        new_start = max(prev_end, f.start - 0.5)
        new_end = min(next_start, f.end + 0.5)
        if new_end - new_start >= MIN_FILLER_DURATION_S:
            result.append(FillerSegment(
                start=new_start, end=new_end,
                word=f.word, confidence=f.confidence, source=f.source,
            ))

    return result
