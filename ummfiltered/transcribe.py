import string
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from ummfiltered.config import WHISPER_FILLER_PROMPT
from ummfiltered.models import Word

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _clean_word(text: str) -> str:
    return text.strip().lower().translate(_PUNCT_TABLE)


def build_whisper_params(model_size: str) -> dict:
    return {
        "initial_prompt": WHISPER_FILLER_PROMPT,
        "suppress_tokens": [],
        "suppress_blank": False,
        "vad_filter": True,
        "word_timestamps": True,
    }


def transcribe_local(audio_path: str | Path, model_size: str = "large") -> list[Word]:
    if WhisperModel is None:
        raise ImportError("faster-whisper is required for local transcription")

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    params = build_whisper_params(model_size)

    segments, _info = model.transcribe(str(audio_path), **params)

    words: list[Word] = []
    for segment in segments:
        if segment.words is None:
            continue
        for w in segment.words:
            words.append(Word(
                text=_clean_word(w.word),
                start=w.start,
                end=w.end,
                probability=w.probability,
            ))
    return words


def transcribe_cloud_deepgram(audio_path: str | Path) -> list[Word]:
    from deepgram import DeepgramClient, PrerecordedOptions

    client = DeepgramClient()
    with open(audio_path, "rb") as f:
        source = {"buffer": f.read(), "mimetype": "audio/wav"}

    options = PrerecordedOptions(
        model="nova-2",
        smart_format=False,
        utterances=False,
        punctuate=False,
        filler_words=True,
    )
    response = client.listen.rest.v("1").transcribe_file(source, options)

    words: list[Word] = []
    for channel in response.results.channels:
        for alt in channel.alternatives:
            for w in alt.words:
                words.append(Word(
                    text=_clean_word(w.word),
                    start=w.start,
                    end=w.end,
                    probability=w.confidence,
                ))
    return words


def transcribe(audio_path: str | Path, model_size: str = "large", cloud: str | None = None) -> list[Word]:
    if cloud == "deepgram":
        return transcribe_cloud_deepgram(audio_path)
    return transcribe_local(audio_path, model_size)
