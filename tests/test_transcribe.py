from unittest.mock import patch, MagicMock
from ummfiltered.transcribe import (
    _clean_word,
    _ensure_whisper_model,
    _load_deepgram_sdk,
    build_whisper_params,
    transcribe_local,
)
from ummfiltered.models import Word
from ummfiltered.config import WHISPER_FILLER_PROMPT


class TestBuildWhisperParams:
    def test_includes_filler_prompt(self):
        params = build_whisper_params("medium")
        assert params["initial_prompt"] == WHISPER_FILLER_PROMPT

    def test_disables_suppression(self):
        params = build_whisper_params("medium")
        assert params["suppress_tokens"] == []
        assert params["suppress_blank"] is False

    def test_enables_vad(self):
        params = build_whisper_params("medium")
        assert params["vad_filter"] is True

    def test_enables_word_timestamps(self):
        params = build_whisper_params("medium")
        assert params["word_timestamps"] is True


class TestCleanWord:
    def test_strips_punctuation(self):
        assert _clean_word(" uh,") == "uh"

    def test_strips_trailing_period(self):
        assert _clean_word("hello.") == "hello"

    def test_lowercases(self):
        assert _clean_word(" Um ") == "um"

    def test_multiple_punctuation(self):
        assert _clean_word("...uh...") == "uh"


class TestTranscribeLocal:
    @patch("ummfiltered.transcribe.WhisperModel")
    def test_returns_word_list(self, mock_model_cls):
        mock_word = MagicMock()
        mock_word.word = " um"
        mock_word.start = 1.0
        mock_word.end = 1.3
        mock_word.probability = 0.4

        mock_segment = MagicMock()
        mock_segment.words = [mock_word]

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], MagicMock())
        mock_model_cls.return_value = mock_model

        words = transcribe_local("test.wav", model_size="tiny")

        assert len(words) == 1
        assert words[0].text == "um"
        assert words[0].start == 1.0
        assert words[0].probability == 0.4


class TestRuntimeProvisioning:
    @patch("ummfiltered.transcribe.importlib.import_module")
    @patch("ummfiltered.transcribe._install_python_dependency")
    @patch("ummfiltered.transcribe.WhisperModel", None)
    def test_installs_faster_whisper_when_missing(self, mock_install, mock_import_module):
        fake_module = MagicMock()
        fake_module.WhisperModel = MagicMock()
        mock_import_module.return_value = fake_module

        model_cls = _ensure_whisper_model()

        mock_install.assert_called_once_with("faster-whisper>=1.0.0")
        assert model_cls is fake_module.WhisperModel

    @patch("ummfiltered.transcribe.importlib.import_module")
    @patch("ummfiltered.transcribe._install_python_dependency")
    def test_installs_deepgram_sdk_when_missing(self, mock_install, mock_import_module):
        first = ImportError("missing")
        fake_module = MagicMock()
        fake_module.DeepgramClient = MagicMock()
        fake_module.PrerecordedOptions = MagicMock()
        mock_import_module.side_effect = [first, fake_module]

        client_cls, options_cls = _load_deepgram_sdk()

        mock_install.assert_called_once_with("deepgram-sdk>=3.0.0")
        assert client_cls is fake_module.DeepgramClient
        assert options_cls is fake_module.PrerecordedOptions
