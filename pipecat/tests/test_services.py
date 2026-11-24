from typing import Any

import pytest
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)

from hn_pipecat.config import STTSettings, TTSSettings
from hn_pipecat.providers import LOCAL_PLACEHOLDER_KEY, Choice, select_stt, select_tts
from hn_pipecat.services import (
    DEFAULT_CARTESIA_VOICE,
    DEFAULT_ELEVENLABS_VOICE,
    DetectingOpenAISTTService,
    build_llm,
    build_stt,
    build_tts,
    stt_language,
    turn_strategies,
)

SECRET = "sk-very-secret-value"
ENV = {"KEY": SECRET}


def choice(
    provider: str, *, model: str | None = None, voice: str | None = None, **kw: Any
) -> Choice:
    return Choice(
        provider=provider,
        model=model,
        base_url=kw.get("base_url"),
        key_env=kw.get("key_env", "KEY"),
        voice=voice,
    )


def client_of(service: Any) -> Any:
    return service._client


# -- LLM ------------------------------------------------------------------------


def test_openai_llm_gets_key_base_url_model_and_prompt() -> None:
    svc = build_llm(
        choice("openai", model="llama-3.3", base_url="https://llm.example/v1"), ENV, "Be brief."
    )
    assert isinstance(svc, OpenAILLMService)
    assert svc._settings.model == "llama-3.3"
    assert svc._settings.system_instruction == "Be brief."
    assert client_of(svc).api_key == SECRET
    assert str(client_of(svc).base_url).rstrip("/") == "https://llm.example/v1"


def test_hypernoesis_llm_is_openai_compatible_with_the_gateway() -> None:
    svc = build_llm(
        choice("hypernoesis", base_url="https://gateway.example/v1", key_env="KEY"), ENV, "x"
    )
    assert isinstance(svc, OpenAILLMService)
    assert svc._settings.model == "gpt-4.1-mini"
    assert str(client_of(svc).base_url).rstrip("/") == "https://gateway.example/v1"


def test_anthropic_and_google_llm() -> None:
    svc = build_llm(choice("anthropic"), ENV, "x")
    assert isinstance(svc, AnthropicLLMService)
    assert svc._settings.model == "claude-haiku-4-5"
    svc = build_llm(choice("google"), ENV, "x")
    assert isinstance(svc, GoogleLLMService)
    assert svc._settings.system_instruction == "x"
    svc = build_llm(choice("google", model="gemini-flash-latest"), ENV, "x")
    assert svc._settings.model == "gemini-flash-latest"


# -- STT ------------------------------------------------------------------------


def test_stt_language_rules() -> None:
    assert stt_language("deepgram", "en") == "en"
    assert stt_language("deepgram", "auto") == "el"
    assert stt_language("speechmatics", "auto") == "el"
    assert stt_language("elevenlabs", "auto") is None
    assert stt_language("local", "auto") is None


@pytest.mark.parametrize(
    ("provider", "cls"),
    [
        ("deepgram", DeepgramSTTService),
        ("elevenlabs", ElevenLabsRealtimeSTTService),
        ("speechmatics", SpeechmaticsSTTService),
    ],
)
def test_streaming_stt_services(provider: str, cls: type) -> None:
    assert isinstance(build_stt(choice(provider), "el", ENV), cls)


def test_deepgram_listens_for_greek_in_auto_mode() -> None:
    svc = build_stt(choice("deepgram", model="nova-3"), "auto", ENV)
    assert svc._settings.model == "nova-3"
    assert str(svc._settings.language) == "el"


def test_elevenlabs_detects_the_language_in_auto_mode() -> None:
    svc = build_stt(choice("elevenlabs"), "auto", ENV)
    assert svc._settings.language is None
    assert svc._settings.model == "scribe_v2_realtime"


def test_local_stt_uses_its_url_and_no_real_key() -> None:
    env = {"HN_STT_URL": "http://hn-stt:8000/v1", "HN_STT_MODEL": "large-v3-turbo"}
    picked = select_stt(STTSettings(), env)
    assert picked is not None
    svc = build_stt(picked, "el", env)
    assert type(svc) is OpenAISTTService
    assert svc._settings.model == "large-v3-turbo"
    assert str(svc._settings.language) == "el"
    assert client_of(svc).api_key == LOCAL_PLACEHOLDER_KEY
    assert str(client_of(svc).base_url).rstrip("/") == "http://hn-stt:8000/v1"


class FakeTranscriptions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return type("T", (), {"text": "γεια σας"})()


async def test_auto_mode_lets_the_endpoint_detect_the_language() -> None:
    svc = build_stt(choice("local", key_env=None, base_url="http://hn-stt:8000/v1"), "auto", {})
    assert isinstance(svc, DetectingOpenAISTTService)
    fake = FakeTranscriptions()
    svc._client.audio.transcriptions = fake  # type: ignore[assignment]
    result = await svc._transcribe(b"RIFF....")
    assert result.text == "γεια σας"
    assert "language" not in fake.kwargs
    assert fake.kwargs["model"] == "whisper-1"


# -- TTS ------------------------------------------------------------------------


def test_elevenlabs_tts_default_and_custom_voice() -> None:
    svc = build_tts(choice("elevenlabs"), "el", ENV)
    assert isinstance(svc, ElevenLabsTTSService)
    assert svc._settings.voice == DEFAULT_ELEVENLABS_VOICE
    assert svc._settings.model == "eleven_flash_v2_5"
    svc = build_tts(choice("elevenlabs", voice="abcDEF123"), "auto", ENV)
    assert svc._settings.voice == "abcDEF123"
    assert svc._settings.language is None


def test_cartesia_tts_language() -> None:
    svc = build_tts(choice("cartesia"), "el", ENV)
    assert isinstance(svc, CartesiaTTSService)
    assert svc._settings.voice == DEFAULT_CARTESIA_VOICE
    assert str(svc._settings.language) == "el"
    assert build_tts(choice("cartesia"), "auto", ENV)._settings.language is None


def test_deepgram_tts_voice_is_the_aura_model() -> None:
    svc = build_tts(choice("deepgram", voice="aura-2-thalia-en"), "en", ENV)
    assert isinstance(svc, DeepgramTTSService)
    assert svc._settings.voice == "aura-2-thalia-en"


def test_local_tts_uses_its_url_model_and_voice() -> None:
    env = {"HN_TTS_URL": "http://hn-tts:8000", "HN_TTS_MODEL": "chatterbox", "HN_TTS_VOICE": "el-1"}
    picked = select_tts(TTSSettings(), env)
    assert picked is not None
    svc = build_tts(picked, "el", env)
    assert isinstance(svc, OpenAITTSService)
    assert (svc._settings.model, svc._settings.voice) == ("chatterbox", "el-1")
    assert client_of(svc).api_key == LOCAL_PLACEHOLDER_KEY
    assert str(client_of(svc).base_url).rstrip("/") == "http://hn-tts:8000/v1"


def test_openai_tts_never_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-process-env")
    svc = build_tts(choice("local", key_env=None, base_url="http://hn-tts:8000/v1"), "el", {})
    assert client_of(svc).api_key == LOCAL_PLACEHOLDER_KEY
    assert str(client_of(svc).base_url).rstrip("/") == "http://hn-tts:8000/v1"


# -- turns ----------------------------------------------------------------------


def test_turn_end_uses_smart_turn_only_for_english() -> None:
    assert isinstance(turn_strategies("en").stop[0], TurnAnalyzerUserTurnStopStrategy)  # type: ignore[index]
    for language in ("el", "auto"):
        stop = turn_strategies(language).stop  # type: ignore[arg-type]
        assert stop is not None
        assert isinstance(stop[0], SpeechTimeoutUserTurnStopStrategy)
