import pytest
from livekit.agents import LanguageCode, inference, stt
from livekit.plugins import anthropic, deepgram, elevenlabs, openai

from hn_agent.config import LLMSettings, STTSettings, TTSSettings
from hn_agent.pipeline import (
    TURN_DETECTOR_LANGUAGES,
    CalibratedTurnDetector,
    build_llm,
    build_stt,
    build_tts,
    stt_language,
    turn_detection_for,
)
from hn_agent.providers import LOCAL_PLACEHOLDER_KEY, select_llm, select_stt, select_tts

LOCAL = {"HN_STT_URL": "http://hn-stt:8000", "HN_TTS_URL": "http://hn-tts:8000/v1"}


@pytest.fixture
def leaky_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The OpenAI SDK reads these when nothing is passed; the builders must pass both.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://must-not-be-used.example/v1")


def test_greek_is_not_a_turn_detector_language() -> None:
    assert "el" not in TURN_DETECTOR_LANGUAGES
    assert "en" in TURN_DETECTOR_LANGUAGES


def test_turn_detection_per_language() -> None:
    assert turn_detection_for("el") == "vad"
    assert isinstance(turn_detection_for("en"), CalibratedTurnDetector)
    assert isinstance(turn_detection_for("auto"), CalibratedTurnDetector)


async def test_auto_turn_detector_hands_greek_turns_to_vad() -> None:
    detector = CalibratedTurnDetector()
    assert detector.model == "turn-detector-v1-mini"  # local, never LiveKit Cloud
    assert await detector.supports_language(LanguageCode("en"))
    assert await detector.supports_language(LanguageCode("de-DE"))
    assert not await detector.supports_language(LanguageCode("el"))
    assert not await detector.supports_language(LanguageCode("ell"))


@pytest.mark.parametrize(
    ("provider", "language", "expected"),
    [
        ("deepgram", "auto", "el"),
        ("speechmatics", "auto", "el"),
        ("elevenlabs", "auto", None),
        ("openai", "auto", None),
        ("local", "auto", None),
        ("deepgram", "en", "en"),
        ("local", "el", "el"),
    ],
)
def test_stt_language(provider: str, language: str, expected: str | None) -> None:
    assert stt_language(provider, language) == expected  # type: ignore[arg-type]


@pytest.mark.usefixtures("leaky_env")
def test_local_speech_services_get_explicit_url_and_no_real_key() -> None:
    vad = inference.VAD()
    stt_choice = select_stt(STTSettings(), LOCAL)
    assert stt_choice is not None
    engine = build_stt(stt_choice, "auto", LOCAL, vad)
    assert isinstance(engine, stt.StreamAdapter)
    inner = engine.wrapped_stt
    assert isinstance(inner, openai.STT)
    assert str(inner._client.base_url) == "http://hn-stt:8000/v1/"
    assert inner._client.api_key == LOCAL_PLACEHOLDER_KEY

    tts_choice = select_tts(TTSSettings(), LOCAL)
    assert tts_choice is not None
    voice = build_tts(tts_choice, "el", LOCAL)
    assert isinstance(voice, openai.TTS)
    assert str(voice._client.base_url) == "http://hn-tts:8000/v1/"
    assert voice._client.api_key == LOCAL_PLACEHOLDER_KEY


@pytest.mark.usefixtures("leaky_env")
def test_gateway_gets_its_own_token() -> None:
    env = {"HN_GATEWAY_URL": "https://gw.example/v1", "HN_GATEWAY_TOKEN": "hn_lk_token"}
    choice = select_llm(LLMSettings(), env)
    assert choice is not None
    model = build_llm(choice, env)
    assert isinstance(model, openai.LLM)
    assert str(model._client.base_url) == "https://gw.example/v1/"
    assert model._client.api_key == "hn_lk_token"


def test_provider_classes() -> None:
    env = {"ANTHROPIC_API_KEY": "a", "DEEPGRAM_API_KEY": "d", "ELEVENLABS_API_KEY": "e"}
    vad = inference.VAD()
    llm_choice = select_llm(LLMSettings(), env)
    stt_choice = select_stt(STTSettings(), env)
    tts_choice = select_tts(TTSSettings(voice_id="voice123"), env)
    assert llm_choice and stt_choice and tts_choice
    assert isinstance(build_llm(llm_choice, env), anthropic.LLM)
    assert isinstance(build_stt(stt_choice, "el", env, vad), deepgram.STT)
    assert isinstance(build_tts(tts_choice, "el", env), elevenlabs.TTS)
