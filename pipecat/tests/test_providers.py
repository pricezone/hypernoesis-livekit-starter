import pytest

from hn_pipecat.config import LLMSettings, STTSettings, TTSSettings
from hn_pipecat.providers import (
    OPENAI_OFFICIAL_BASE_URL,
    base_url_mismatch,
    describe,
    local_service_base_url,
    select_llm,
    select_stt,
    select_tts,
)

SECRET = "sk-very-secret-value"
GATEWAY = {
    "HN_GATEWAY_URL": "https://gateway.example/gateway/v1",
    "HN_GATEWAY_TOKEN": SECRET,
}
LOCAL = {"HN_STT_URL": "http://hn-stt:8000/v1", "HN_TTS_URL": "http://hn-tts:8000"}


# -- LLM ----------------------------------------------------------------------


def test_llm_none_without_keys() -> None:
    assert select_llm(LLMSettings(), {}) is None
    assert select_llm(LLMSettings(), LOCAL) is None  # there is no free local LLM


def test_llm_auto_order_prefers_own_keys() -> None:
    env = {"ANTHROPIC_API_KEY": SECRET, "GOOGLE_API_KEY": SECRET, **GATEWAY}
    choice = select_llm(LLMSettings(), env)
    assert choice is not None
    assert (choice.provider, choice.model, choice.key_env) == (
        "anthropic",
        "claude-haiku-4-5",
        "ANTHROPIC_API_KEY",
    )
    env["OPENAI_API_KEY"] = SECRET
    choice = select_llm(LLMSettings(), env)
    assert choice is not None
    assert (choice.provider, choice.model) == ("openai", "gpt-4.1-mini")
    assert choice.base_url == OPENAI_OFFICIAL_BASE_URL


def test_llm_openai_compatible_base_url() -> None:
    env = {"OPENAI_API_KEY": SECRET, "OPENAI_BASE_URL": "https://llm.example/v1/"}
    choice = select_llm(LLMSettings(provider="openai", model="llama-3.3-70b"), env)
    assert choice is not None
    assert choice.base_url == "https://llm.example/v1"
    assert choice.model == "llama-3.3-70b"
    assert choice.reason == "configured"


def test_llm_settings_base_url_cannot_redirect_a_key() -> None:
    env = {"OPENAI_API_KEY": SECRET}
    settings = LLMSettings(provider="openai", base_url="https://attacker.example/v1")
    choice = select_llm(settings, env)
    assert choice is not None
    assert choice.base_url == OPENAI_OFFICIAL_BASE_URL
    assert base_url_mismatch(settings, choice)
    same = LLMSettings(provider="openai", base_url=OPENAI_OFFICIAL_BASE_URL)
    assert not base_url_mismatch(same, choice)


def test_llm_configured_provider_falls_back_when_missing() -> None:
    choice = select_llm(
        LLMSettings(provider="anthropic", model="claude-x"), {"OPENAI_API_KEY": SECRET}
    )
    assert choice is not None
    assert choice.provider == "openai"
    assert choice.reason == "fallback"
    # the model was meant for Anthropic, so the OpenAI default is used
    assert choice.model == "gpt-4.1-mini"


def test_llm_auto_applies_the_configured_model() -> None:
    choice = select_llm(LLMSettings(model="gpt-4.1"), {"OPENAI_API_KEY": SECRET})
    assert choice is not None
    assert choice.model == "gpt-4.1"


def test_llm_google_accepts_gemini_key_name() -> None:
    choice = select_llm(LLMSettings(), {"GEMINI_API_KEY": SECRET})
    assert choice is not None
    assert (choice.provider, choice.key_env, choice.model) == ("google", "GEMINI_API_KEY", None)


def test_llm_hypernoesis_gateway() -> None:
    choice = select_llm(LLMSettings(provider="hypernoesis", model="openai/gpt-4.1-mini"), GATEWAY)
    assert choice is not None
    assert choice.provider == "hypernoesis"
    assert choice.base_url == "https://gateway.example/gateway/v1"
    assert choice.key_env == "HN_GATEWAY_TOKEN"
    assert choice.model == "openai/gpt-4.1-mini"


def test_gateway_token_needs_https_for_public_hosts() -> None:
    env = {"HN_GATEWAY_URL": "http://gateway.example/v1", "HN_GATEWAY_TOKEN": SECRET}
    assert select_llm(LLMSettings(), env) is None
    env["HN_GATEWAY_URL"] = "http://127.0.0.1:3000/v1"
    assert select_llm(LLMSettings(), env) is not None


def test_gateway_needs_both_url_and_token() -> None:
    assert select_llm(LLMSettings(), {"HN_GATEWAY_URL": GATEWAY["HN_GATEWAY_URL"]}) is None


# -- STT ----------------------------------------------------------------------


def test_stt_free_local_service_when_no_key() -> None:
    choice = select_stt(STTSettings(), LOCAL)
    assert choice is not None
    assert (choice.provider, choice.key_env, choice.model) == ("local", None, "whisper-1")
    assert choice.base_url == "http://hn-stt:8000/v1"


def test_stt_local_model_from_env() -> None:
    choice = select_stt(STTSettings(), {**LOCAL, "HN_STT_MODEL": "large-v3-turbo"})
    assert choice is not None
    assert choice.model == "large-v3-turbo"


def test_stt_auto_order() -> None:
    env = {**LOCAL, **GATEWAY, "SPEECHMATICS_API_KEY": SECRET}
    order = []
    for key in ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "DEEPGRAM_API_KEY"):
        choice = select_stt(STTSettings(), env)
        assert choice is not None
        order.append(choice.provider)
        env[key] = SECRET
    choice = select_stt(STTSettings(), env)
    assert choice is not None
    order.append(choice.provider)
    assert order == ["speechmatics", "openai", "elevenlabs", "deepgram"]

    only_gateway = select_stt(STTSettings(), {**LOCAL, **GATEWAY})
    assert only_gateway is not None
    assert only_gateway.provider == "hypernoesis"


def test_stt_skips_openai_compatible_endpoint_in_auto() -> None:
    env = {"OPENAI_API_KEY": SECRET, "OPENAI_BASE_URL": "https://llm.example/v1", **LOCAL}
    choice = select_stt(STTSettings(), env)
    assert choice is not None
    assert choice.provider == "local"
    # chosen explicitly, it uses the endpoint the key belongs to
    explicit = select_stt(STTSettings(provider="openai"), env)
    assert explicit is not None
    assert (explicit.provider, explicit.base_url) == ("openai", "https://llm.example/v1")


def test_stt_configured_deepgram() -> None:
    env = {"DEEPGRAM_API_KEY": SECRET, "ELEVENLABS_API_KEY": SECRET}
    choice = select_stt(STTSettings(provider="elevenlabs", model="scribe_v2_realtime"), env)
    assert choice is not None
    assert (choice.provider, choice.model, choice.reason) == (
        "elevenlabs",
        "scribe_v2_realtime",
        "configured",
    )


# -- TTS ----------------------------------------------------------------------


def test_tts_free_local_service_when_no_key() -> None:
    choice = select_tts(TTSSettings(), {**LOCAL, "HN_TTS_VOICE": "greek-female"})
    assert choice is not None
    assert (choice.provider, choice.base_url, choice.model, choice.voice) == (
        "local",
        "http://hn-tts:8000/v1",
        "tts-1",
        "greek-female",
    )


def test_tts_auto_order() -> None:
    env = {**LOCAL, **GATEWAY}
    order = []
    for key in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "CARTESIA_API_KEY", "ELEVENLABS_API_KEY"):
        env[key] = SECRET
        choice = select_tts(TTSSettings(), env)
        assert choice is not None
        order.append(choice.provider)
    assert order == ["deepgram", "openai", "cartesia", "elevenlabs"]


def test_tts_voice_and_model_only_for_their_provider() -> None:
    env = {"ELEVENLABS_API_KEY": SECRET, "CARTESIA_API_KEY": SECRET}
    choice = select_tts(TTSSettings(provider="elevenlabs", voice_id="voice123"), env)
    assert choice is not None
    assert (choice.provider, choice.voice, choice.model) == (
        "elevenlabs",
        "voice123",
        "eleven_flash_v2_5",
    )
    fallback = select_tts(TTSSettings(provider="openai", voice_id="nova"), env)
    assert fallback is not None
    assert (fallback.provider, fallback.voice, fallback.reason) == ("elevenlabs", None, "fallback")


def test_tts_none_without_anything() -> None:
    assert select_tts(TTSSettings(), {}) is None
    assert select_stt(STTSettings(), {}) is None


# -- helpers ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "suffix", "expected"),
    [
        ("http://hn-stt:8000/v1", "/audio/transcriptions", "http://hn-stt:8000/v1"),
        ("http://hn-stt:8000/v1/", "/audio/transcriptions", "http://hn-stt:8000/v1"),
        ("http://hn-stt:8000", "/audio/transcriptions", "http://hn-stt:8000/v1"),
        (
            "http://hn-stt:8000/v1/audio/transcriptions",
            "/audio/transcriptions",
            "http://hn-stt:8000/v1",
        ),
        (
            "http://127.0.0.1:9000/api/v1/audio/speech",
            "/audio/speech",
            "http://127.0.0.1:9000/api/v1",
        ),
        ("not a url", "/audio/speech", None),
    ],
)
def test_local_service_base_url(value: str, suffix: str, expected: str | None) -> None:
    assert local_service_base_url(value, suffix) == expected


def test_describe_never_contains_a_secret() -> None:
    env = {"OPENAI_API_KEY": SECRET, "ELEVENLABS_API_KEY": SECRET, **GATEWAY}
    for choice in (
        select_llm(LLMSettings(), env),
        select_stt(STTSettings(), env),
        select_tts(TTSSettings(voice_id="v1"), env),
    ):
        assert choice is not None
        assert SECRET not in describe("x", choice)
        assert SECRET not in repr(choice)
