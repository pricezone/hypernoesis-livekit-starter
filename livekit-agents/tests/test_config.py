from hn_agent.config import (
    DEFAULT_CONFIG,
    MAX_PROMPT_CHARS,
    AgentConfig,
    DisclosureSettings,
    LLMSettings,
    STTSettings,
    TTSSettings,
    normalize_base_url,
    parse_agent_config,
)

FULL = {
    "prompt": "You are the front desk of a dental clinic.\nBe brief.",
    "greeting": "Καλώς ήρθατε στο ιατρείο.",
    "language": "el",
    "llm": {"provider": "openai", "model": "gpt-4.1-mini", "baseUrl": "https://api.openai.com/v1/"},
    "stt": {"provider": "deepgram", "model": "nova-3"},
    "tts": {"provider": "elevenlabs", "voiceId": "abcDEF123", "model": "eleven_flash_v2_5"},
    "noiseCancellation": True,
    "aiDisclosure": {"enabled": True, "text": "Είμαι ψηφιακός βοηθός της κλινικής."},
}


def test_full_document_parses() -> None:
    result = parse_agent_config(FULL)
    assert result.invalid == []
    assert result.config == AgentConfig(
        prompt="You are the front desk of a dental clinic.\nBe brief.",
        greeting="Καλώς ήρθατε στο ιατρείο.",
        language="el",
        llm=LLMSettings("openai", "gpt-4.1-mini", "https://api.openai.com/v1"),
        stt=STTSettings("deepgram", "nova-3"),
        tts=TTSSettings("elevenlabs", "abcDEF123", "eleven_flash_v2_5"),
        noise_cancellation=True,
        ai_disclosure=DisclosureSettings(True, "Είμαι ψηφιακός βοηθός της κλινικής."),
    )


def test_empty_document_is_the_defaults() -> None:
    result = parse_agent_config({})
    assert result.invalid == []
    assert result.config == DEFAULT_CONFIG


def test_defaults_are_bilingual_with_disclosure_on() -> None:
    assert DEFAULT_CONFIG.language == "auto"
    assert DEFAULT_CONFIG.ai_disclosure.enabled is True
    assert DEFAULT_CONFIG.noise_cancellation is False
    assert DEFAULT_CONFIG.llm.provider == "auto"


def test_non_object_falls_back_to_defaults() -> None:
    for value in (None, [], "x", 3):
        result = parse_agent_config(value)
        assert result.config == DEFAULT_CONFIG
        assert result.invalid == ["$"]


def test_invalid_fields_fall_back_one_by_one() -> None:
    result = parse_agent_config(
        {
            "prompt": 42,
            "language": "fr",
            "llm": {"provider": "mystery", "model": "bad model id"},
            "stt": "deepgram",
            "tts": {"voiceId": "x" * 300},
            "noiseCancellation": "yes",
            "aiDisclosure": {"enabled": "no"},
            "greeting": "Γεια!",
        }
    )
    assert sorted(result.invalid) == [
        "aiDisclosure.enabled",
        "language",
        "llm.model",
        "llm.provider",
        "noiseCancellation",
        "prompt",
        "stt",
        "tts.voiceId",
    ]
    config = result.config
    assert config.prompt is None
    assert config.language == "auto"
    assert config.llm == LLMSettings()
    assert config.stt == STTSettings()
    assert config.tts.voice_id is None
    assert config.noise_cancellation is False
    assert config.ai_disclosure.enabled is True
    # valid fields next to invalid ones are kept
    assert config.greeting == "Γεια!"


def test_unknown_fields_are_ignored() -> None:
    result = parse_agent_config({"language": "en", "somethingNew": {"a": 1}})
    assert result.invalid == []
    assert result.config.language == "en"


def test_case_and_whitespace_are_forgiven() -> None:
    result = parse_agent_config({"language": " EN ", "tts": {"provider": "ElevenLabs"}})
    assert result.invalid == []
    assert result.config.language == "en"
    assert result.config.tts.provider == "elevenlabs"


def test_disclosure_can_be_turned_off() -> None:
    result = parse_agent_config({"aiDisclosure": {"enabled": False}})
    assert result.config.ai_disclosure == DisclosureSettings(enabled=False, text=None)


def test_empty_greeting_means_silence_but_empty_prompt_means_default() -> None:
    config = parse_agent_config({"greeting": "  ", "prompt": ""}).config
    assert config.greeting == ""
    assert config.prompt is None


def test_greeting_is_one_line_and_control_characters_are_removed() -> None:
    config = parse_agent_config({"greeting": "Hello\n\nthere\x07!"}).config
    assert config.greeting == "Hello there!"


def test_overlong_prompt_is_truncated_and_reported() -> None:
    result = parse_agent_config({"prompt": "a" * (MAX_PROMPT_CHARS + 5)})
    assert result.invalid == ["prompt"]
    assert result.config.prompt is not None
    assert len(result.config.prompt) == MAX_PROMPT_CHARS


def test_null_values_are_treated_as_missing() -> None:
    result = parse_agent_config(
        {"prompt": None, "llm": None, "tts": {"voiceId": None}, "aiDisclosure": None}
    )
    assert result.invalid == []
    assert result.config == DEFAULT_CONFIG


def test_model_ids_accept_real_world_shapes() -> None:
    for model in ("gpt-4.1-mini", "claude-haiku-4-5", "openai/gpt-4.1", "models/gemini:flash"):
        assert parse_agent_config({"llm": {"model": model}}).config.llm.model == model


def test_base_url_rules() -> None:
    assert normalize_base_url("https://Example.com/v1/") == "https://example.com/v1"
    assert normalize_base_url("http://10.0.0.2:8000") == "http://10.0.0.2:8000"
    assert normalize_base_url("ftp://example.com") is None
    assert normalize_base_url("https://user:pw@example.com/v1") is None
    assert normalize_base_url("https://example.com/v1?key=1") is None
    assert normalize_base_url("not a url") is None
    result = parse_agent_config({"llm": {"baseUrl": "https://u:p@evil.example/v1"}})
    assert result.invalid == ["llm.baseUrl"]
    assert result.config.llm.base_url is None
