"""Agent settings: the JSON contract, the built-in defaults and the parser.

The settings are not secrets. HyperNoesis serves them from the subscriber's
`/livekit` page; provider keys never travel in this document (they stay in the
environment of the subscriber's own server).

Contract (`GET ${HN_AGENT_CONFIG_URL}`, `Authorization: Bearer ${HN_AGENT_TOKEN}`)::

    {
      "prompt": "string",
      "greeting": "string",
      "language": "el" | "en" | "auto",
      "llm": {"provider": "string", "model": "string", "baseUrl": "string"},
      "stt": {"provider": "string", "model": "string"},
      "tts": {"provider": "string", "voiceId": "string", "model": "string"},
      "noiseCancellation": true,
      "aiDisclosure": {"enabled": true, "text": "string"}
    }

Every field is optional. A missing or invalid field falls back to its built-in
default; unknown fields are ignored. See README.md for the full rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final, Literal
from urllib.parse import urlsplit

Language = Literal["el", "en", "auto"]

LANGUAGES: Final[tuple[str, ...]] = ("el", "en", "auto")
LLM_PROVIDERS: Final[tuple[str, ...]] = ("auto", "openai", "anthropic", "google", "hypernoesis")
STT_PROVIDERS: Final[tuple[str, ...]] = (
    "auto",
    "deepgram",
    "elevenlabs",
    "openai",
    "speechmatics",
    "hypernoesis",
    "local",
)
TTS_PROVIDERS: Final[tuple[str, ...]] = (
    "auto",
    "elevenlabs",
    "cartesia",
    "openai",
    "deepgram",
    "hypernoesis",
    "local",
)

MAX_PROMPT_CHARS: Final = 20_000
MAX_GREETING_CHARS: Final = 1_000
MAX_DISCLOSURE_CHARS: Final = 500
MAX_ID_CHARS: Final = 200
MAX_URL_CHARS: Final = 2_000

# Model and voice ids: letters, digits and the punctuation real ids use
# (`gpt-4.1-mini`, `claude-haiku-4-5`, `openai/gpt-4.1`, `aura-2-thalia-en`, `models/x:y`).
_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")
_CONTROL_RE: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "auto"
    model: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class STTSettings:
    provider: str = "auto"
    model: str | None = None


@dataclass(frozen=True)
class TTSSettings:
    provider: str = "auto"
    voice_id: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class DisclosureSettings:
    enabled: bool = True
    text: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    """Parsed settings. `prompt`/`greeting` of None mean "use the built-in text"."""

    prompt: str | None = None
    greeting: str | None = None
    language: Language = "auto"
    llm: LLMSettings = field(default_factory=LLMSettings)
    stt: STTSettings = field(default_factory=STTSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    noise_cancellation: bool = False
    ai_disclosure: DisclosureSettings = field(default_factory=DisclosureSettings)


DEFAULT_CONFIG: Final = AgentConfig()


@dataclass
class ParseResult:
    config: AgentConfig
    # JSON paths of fields that were present but invalid (never their values).
    invalid: list[str] = field(default_factory=list)


def parse_agent_config(data: Any) -> ParseResult:
    """Parse the agent-config JSON leniently: bad fields fall back to defaults."""
    invalid: list[str] = []
    if not isinstance(data, dict):
        return ParseResult(DEFAULT_CONFIG, ["$"])

    prompt = _text(data, "prompt", MAX_PROMPT_CHARS, invalid, allow_newlines=True)
    greeting = _text(data, "greeting", MAX_GREETING_CHARS, invalid, allow_empty=True)

    language: Language = DEFAULT_CONFIG.language
    if "language" in data:
        raw = data["language"]
        if isinstance(raw, str) and raw.strip().lower() in LANGUAGES:
            language = raw.strip().lower()  # type: ignore[assignment]
        else:
            invalid.append("language")

    llm_raw = _section(data, "llm", invalid)
    llm = LLMSettings(
        provider=_provider(llm_raw, "llm", LLM_PROVIDERS, invalid),
        model=_ident(llm_raw, "llm", "model", invalid),
        base_url=_url(llm_raw, "llm", "baseUrl", invalid),
    )
    stt_raw = _section(data, "stt", invalid)
    stt = STTSettings(
        provider=_provider(stt_raw, "stt", STT_PROVIDERS, invalid),
        model=_ident(stt_raw, "stt", "model", invalid),
    )
    tts_raw = _section(data, "tts", invalid)
    tts = TTSSettings(
        provider=_provider(tts_raw, "tts", TTS_PROVIDERS, invalid),
        voice_id=_ident(tts_raw, "tts", "voiceId", invalid),
        model=_ident(tts_raw, "tts", "model", invalid),
    )

    noise_cancellation = DEFAULT_CONFIG.noise_cancellation
    if "noiseCancellation" in data:
        if isinstance(data["noiseCancellation"], bool):
            noise_cancellation = data["noiseCancellation"]
        else:
            invalid.append("noiseCancellation")

    disclosure_raw = _section(data, "aiDisclosure", invalid)
    disclosure_enabled = DisclosureSettings.enabled
    if "enabled" in disclosure_raw:
        if isinstance(disclosure_raw["enabled"], bool):
            disclosure_enabled = disclosure_raw["enabled"]
        else:
            invalid.append("aiDisclosure.enabled")
    disclosure_text = _text(
        disclosure_raw, "text", MAX_DISCLOSURE_CHARS, invalid, path="aiDisclosure.text"
    )

    config = AgentConfig(
        prompt=prompt,
        greeting=greeting,
        language=language,
        llm=llm,
        stt=stt,
        tts=tts,
        noise_cancellation=noise_cancellation,
        ai_disclosure=DisclosureSettings(enabled=disclosure_enabled, text=disclosure_text),
    )
    return ParseResult(config, invalid)


def _section(data: dict[str, Any], key: str, invalid: list[str]) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        invalid.append(key)
        return {}
    return value


def _text(
    data: dict[str, Any],
    key: str,
    limit: int,
    invalid: list[str],
    *,
    path: str | None = None,
    allow_newlines: bool = False,
    allow_empty: bool = False,
) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str):
        invalid.append(path or key)
        return None
    if not allow_newlines:
        value = " ".join(value.split())
    value = _CONTROL_RE.sub("", value).strip()
    if not value:
        # An explicit empty greeting means "say nothing after the disclosure".
        return "" if allow_empty else None
    if len(value) > limit:
        invalid.append(path or key)
        return value[:limit]
    return value


def _provider(
    section: dict[str, Any], name: str, allowed: tuple[str, ...], invalid: list[str]
) -> str:
    if "provider" not in section or section["provider"] is None:
        return "auto"
    value = section["provider"]
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    invalid.append(f"{name}.provider")
    return "auto"


def _ident(section: dict[str, Any], name: str, key: str, invalid: list[str]) -> str | None:
    if key not in section or section[key] is None:
        return None
    value = section[key]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if len(value) <= MAX_ID_CHARS and _ID_RE.match(value):
            return value
    invalid.append(f"{name}.{key}")
    return None


def _url(section: dict[str, Any], name: str, key: str, invalid: list[str]) -> str | None:
    if key not in section or section[key] is None:
        return None
    value = section[key]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        normalized = normalize_base_url(value)
        if normalized is not None and len(value) <= MAX_URL_CHARS:
            return normalized
    invalid.append(f"{name}.{key}")
    return None


def normalize_base_url(value: str) -> str | None:
    """Return `scheme://host[:port]/path` without a trailing slash, or None."""
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    if parts.username or parts.password or parts.query or parts.fragment:
        return None
    path = parts.path.rstrip("/")
    return f"{parts.scheme}://{parts.netloc.lower()}{path}"
