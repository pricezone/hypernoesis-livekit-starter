"""Choose the LLM, STT and TTS providers from the keys present on this server.

Selection is pure (settings + environment in, a `Choice` out) so it can be
tested without any network. A `Choice` names the environment variable holding
the credential, never the credential itself, so it is safe to log.

Rules (README.md has the table):

* A provider is *available* when its key (or service URL) is set in the
  environment of this server.
* `provider: "auto"` picks the first available provider in the fixed order
  below: own keys first, then the admin-only HyperNoesis API, then the free
  local services that run next to LiveKit on the same server.
* A named provider that is not available falls back to `auto` (logged).
* Keys only ever go to the URL set next to them in the environment. A
  `baseUrl` from the settings document can select an endpoint that is already
  configured here, but it can never redirect a key somewhere else.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from .config import LLMSettings, STTSettings, TTSSettings, normalize_base_url
from .remote import is_private_host

OPENAI_OFFICIAL_BASE_URL: Final = "https://api.openai.com/v1"

LLM_AUTO_ORDER: Final = ("openai", "anthropic", "google", "hypernoesis")
STT_AUTO_ORDER: Final = ("deepgram", "elevenlabs", "openai", "speechmatics", "hypernoesis", "local")
TTS_AUTO_ORDER: Final = ("elevenlabs", "cartesia", "openai", "deepgram", "hypernoesis", "local")

# Fast models for a voice conversation. None = the LiveKit plugin's own default.
DEFAULT_LLM_MODELS: Final[dict[str, str | None]] = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5",
    "google": None,
    "hypernoesis": "gpt-4.1-mini",
}
DEFAULT_STT_MODELS: Final[dict[str, str | None]] = {
    "deepgram": "nova-3",
    "elevenlabs": None,
    "openai": "gpt-4o-mini-transcribe",
    "speechmatics": None,
    "hypernoesis": "gpt-4o-mini-transcribe",
    "local": "whisper-1",
}
DEFAULT_TTS_MODELS: Final[dict[str, str | None]] = {
    "elevenlabs": "eleven_flash_v2_5",
    "cartesia": None,
    "openai": "gpt-4o-mini-tts",
    "deepgram": None,
    "hypernoesis": "tts-1",
    "local": "tts-1",
}

LOCAL_PLACEHOLDER_KEY: Final = "hn-local"  # the free local services need no key


@dataclass(frozen=True)
class Choice:
    provider: str
    model: str | None
    base_url: str | None
    key_env: str | None  # name of the env var with the credential; None = no key
    voice: str | None = None
    # "configured" = the settings asked for it; "auto" = picked by key presence;
    # "fallback" = the settings asked for another provider that is not available here.
    reason: str = "auto"
    requested: str | None = None


@dataclass(frozen=True)
class _Endpoint:
    key_env: str | None
    base_url: str | None


def _env(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return value.strip() if value and value.strip() else None


def _openai_endpoint(env: Mapping[str, str], *, official_only: bool) -> _Endpoint | None:
    if not _env(env, "OPENAI_API_KEY"):
        return None
    custom = _env(env, "OPENAI_BASE_URL")
    if custom:
        base = normalize_base_url(custom)
        if base is None or (official_only and base != OPENAI_OFFICIAL_BASE_URL):
            return None
        return _Endpoint("OPENAI_API_KEY", base)
    return _Endpoint("OPENAI_API_KEY", OPENAI_OFFICIAL_BASE_URL)


def _gateway_endpoint(env: Mapping[str, str]) -> _Endpoint | None:
    url, token = _env(env, "HN_GATEWAY_URL"), _env(env, "HN_GATEWAY_TOKEN")
    if not url or not token:
        return None
    base = normalize_base_url(url)
    if base is None:
        return None
    parts = urlsplit(base)
    if parts.scheme != "https" and not is_private_host(parts.hostname or ""):
        return None  # never send the gateway token over plain http to a public host
    return _Endpoint("HN_GATEWAY_TOKEN", base)


def local_service_base_url(value: str, endpoint_suffix: str) -> str | None:
    """`http://host:port[/v1][/audio/...]` -> the OpenAI-compatible base URL."""
    base = normalize_base_url(value)
    if base is None:
        return None
    parts = urlsplit(base)
    path = parts.path
    if path.endswith(endpoint_suffix):
        path = path[: -len(endpoint_suffix)]
    path = path.rstrip("/") or "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _local_endpoint(env: Mapping[str, str], var: str, suffix: str) -> _Endpoint | None:
    value = _env(env, var)
    if not value:
        return None
    base = local_service_base_url(value, suffix)
    return _Endpoint(None, base) if base else None


def _google_endpoint(env: Mapping[str, str]) -> _Endpoint | None:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if _env(env, name):
            return _Endpoint(name, None)
    return None


def _key_endpoint(env: Mapping[str, str], name: str) -> _Endpoint | None:
    return _Endpoint(name, None) if _env(env, name) else None


def llm_endpoints(env: Mapping[str, str]) -> dict[str, _Endpoint | None]:
    return {
        "openai": _openai_endpoint(env, official_only=False),
        "anthropic": _key_endpoint(env, "ANTHROPIC_API_KEY"),
        "google": _google_endpoint(env),
        "hypernoesis": _gateway_endpoint(env),
    }


def stt_endpoints(env: Mapping[str, str], *, requested: str) -> dict[str, _Endpoint | None]:
    return {
        "deepgram": _key_endpoint(env, "DEEPGRAM_API_KEY"),
        "elevenlabs": _key_endpoint(env, "ELEVENLABS_API_KEY"),
        # In auto mode only official OpenAI: an OpenAI-compatible LLM endpoint
        # (OPENAI_BASE_URL) usually has no speech API.
        "openai": _openai_endpoint(env, official_only=requested != "openai"),
        "speechmatics": _key_endpoint(env, "SPEECHMATICS_API_KEY"),
        "hypernoesis": _gateway_endpoint(env),
        "local": _local_endpoint(env, "HN_STT_URL", "/audio/transcriptions"),
    }


def tts_endpoints(env: Mapping[str, str], *, requested: str) -> dict[str, _Endpoint | None]:
    return {
        "elevenlabs": _key_endpoint(env, "ELEVENLABS_API_KEY"),
        "cartesia": _key_endpoint(env, "CARTESIA_API_KEY"),
        "openai": _openai_endpoint(env, official_only=requested != "openai"),
        "deepgram": _key_endpoint(env, "DEEPGRAM_API_KEY"),
        "hypernoesis": _gateway_endpoint(env),
        "local": _local_endpoint(env, "HN_TTS_URL", "/audio/speech"),
    }


def _pick(
    requested: str, order: tuple[str, ...], endpoints: Mapping[str, _Endpoint | None]
) -> tuple[str, _Endpoint, str] | None:
    if requested != "auto":
        endpoint = endpoints.get(requested)
        if endpoint is not None:
            return requested, endpoint, "configured"
    for name in order:
        endpoint = endpoints.get(name)
        if endpoint is not None:
            return name, endpoint, "auto" if requested == "auto" else "fallback"
    return None


def _model_for(requested: str, chosen: str, configured: str | None) -> str | None:
    # A configured model only makes sense for the provider it was chosen for.
    return configured if requested in ("auto", chosen) else None


def select_llm(settings: LLMSettings, env: Mapping[str, str]) -> Choice | None:
    endpoints = llm_endpoints(env)
    picked = _pick(settings.provider, LLM_AUTO_ORDER, endpoints)
    if picked is None:
        return None
    name, endpoint, reason = picked
    model = _model_for(settings.provider, name, settings.model) or DEFAULT_LLM_MODELS[name]
    return Choice(
        provider=name,
        model=model,
        base_url=endpoint.base_url,
        key_env=endpoint.key_env,
        reason=reason,
        requested=settings.provider,
    )


def base_url_mismatch(settings: LLMSettings, choice: Choice) -> bool:
    """True when the settings asked for a base URL this server is not configured for.

    The configured endpoint is used anyway: a key is only sent to the URL set
    next to it in this server's environment.
    """
    if settings.base_url is None:
        return False
    return settings.base_url != choice.base_url


def select_stt(settings: STTSettings, env: Mapping[str, str]) -> Choice | None:
    endpoints = stt_endpoints(env, requested=settings.provider)
    picked = _pick(settings.provider, STT_AUTO_ORDER, endpoints)
    if picked is None:
        return None
    name, endpoint, reason = picked
    model = _model_for(settings.provider, name, settings.model)
    if model is None and name == "local":
        model = _env(env, "HN_STT_MODEL")
    return Choice(
        provider=name,
        model=model or DEFAULT_STT_MODELS[name],
        base_url=endpoint.base_url,
        key_env=endpoint.key_env,
        reason=reason,
        requested=settings.provider,
    )


def select_tts(settings: TTSSettings, env: Mapping[str, str]) -> Choice | None:
    endpoints = tts_endpoints(env, requested=settings.provider)
    picked = _pick(settings.provider, TTS_AUTO_ORDER, endpoints)
    if picked is None:
        return None
    name, endpoint, reason = picked
    model = _model_for(settings.provider, name, settings.model)
    voice = _model_for(settings.provider, name, settings.voice_id)
    if name == "local":
        model = model or _env(env, "HN_TTS_MODEL")
        voice = voice or _env(env, "HN_TTS_VOICE")
    return Choice(
        provider=name,
        model=model or DEFAULT_TTS_MODELS[name],
        base_url=endpoint.base_url,
        key_env=endpoint.key_env,
        voice=voice,
        reason=reason,
        requested=settings.provider,
    )


def missing_hint(kind: str) -> str:
    hints = {
        "llm": "set OPENAI_API_KEY (optionally OPENAI_BASE_URL), ANTHROPIC_API_KEY or GOOGLE_API_KEY",
        "stt": "set DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, OPENAI_API_KEY, SPEECHMATICS_API_KEY "
        "or HN_STT_URL (free local speech recognition)",
        "tts": "set ELEVENLABS_API_KEY, CARTESIA_API_KEY, OPENAI_API_KEY, DEEPGRAM_API_KEY "
        "or HN_TTS_URL (free local speech synthesis)",
    }
    return hints[kind]


def describe(kind: str, choice: Choice) -> str:
    """One log line without any secret: provider, model, voice and why."""
    parts = [f"{kind}={choice.provider}"]
    if choice.model:
        parts.append(f"model={choice.model}")
    if choice.voice:
        parts.append(f"voice={choice.voice}")
    if choice.reason == "fallback":
        parts.append(f"(requested {choice.requested} is not configured on this server)")
    return " ".join(parts)
