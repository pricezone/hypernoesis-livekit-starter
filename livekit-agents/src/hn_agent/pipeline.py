"""Turn provider choices into LiveKit Agents plugin instances.

Every plugin gets its key and base URL passed explicitly, so the OpenAI SDK
never falls back to OPENAI_API_KEY / OPENAI_BASE_URL from the environment for
the local services or the HyperNoesis API.

LiveKit plugins register themselves when imported and must be imported on the
main thread, which is why they are imported here at module level.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from livekit.agents import LanguageCode, inference, llm, stt, tts
from livekit.agents.inference.eot.languages import LOCAL_LANGUAGES, ThresholdOptions
from livekit.agents.voice.turn import TurnDetectionMode
from livekit.plugins import anthropic, cartesia, deepgram, elevenlabs, google, openai, speechmatics

from .config import Language
from .providers import LOCAL_PLACEHOLDER_KEY, Choice

logger = logging.getLogger("hn_agent.pipeline")

# Providers whose streaming STT cannot detect Greek and English by itself:
# with language "auto" they listen for Greek (set "en" for English callers).
_AUTO_AS_GREEK_STT = frozenset({"deepgram", "speechmatics"})

TURN_DETECTOR_LANGUAGES = frozenset(LOCAL_LANGUAGES)


def _key(choice: Choice, env: Mapping[str, str]) -> str:
    if choice.key_env is None:
        return LOCAL_PLACEHOLDER_KEY
    return env[choice.key_env].strip()


def build_llm(choice: Choice, env: Mapping[str, str]) -> llm.LLM:
    key = _key(choice, env)
    if choice.provider == "anthropic":
        return anthropic.LLM(model=choice.model or "claude-haiku-4-5", api_key=key)
    if choice.provider == "google":
        kwargs: dict[str, Any] = {"api_key": key}
        if choice.model:
            kwargs["model"] = choice.model
        return google.LLM(**kwargs)
    # openai, any OpenAI-compatible base URL, and the HyperNoesis API
    return openai.LLM(model=choice.model or "gpt-4.1-mini", api_key=key, base_url=choice.base_url)


def stt_language(provider: str, language: Language) -> str | None:
    """The language code to send, or None to let the provider detect it."""
    if language != "auto":
        return language
    return "el" if provider in _AUTO_AS_GREEK_STT else None


def build_stt(choice: Choice, language: Language, env: Mapping[str, str], vad: Any) -> stt.STT:
    key = _key(choice, env)
    lang = stt_language(choice.provider, language)
    engine: stt.STT
    if choice.provider == "deepgram":
        engine = deepgram.STT(model=choice.model or "nova-3", language=lang or "el", api_key=key)
    elif choice.provider == "elevenlabs":
        kwargs: dict[str, Any] = {"api_key": key}
        if lang:
            kwargs["language_code"] = lang
        if choice.model:
            kwargs["model_id"] = choice.model
        engine = elevenlabs.STT(**kwargs)
    elif choice.provider == "speechmatics":
        kwargs = {"api_key": key, "language": lang or "el"}
        if choice.model:
            kwargs["model"] = choice.model
        engine = speechmatics.STT(**kwargs)
    else:
        # openai, the HyperNoesis API and the free local service: the
        # OpenAI-compatible /v1/audio/transcriptions endpoint.
        engine = openai.STT(
            model=choice.model or "whisper-1",
            language=lang or "en",
            detect_language=lang is None,
            api_key=key,
            base_url=choice.base_url,
            use_realtime=False,
        )
    if not engine.capabilities.streaming:
        # File-based transcription: the VAD cuts the audio into utterances.
        engine = stt.StreamAdapter(stt=engine, vad=vad)
    return engine


def build_tts(choice: Choice, language: Language, env: Mapping[str, str]) -> tts.TTS:
    key = _key(choice, env)
    lang = None if language == "auto" else language
    if choice.provider == "elevenlabs":
        kwargs: dict[str, Any] = {"api_key": key, "model": choice.model or "eleven_flash_v2_5"}
        if choice.voice:
            kwargs["voice_id"] = choice.voice
        if lang:
            kwargs["language"] = lang
        return elevenlabs.TTS(**kwargs)
    if choice.provider == "cartesia":
        kwargs = {"api_key": key, "language": lang}
        if choice.model:
            kwargs["model"] = choice.model
        if choice.voice:
            kwargs["voice"] = choice.voice
        return cartesia.TTS(**kwargs)
    if choice.provider == "deepgram":
        # Aura voices are models ("aura-2-thalia-en"); there is no Greek Aura voice.
        model = choice.voice or choice.model
        if language == "el" and not model:
            logger.warning("Deepgram Aura has no Greek voice: Greek replies will sound wrong")
        return deepgram.TTS(model=model, api_key=key) if model else deepgram.TTS(api_key=key)
    # openai, the HyperNoesis API and the free local service: /v1/audio/speech
    kwargs = {"model": choice.model or "tts-1", "api_key": key, "base_url": choice.base_url}
    if choice.voice:
        kwargs["voice"] = choice.voice
    return openai.TTS(**kwargs)


class _CalibratedThresholds(ThresholdOptions):
    """Report unsupported for languages the local turn detector was not calibrated on.

    The stock options fall back to the English threshold for any language, so a
    Greek turn would be judged by an English-calibrated cut-off. Saying "not
    supported" makes LiveKit end that turn on VAD silence instead.
    """

    def supports(self, language: LanguageCode | None) -> bool:
        if language is not None and language.language not in TURN_DETECTOR_LANGUAGES:
            return False
        return super().supports(language)


class CalibratedTurnDetector(inference.TurnDetector):
    """LiveKit's local multilingual turn detector, limited to its calibrated languages."""

    def __init__(self) -> None:
        super().__init__(version="v1-mini")
        self._opts.thresholds = _CalibratedThresholds(self._opts.thresholds.model)


def turn_detection_for(language: Language) -> TurnDetectionMode:
    """Greek is not among the turn detector's languages: Greek turns end on VAD silence."""
    if language == "el":
        return "vad"
    return CalibratedTurnDetector()
