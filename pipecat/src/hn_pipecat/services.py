"""Turn provider choices into Pipecat services, and pick the turn strategy.

Every service gets its key and base URL passed explicitly, so the OpenAI SDK
never falls back to OPENAI_API_KEY / OPENAI_BASE_URL from the environment for
the local services or the HyperNoesis API.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

from openai.types.audio import Transcription
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language as PipecatLanguage
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.types import NOT_GIVEN

from .config import Language
from .providers import LOCAL_PLACEHOLDER_KEY, Choice

logger = logging.getLogger("hn_pipecat.services")

# Providers whose streaming STT cannot detect Greek and English by itself:
# with language "auto" they listen for Greek (set "en" for English callers).
_AUTO_AS_GREEK_STT: Final = frozenset({"deepgram", "speechmatics"})

# Premade voices used when the settings name none (the same defaults as the
# LiveKit Agents plugins, so both starters sound alike).
DEFAULT_ELEVENLABS_VOICE: Final = "hpp4J3VqNfWAUOO0d1Us"
DEFAULT_CARTESIA_VOICE: Final = "f786b574-daa5-4673-aa0c-cbe3e8534c02"

# Seconds of silence that end a turn when no turn model is used (Greek).
SILENCE_TURN_SECONDS: Final = 0.5


def _key(choice: Choice, env: Mapping[str, str]) -> str:
    if choice.key_env is None:
        return LOCAL_PLACEHOLDER_KEY
    return env[choice.key_env].strip()


def _lang(code: str | None) -> Any:
    """A Pipecat language for a settings field, or NOT_GIVEN for "detect"."""
    return PipecatLanguage(code) if code else NOT_GIVEN


def build_llm(choice: Choice, env: Mapping[str, str], instructions: str) -> LLMService:
    key = _key(choice, env)
    if choice.provider == "anthropic":
        return AnthropicLLMService(
            api_key=key,
            settings=AnthropicLLMService.Settings(
                model=choice.model or "claude-haiku-4-5", system_instruction=instructions
            ),
        )
    if choice.provider == "google":
        return GoogleLLMService(
            api_key=key,
            settings=GoogleLLMService.Settings(
                model=choice.model or NOT_GIVEN, system_instruction=instructions
            ),
        )
    # openai, any OpenAI-compatible base URL, and the HyperNoesis API
    return OpenAILLMService(
        api_key=key,
        base_url=choice.base_url,
        settings=OpenAILLMService.Settings(
            model=choice.model or "gpt-4.1-mini", system_instruction=instructions
        ),
    )


def stt_language(provider: str, language: Language) -> str | None:
    """The language code to send, or None to let the provider detect it."""
    if language != "auto":
        return language
    return "el" if provider in _AUTO_AS_GREEK_STT else None


class DetectingOpenAISTTService(OpenAISTTService):
    """OpenAI-compatible transcription that lets the endpoint detect the language.

    Pipecat's `OpenAISTTService` always sends a language; with `language: auto`
    the request must leave it out so Greek and English are both recognised.
    """

    async def _transcribe(self, audio: bytes) -> Transcription:
        model = self._settings.model
        return await self._client.audio.transcriptions.create(
            file=("audio.wav", audio, "audio/wav"),
            model=model if isinstance(model, str) else "whisper-1",
        )


def build_stt(choice: Choice, language: Language, env: Mapping[str, str]) -> STTService:
    key = _key(choice, env)
    lang = stt_language(choice.provider, language)
    if choice.provider == "deepgram":
        return DeepgramSTTService(
            api_key=key,
            settings=DeepgramSTTService.Settings(
                model=choice.model or "nova-3", language=PipecatLanguage(lang or "el")
            ),
        )
    if choice.provider == "elevenlabs":
        return ElevenLabsRealtimeSTTService(
            api_key=key,
            settings=ElevenLabsRealtimeSTTService.Settings(
                model=choice.model or NOT_GIVEN, language=_lang(lang)
            ),
        )
    if choice.provider == "speechmatics":
        return SpeechmaticsSTTService(
            api_key=key,
            settings=SpeechmaticsSTTService.Settings(
                model=choice.model or NOT_GIVEN, language=PipecatLanguage(lang or "el")
            ),
        )
    # openai, the HyperNoesis API and the free local service: the
    # OpenAI-compatible /v1/audio/transcriptions endpoint, one request per
    # utterance cut by the VAD.
    model = choice.model or "whisper-1"
    if lang is None:
        return DetectingOpenAISTTService(
            api_key=key,
            base_url=choice.base_url,
            settings=OpenAISTTService.Settings(model=model),
        )
    return OpenAISTTService(
        api_key=key,
        base_url=choice.base_url,
        settings=OpenAISTTService.Settings(model=model, language=PipecatLanguage(lang)),
    )


def build_tts(choice: Choice, language: Language, env: Mapping[str, str]) -> TTSService:
    key = _key(choice, env)
    lang = None if language == "auto" else language
    if choice.provider == "elevenlabs":
        return ElevenLabsTTSService(
            api_key=key,
            settings=ElevenLabsTTSService.Settings(
                voice=choice.voice or DEFAULT_ELEVENLABS_VOICE,
                model=choice.model or "eleven_flash_v2_5",
                language=_lang(lang),
            ),
        )
    if choice.provider == "cartesia":
        # Without a language Cartesia's request leaves it out.
        return CartesiaTTSService(
            api_key=key,
            settings=CartesiaTTSService.Settings(
                voice=choice.voice or DEFAULT_CARTESIA_VOICE,
                model=choice.model or NOT_GIVEN,
                language=PipecatLanguage(lang) if lang else None,
            ),
        )
    if choice.provider == "deepgram":
        # Aura voices are models ("aura-2-thalia-en"); there is no Greek Aura voice.
        voice = choice.voice or choice.model
        if language == "el" and not voice:
            logger.warning("Deepgram Aura has no Greek voice: Greek replies will sound wrong")
        return DeepgramTTSService(
            api_key=key, settings=DeepgramTTSService.Settings(voice=voice or NOT_GIVEN)
        )
    # openai, the HyperNoesis API and the free local service: /v1/audio/speech
    return OpenAITTSService(
        api_key=key,
        base_url=choice.base_url,
        settings=OpenAITTSService.Settings(
            model=choice.model or "tts-1", voice=choice.voice or NOT_GIVEN
        ),
    )


def turn_strategies(language: Language) -> UserTurnStrategies:
    """When the caller's turn ends.

    Pipecat's local Smart Turn v3 model listens to the audio to tell a finished
    sentence from a pause. Its language list does not include Greek, so Greek
    (and the bilingual default, which prefers Greek) ends a turn after a short
    silence instead.
    """
    start = [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]
    if language == "en":
        stop: list[Any] = [
            TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())
        ]
    else:
        stop = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=SILENCE_TURN_SECONDS)]
    return UserTurnStrategies(start=start, stop=stop)
