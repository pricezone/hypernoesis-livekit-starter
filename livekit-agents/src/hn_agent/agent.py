"""The HyperNoesis starter voice agent for Self-Hosted LiveKit.

Pipeline: speech-to-text -> LLM -> text-to-speech, with the Silero VAD and
LiveKit's local multilingual turn detector, all chosen per session from the
settings HyperNoesis serves and the keys present on this server.
"""

from __future__ import annotations

import logging
import os
from typing import Final

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    inference,
    room_io,
)

from . import noise
from .config import AgentConfig
from .greeting import build_greeting, build_instructions, disclosure_line
from .pipeline import build_llm, build_stt, build_tts, turn_detection_for
from .providers import (
    Choice,
    base_url_mismatch,
    describe,
    missing_hint,
    select_llm,
    select_stt,
    select_tts,
)
from .remote import AgentConfigLoader, loader_from_env

logger = logging.getLogger("hn_agent")

DEFAULT_AGENT_NAME: Final = "hypernoesis-starter"
_TRUE: Final = frozenset({"1", "true", "yes", "on"})


def agent_name_from_env() -> str:
    """Name for explicit dispatch; "" (automatic dispatch to every room) on request."""
    if os.environ.get("HN_AUTO_DISPATCH", "").strip().lower() in _TRUE:
        return ""
    return os.environ.get("HN_AGENT_NAME", "").strip() or DEFAULT_AGENT_NAME


server = AgentServer()
_loader: AgentConfigLoader | None = None


def _get_loader() -> AgentConfigLoader:
    global _loader
    if _loader is None:
        _loader = loader_from_env(os.environ)
    return _loader


@server.rtc_session(agent_name=agent_name_from_env())
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    config = await _get_loader().get()
    env = os.environ

    llm_choice = select_llm(config.llm, env)
    stt_choice = select_stt(config.stt, env)
    tts_choice = select_tts(config.tts, env)
    missing = [
        kind
        for kind, choice in (("llm", llm_choice), ("stt", stt_choice), ("tts", tts_choice))
        if choice is None
    ]
    if missing or llm_choice is None or stt_choice is None or tts_choice is None:
        for kind in missing:
            logger.error("no %s provider is configured: %s", kind.upper(), missing_hint(kind))
        ctx.shutdown(reason="no provider configured")
        return
    _log_choices(config, llm_choice, stt_choice, tts_choice)

    vad = inference.VAD(model="silero")
    noise_filter = _noise_filter(config)
    session: AgentSession = AgentSession(
        stt=build_stt(stt_choice, config.language, env, vad),
        llm=build_llm(llm_choice, env),
        tts=build_tts(tts_choice, config.language, env),
        vad=vad,
        turn_handling=TurnHandlingOptions(
            turn_detection=turn_detection_for(config.language),
            # "vad" keeps interruption handling on this server; the adaptive
            # mode would call LiveKit Cloud.
            interruption={"mode": "vad"},
            preemptive_generation={"enabled": True},
        ),
    )

    audio_input = room_io.AudioInputOptions(
        sample_rate=noise.DF_SAMPLE_RATE if noise_filter else 24_000,
        noise_cancellation=noise_filter,
    )
    await session.start(
        agent=Agent(instructions=build_instructions(config)),
        room=ctx.room,
        room_options=room_io.RoomOptions(audio_input=audio_input),
    )
    await ctx.connect()

    greeting = build_greeting(config)
    if greeting:
        # The disclosure line must be heard in full, so it cannot be interrupted.
        session.say(greeting, allow_interruptions=disclosure_line(config) is None)


def _noise_filter(config: AgentConfig) -> noise.DeepFilterProcessor | None:
    if not config.noise_cancellation:
        return None
    runtime = noise.load_runtime(os.environ)
    if isinstance(runtime, str):
        logger.warning("noise cancellation is on but unavailable: %s", runtime)
        return None
    # The model loads in a background thread; audio passes through until then.
    return runtime.create()


def _log_choices(
    config: AgentConfig, llm_choice: Choice, stt_choice: Choice, tts_choice: Choice
) -> None:
    if base_url_mismatch(config.llm, llm_choice):
        logger.warning(
            "llm.baseUrl from the settings is not configured on this server; "
            "keys are only sent to the URL set in this server's environment"
        )
    logger.info(
        "session pipeline: %s | %s | %s | language=%s",
        describe("llm", llm_choice),
        describe("stt", stt_choice),
        describe("tts", tts_choice),
        config.language,
    )
