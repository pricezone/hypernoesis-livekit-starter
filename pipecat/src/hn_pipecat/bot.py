"""One Pipecat voice pipeline in one LiveKit room.

Pipeline: LiveKit audio in (optional DeepFilterNet3 filter) -> Silero VAD ->
speech-to-text -> LLM -> text-to-speech -> LiveKit audio out. The bot joins with
a token it mints itself, greets the first caller (AI disclosure first), and
leaves when the last caller leaves, when no caller arrives within
`HN_JOIN_TIMEOUT` seconds, when the room closes or when the worker stops.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from livekit import rtc
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
from pipecat.turns.user_mute import FirstSpeechUserMuteStrategy
from pipecat.workers.runner import WorkerRunner

from . import noise
from .config import AgentConfig
from .dispatch import RoomJob
from .greeting import build_greeting, build_instructions, disclosure_line
from .livekit_auth import Credentials, mint_room_token
from .providers import (
    Choice,
    base_url_mismatch,
    describe,
    missing_hint,
    select_llm,
    select_stt,
    select_tts,
)
from .remote import AgentConfigLoader
from .services import build_llm, build_stt, build_tts, turn_strategies

logger = logging.getLogger("hn_pipecat.bot")

CALLER_KINDS: Final = frozenset(
    {rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD, rtc.ParticipantKind.PARTICIPANT_KIND_SIP}
)
# Pipecat's own idle timeout: leave after this long without anyone speaking.
IDLE_TIMEOUT_SECONDS: Final = 300.0


@dataclass(frozen=True)
class BotContext:
    livekit_url: str
    credentials: Credentials
    agent_name: str
    loader: AgentConfigLoader
    env: Mapping[str, str]
    join_timeout: float


@dataclass(frozen=True)
class Plan:
    """Everything chosen for one session, before any connection is made."""

    config: AgentConfig
    llm: Choice
    stt: Choice
    tts: Choice


def plan_session(config: AgentConfig, env: Mapping[str, str]) -> Plan | None:
    """Pick the providers; None (with a log line per missing stage) if one is missing."""
    llm_choice = select_llm(config.llm, env)
    stt_choice = select_stt(config.stt, env)
    tts_choice = select_tts(config.tts, env)
    if llm_choice is None or stt_choice is None or tts_choice is None:
        for kind, choice in (("llm", llm_choice), ("stt", stt_choice), ("tts", tts_choice)):
            if choice is None:
                logger.error("no %s provider is configured: %s", kind.upper(), missing_hint(kind))
        return None
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
    return Plan(config, llm_choice, stt_choice, tts_choice)


def noise_filter(config: AgentConfig, env: Mapping[str, str]) -> noise.DeepFilterAudioFilter | None:
    if not config.noise_cancellation:
        return None
    runtime = noise.load_runtime(env)
    if isinstance(runtime, str):
        logger.warning("noise cancellation is on but unavailable: %s", runtime)
        return None
    # The model loads in a background thread; audio passes through until then.
    return noise.DeepFilterAudioFilter(runtime)


def is_caller(kind: int) -> bool:
    return kind in CALLER_KINDS


def caller_identities(transport: Any) -> set[str]:
    """Identities of the callers (standard and SIP participants) in the room.

    Pipecat's transport reports identities only, so the participant kinds come
    from the LiveKit room object it holds. If that is not reachable, every
    remote participant counts as a caller.
    """
    try:
        room: rtc.Room = transport._client.room
        participants = list(room.remote_participants.values())
    except Exception:
        return set(transport.get_participants())
    return {p.identity for p in participants if is_caller(p.kind)}


@dataclass
class Session:
    """A built, not yet connected, pipeline for one room."""

    transport: LiveKitTransport
    worker: PipelineWorker
    greeting: str
    disclosure: str | None

    def greeting_frame(self) -> TTSSpeakFrame | None:
        """The first words; the AI disclosure in them cannot be interrupted."""
        if not self.greeting:
            return None
        frame = TTSSpeakFrame(self.greeting)
        frame.interruptible = self.disclosure is None
        return frame


def build_session(job: RoomJob, ctx: BotContext, plan: Plan) -> Session:
    config = plan.config
    token = mint_room_token(
        ctx.credentials, room=job.room, identity=job.identity, name=ctx.agent_name
    )
    transport = LiveKitTransport(
        url=ctx.livekit_url,
        token=token,
        room_name=job.room,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # One mixed stream for the single STT + VAD of this pipeline.
            audio_in_user_tracks=False,
            audio_in_filter=noise_filter(config, ctx.env),
        ),
    )
    disclosure = disclosure_line(config)
    aggregators = LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_turn_strategies=turn_strategies(config.language),
            # The disclosure must be heard in full: callers cannot talk over
            # the first thing the bot says.
            user_mute_strategies=[FirstSpeechUserMuteStrategy()] if disclosure else [],
        ),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            build_stt(plan.stt, config.language, ctx.env),
            aggregators.user(),
            build_llm(plan.llm, ctx.env, build_instructions(config)),
            build_tts(plan.tts, config.language, ctx.env),
            transport.output(),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=False),
        enable_rtvi=False,
        idle_timeout_secs=IDLE_TIMEOUT_SECONDS,
        # A transport that cannot connect, or a provider that refuses the key,
        # will not recover: leave the room instead of sitting in it silently.
        processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
    )
    return Session(transport, worker, build_greeting(config), disclosure)


async def run_room_bot(job: RoomJob, ctx: BotContext) -> None:
    """Join `job.room`, talk to its callers and return once the bot has left."""
    if job.leave.is_set():
        return
    config = await ctx.loader.get()
    plan = plan_session(config, ctx.env)
    if plan is None:
        return
    session = build_session(job, ctx, plan)
    transport, worker = session.transport, session.worker
    state = _RoomState()

    async def stop(reason: str) -> None:
        if state.stopping:
            return
        state.stopping = True
        logger.info("leaving room %s: %s", job.room, reason)
        await worker.cancel(reason=reason)

    @transport.event_handler("on_participant_connected")
    async def on_participant_connected(transport: LiveKitTransport, participant_id: str) -> None:
        if participant_id not in caller_identities(transport):
            return
        state.caller_seen.set()
        if not state.greeted:
            state.greeted = True
            frame = session.greeting_frame()
            if frame is not None:
                await worker.queue_frame(frame)

    @transport.event_handler("on_participant_disconnected")
    async def on_participant_disconnected(transport: LiveKitTransport, participant_id: str) -> None:
        if state.caller_seen.is_set() and not caller_identities(transport):
            await stop("the last caller left")

    @transport.event_handler("on_disconnected")
    async def on_disconnected(transport: LiveKitTransport) -> None:
        await stop("disconnected from the room")

    async def leave_if_nobody_comes() -> None:
        try:
            await asyncio.wait_for(state.caller_seen.wait(), timeout=ctx.join_timeout)
        except TimeoutError:
            await stop(f"no caller within {ctx.join_timeout:.0f} s")

    async def leave_on_request() -> None:
        await job.leave.wait()
        await stop(job.leave_reason or "asked to leave")

    runner = WorkerRunner(handle_sigint=False, name=f"room-{job.identity}")
    await runner.add_workers(worker)
    watchers = [
        asyncio.create_task(leave_if_nobody_comes()),
        asyncio.create_task(leave_on_request()),
    ]
    try:
        await runner.run()
    finally:
        for task in watchers:
            task.cancel()
        for task in watchers:
            with contextlib.suppress(asyncio.CancelledError):
                await task


class _RoomState:
    def __init__(self) -> None:
        self.caller_seen = asyncio.Event()
        self.greeted = False
        self.stopping = False
