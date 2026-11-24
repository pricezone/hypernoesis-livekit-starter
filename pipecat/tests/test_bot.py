import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from livekit import api, rtc
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.livekit.transport import LiveKitTransport

from helpers import API_KEY, API_SECRET
from hn_pipecat import bot as bot_module
from hn_pipecat.bot import (
    BotContext,
    Session,
    build_session,
    caller_identities,
    noise_filter,
    plan_session,
    run_room_bot,
)
from hn_pipecat.config import DEFAULT_CONFIG, AgentConfig, DisclosureSettings
from hn_pipecat.dispatch import RoomJob
from hn_pipecat.livekit_auth import Credentials
from hn_pipecat.remote import AgentConfigLoader

LOCAL_ENV = {
    "OPENAI_API_KEY": "sk-test",
    "OPENAI_BASE_URL": "http://llm.internal:8000/v1",
    "HN_STT_URL": "http://hn-stt:8000/v1",
    "HN_TTS_URL": "http://hn-tts:8000/v1",
}


def context(env: dict[str, str]) -> BotContext:
    return BotContext(
        livekit_url="ws://127.0.0.1:7880",
        credentials=Credentials(API_KEY, API_SECRET),
        agent_name="hypernoesis-pipecat",
        loader=AgentConfigLoader(None, None),
        env=env,
        join_timeout=60.0,
    )


def job(room: str = "call-1") -> RoomJob:
    return RoomJob(room=room, identity="hypernoesis-pipecat-0001", reason="test", started_at=0.0)


def test_plan_needs_all_three_stages(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="hn_pipecat.bot"):
        assert plan_session(DEFAULT_CONFIG, {"HN_STT_URL": "http://hn-stt:8000"}) is None
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("no LLM provider is configured") for m in messages)
    assert any(m.startswith("no TTS provider is configured") for m in messages)
    assert not any("STT" in m for m in messages)


def test_plan_with_free_local_speech_and_an_llm_key() -> None:
    plan = plan_session(DEFAULT_CONFIG, LOCAL_ENV)
    assert plan is not None
    assert (plan.llm.provider, plan.stt.provider, plan.tts.provider) == ("openai", "local", "local")


def flatten(processor: Any) -> list[Any]:
    children = getattr(processor, "processors", None)
    if not children:
        return [processor]
    return [leaf for child in children for leaf in flatten(child)]


def test_session_pipeline_is_built_in_order_without_connecting() -> None:
    plan = plan_session(DEFAULT_CONFIG, LOCAL_ENV)
    assert plan is not None
    session = build_session(job(), context(LOCAL_ENV), plan)
    assert isinstance(session.transport, LiveKitTransport)
    params = session.transport._params
    assert params.audio_in_enabled and params.audio_out_enabled
    assert params.audio_in_user_tracks is False
    assert params.audio_in_filter is None

    processors = flatten(session.worker.pipeline)
    names = [type(p).__name__ for p in processors]
    stt_at = next(i for i, p in enumerate(processors) if isinstance(p, OpenAISTTService))
    llm_at = next(i for i, p in enumerate(processors) if isinstance(p, OpenAILLMService))
    tts_at = next(i for i, p in enumerate(processors) if isinstance(p, OpenAITTSService))
    assert names.index("LiveKitInputTransport") < stt_at < llm_at < tts_at
    assert tts_at < names.index("LiveKitOutputTransport")


def test_session_token_is_for_this_room_and_identity() -> None:
    plan = plan_session(DEFAULT_CONFIG, LOCAL_ENV)
    assert plan is not None
    session = build_session(job("call-77"), context(LOCAL_ENV), plan)
    token = session.transport._client._token
    claims = api.TokenVerifier(API_KEY, API_SECRET).verify(token)
    assert claims.identity == "hypernoesis-pipecat-0001"
    assert claims.video is not None and claims.video.room == "call-77"
    assert claims.kind == "agent"


def test_greeting_starts_with_an_uninterruptible_disclosure() -> None:
    plan = plan_session(DEFAULT_CONFIG, LOCAL_ENV)
    assert plan is not None
    frame = build_session(job(), context(LOCAL_ENV), plan).greeting_frame()
    assert frame is not None
    assert frame.text.startswith("Είμαι ψηφιακός βοηθός. I am a digital assistant.")
    assert frame.interruptible is False


def test_greeting_without_disclosure_can_be_interrupted_and_can_be_silent() -> None:
    config = AgentConfig(greeting="Hello!", ai_disclosure=DisclosureSettings(enabled=False))
    plan = plan_session(config, LOCAL_ENV)
    assert plan is not None
    frame = build_session(job(), context(LOCAL_ENV), plan).greeting_frame()
    assert frame is not None
    assert (frame.text, frame.interruptible) == ("Hello!", True)

    silent = AgentConfig(greeting="", ai_disclosure=DisclosureSettings(enabled=False))
    plan = plan_session(silent, LOCAL_ENV)
    assert plan is not None
    assert build_session(job(), context(LOCAL_ENV), plan).greeting_frame() is None


def test_noise_filter_toggle(caplog: pytest.LogCaptureFixture) -> None:
    assert noise_filter(DEFAULT_CONFIG, {}) is None
    config = AgentConfig(noise_cancellation=True)
    with caplog.at_level(logging.WARNING, logger="hn_pipecat.bot"):
        assert noise_filter(config, {"HN_DFN_LIB": "/nonexistent/libdf.so"}) is None
    assert "noise cancellation is on but unavailable" in caplog.text


def fake_transport(*participants: tuple[str, int]) -> SimpleNamespace:
    remote = {
        identity: SimpleNamespace(identity=identity, kind=kind) for identity, kind in participants
    }
    return SimpleNamespace(
        _client=SimpleNamespace(room=SimpleNamespace(remote_participants=remote)),
        get_participants=lambda: list(remote),
    )


def test_callers_are_standard_and_sip_participants_only() -> None:
    k = rtc.ParticipantKind
    transport = fake_transport(
        ("web-user", k.PARTICIPANT_KIND_STANDARD),
        ("sip_+30210", k.PARTICIPANT_KIND_SIP),
        ("other-agent", k.PARTICIPANT_KIND_AGENT),
        ("recorder", k.PARTICIPANT_KIND_EGRESS),
    )
    assert caller_identities(transport) == {"web-user", "sip_+30210"}
    assert caller_identities(fake_transport(("other-agent", k.PARTICIPANT_KIND_AGENT))) == set()


def test_callers_fall_back_to_all_participants_without_a_room() -> None:
    transport = SimpleNamespace(get_participants=lambda: ["a", "b"])
    assert caller_identities(transport) == {"a", "b"}


# -- lifecycle (fake transport and runner; no LiveKit server) --------------------


class FakeTransport:
    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.remote: dict[str, SimpleNamespace] = {}
        self._client = SimpleNamespace(room=SimpleNamespace(remote_participants=self.remote))

    def event_handler(self, name: str) -> Any:
        def register(fn: Any) -> Any:
            self.handlers[name] = fn
            return fn

        return register

    def get_participants(self) -> list[str]:
        return list(self.remote)

    async def join(self, identity: str, kind: int) -> None:
        self.remote[identity] = SimpleNamespace(identity=identity, kind=kind)
        await self.handlers["on_participant_connected"](self, identity)

    async def leave(self, identity: str) -> None:
        del self.remote[identity]
        await self.handlers["on_participant_disconnected"](self, identity)


class FakeWorker:
    def __init__(self) -> None:
        self.frames: list[Any] = []
        self.cancelled = asyncio.Event()
        self.reasons: list[str] = []

    async def queue_frame(self, frame: Any) -> None:
        self.frames.append(frame)

    async def cancel(self, *, reason: str | None = None) -> None:
        self.reasons.append(reason or "")
        self.cancelled.set()


class FakeRunner:
    def __init__(self, **kwargs: Any) -> None:
        self.worker: FakeWorker | None = None

    async def add_workers(self, worker: FakeWorker) -> None:
        self.worker = worker

    async def run(self) -> None:
        assert self.worker is not None
        await self.worker.cancelled.wait()


@pytest.fixture
def fake_room(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeTransport, FakeWorker]:
    transport, worker = FakeTransport(), FakeWorker()

    def fake_build(job: RoomJob, ctx: BotContext, plan: Any) -> Session:
        return Session(transport, worker, "Είμαι ψηφιακός βοηθός. Γεια!", "Είμαι ψηφιακός βοηθός.")  # type: ignore[arg-type]

    monkeypatch.setattr(bot_module, "build_session", fake_build)
    monkeypatch.setattr(bot_module, "WorkerRunner", FakeRunner)
    return transport, worker


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


K = rtc.ParticipantKind


async def test_greets_the_first_caller_once_and_leaves_with_the_last(
    fake_room: tuple[FakeTransport, FakeWorker],
) -> None:
    transport, worker = fake_room
    task = asyncio.create_task(run_room_bot(job(), context(LOCAL_ENV)))
    await settle()
    await transport.join("recorder", K.PARTICIPANT_KIND_EGRESS)
    await transport.join("other-agent", K.PARTICIPANT_KIND_AGENT)
    assert worker.frames == []
    await transport.join("sip_+30210", K.PARTICIPANT_KIND_SIP)
    await transport.join("web-user", K.PARTICIPANT_KIND_STANDARD)
    assert len(worker.frames) == 1
    assert worker.frames[0].text == "Είμαι ψηφιακός βοηθός. Γεια!"
    assert worker.frames[0].interruptible is False

    await transport.leave("sip_+30210")
    await transport.leave("other-agent")
    assert not worker.cancelled.is_set()
    await transport.leave("web-user")
    await asyncio.wait_for(task, 1)
    assert worker.reasons == ["the last caller left"]


async def test_leaves_when_no_caller_comes(fake_room: tuple[FakeTransport, FakeWorker]) -> None:
    _, worker = fake_room
    ctx = context(LOCAL_ENV)
    quick = BotContext(**{**ctx.__dict__, "join_timeout": 0.05})
    await asyncio.wait_for(run_room_bot(job(), quick), 2)
    assert worker.reasons == ["no caller within 0 s"]
    assert worker.frames == []


async def test_leaves_when_asked_and_when_disconnected(
    fake_room: tuple[FakeTransport, FakeWorker],
) -> None:
    transport, worker = fake_room
    room_job = job()
    task = asyncio.create_task(run_room_bot(room_job, context(LOCAL_ENV)))
    await settle()
    await transport.join("caller", K.PARTICIPANT_KIND_STANDARD)
    room_job.request_leave("room finished")
    await asyncio.wait_for(task, 1)
    assert worker.reasons == ["room finished"]

    worker2 = FakeWorker()
    transport2 = FakeTransport()
    bot_module.build_session = lambda j, c, p: Session(transport2, worker2, "", None)  # type: ignore[assignment,arg-type]
    task = asyncio.create_task(run_room_bot(job(), context(LOCAL_ENV)))
    await settle()
    await transport2.handlers["on_disconnected"](transport2)
    await asyncio.wait_for(task, 1)
    assert worker2.reasons == ["disconnected from the room"]


async def test_no_providers_means_no_join(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_build(*args: Any) -> None:
        raise AssertionError("must not build a session")

    monkeypatch.setattr(bot_module, "build_session", must_not_build)
    await run_room_bot(job(), context({}))


async def test_a_job_told_to_leave_before_it_starts_does_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_build(*args: Any) -> None:
        raise AssertionError("must not build a session")

    monkeypatch.setattr(bot_module, "build_session", must_not_build)
    room_job = job()
    room_job.request_leave("worker shutting down")
    await run_room_bot(room_job, context(LOCAL_ENV))
