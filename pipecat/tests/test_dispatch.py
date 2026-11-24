import asyncio
import json

import pytest
from google.protobuf.json_format import Parse
from livekit.protocol import models
from livekit.protocol.webhook import WebhookEvent

from helpers import event_body, participant_joined
from hn_pipecat.dispatch import Decision, Dispatcher, RoomJob, decide, has_caller, other_agent


def event(body: bytes) -> WebhookEvent:
    return Parse(body.decode(), WebhookEvent(), ignore_unknown_fields=True)


def with_id(body: bytes, event_id: str) -> bytes:
    data = json.loads(body)
    data["id"] = event_id
    return json.dumps(data).encode()


def participant(identity: str, kind: int) -> models.ParticipantInfo:
    return models.ParticipantInfo(identity=identity, kind=kind)  # type: ignore[arg-type]


K = models.ParticipantInfo.Kind


# -- decide() -------------------------------------------------------------------


def test_room_started_joins() -> None:
    d = decide(event(event_body("room_started", "call-1")), prefix="", served=(), capacity_left=1)
    assert d == Decision("join", "call-1", "room started")


@pytest.mark.parametrize(
    ("kind", "action", "reason"),
    [
        ("SIP", "join", "SIP caller joined"),
        ("STANDARD", "join", "participant joined"),
        ("AGENT", "ignore", "participant is not a caller"),
        ("EGRESS", "ignore", "participant is not a caller"),
        ("INGRESS", "ignore", "participant is not a caller"),
    ],
)
def test_participant_joined_by_kind(kind: str, action: str, reason: str) -> None:
    d = decide(event(participant_joined("call-1", kind)), prefix="", served=(), capacity_left=1)
    assert (d.action, d.reason) == (action, reason)


def test_prefix_filters_rooms() -> None:
    e = event(event_body("room_started", "meeting-1"))
    assert decide(e, prefix="call-", served=(), capacity_left=1).action == "ignore"
    e = event(event_body("room_started", "call-7"))
    assert decide(e, prefix="call-", served=(), capacity_left=1).action == "join"


def test_one_bot_per_room_capacity_and_shutdown() -> None:
    e = event(event_body("room_started", "call-1"))
    assert decide(e, prefix="", served={"call-1"}, capacity_left=1).reason == "already in this room"
    assert decide(e, prefix="", served=(), capacity_left=0).reason.startswith("at capacity")
    assert decide(e, prefix="", served=(), capacity_left=1, accepting=False).reason == (
        "shutting down"
    )


def test_room_finished_leaves_only_served_rooms() -> None:
    e = event(event_body("room_finished", "call-1"))
    assert decide(e, prefix="", served={"call-1"}, capacity_left=0).action == "leave"
    assert decide(e, prefix="", served=(), capacity_left=1).action == "ignore"


def test_other_events_and_missing_room_are_ignored() -> None:
    e = event(event_body("track_published", "call-1"))
    assert decide(e, prefix="", served=(), capacity_left=1).action == "ignore"
    e = event(b'{"event": "room_started"}')
    assert decide(e, prefix="", served=(), capacity_left=1).reason == "event has no room name"


def test_participant_helpers() -> None:
    people = [participant("caller", K.SIP), participant("hypernoesis-pipecat-ab", K.AGENT)]
    assert has_caller(people)
    assert other_agent(people, "hypernoesis-pipecat-") is None
    people.append(participant("agent-AJ_123", K.AGENT))
    assert other_agent(people, "hypernoesis-pipecat-") == "agent-AJ_123"
    assert not has_caller([participant("rec", K.EGRESS)])


# -- Dispatcher -------------------------------------------------------------------


class FakeBots:
    """Bots that stay in their room until asked to leave."""

    def __init__(self) -> None:
        self.started: list[RoomJob] = []
        self.entered = asyncio.Event()

    async def __call__(self, job: RoomJob) -> None:
        self.started.append(job)
        self.entered.set()
        await job.leave.wait()


class FakeRooms:
    def __init__(
        self,
        rooms: list[models.Room] | None = None,
        participants: dict[str, list[models.ParticipantInfo]] | None = None,
        fail: bool = False,
    ) -> None:
        self.rooms = rooms or []
        self.participants = participants or {}
        self.fail = fail

    async def list_rooms(self) -> list[models.Room]:
        return self.rooms

    async def list_participants(self, room: str) -> list[models.ParticipantInfo]:
        if self.fail:
            raise ConnectionError("down")
        return self.participants.get(room, [])


def make(bots: FakeBots, rooms: FakeRooms | None = None, **kwargs) -> Dispatcher:
    options = {"agent_name": "hypernoesis-pipecat", "room_prefix": "", "max_rooms": 2}
    options.update(kwargs)
    return Dispatcher(run_bot=bots, rooms=rooms or FakeRooms(), **options)


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_one_bot_per_room_even_for_back_to_back_events() -> None:
    bots = FakeBots()
    d = make(bots)
    assert d.handle(event(event_body("room_started", "call-1"))).action == "join"
    assert d.handle(event(participant_joined("call-1", "SIP"))).reason == "already in this room"
    await settle()
    assert [j.room for j in bots.started] == ["call-1"]
    job = bots.started[0]
    assert job.identity.startswith("hypernoesis-pipecat-")
    assert d.status()["active_jobs"] == 1


async def test_capacity_and_status() -> None:
    bots = FakeBots()
    d = make(bots, max_rooms=2)
    for name in ("a", "b", "c"):
        d.handle(event(event_body("room_started", name)))
    await settle()
    status = d.status()
    assert status == {
        "agent_name": "hypernoesis-pipecat",
        "worker_type": "JT_ROOM",
        "active_jobs": 2,
        "capacity": 2,
        "load": 1.0,
        "worker_load": 1.0,
        "accepting_jobs": False,
    }
    assert sorted(d.jobs) == ["a", "b"]


async def test_room_finished_makes_the_bot_leave_and_frees_the_slot() -> None:
    bots = FakeBots()
    d = make(bots)
    d.handle(event(event_body("room_started", "call-1")))
    await settle()
    decision = d.handle(event(event_body("room_finished", "call-1")))
    assert decision.action == "leave"
    job = bots.started[0]
    assert job.leave.is_set()
    assert job.leave_reason == "room finished"
    await settle()
    assert d.jobs == {}
    assert d.status()["load"] == 0.0


async def test_duplicate_deliveries_are_ignored() -> None:
    bots = FakeBots()
    d = make(bots)
    body = with_id(event_body("room_started", "call-1"), "EV_1")
    assert d.handle(event(body)).action == "join"
    await settle()
    d.handle(event(with_id(event_body("room_finished", "call-1"), "EV_2")))
    await settle()
    assert d.handle(event(body)).reason == "duplicate delivery"
    assert d.jobs == {}


async def test_room_with_another_agent_is_not_joined() -> None:
    bots = FakeBots()
    rooms = FakeRooms(participants={"call-1": [participant("agent-AJ_1", K.AGENT)]})
    d = make(bots, rooms)
    assert d.handle(event(participant_joined("call-1", "SIP"))).action == "join"
    await settle()
    assert bots.started == []
    assert d.jobs == {}


async def test_participant_listing_failure_still_joins() -> None:
    bots = FakeBots()
    d = make(bots, FakeRooms(fail=True))
    d.handle(event(participant_joined("call-1", "SIP")))
    await settle()
    assert [j.room for j in bots.started] == ["call-1"]


async def test_a_failing_bot_frees_its_slot() -> None:
    async def broken(job: RoomJob) -> None:
        raise RuntimeError("boom")

    d = Dispatcher(agent_name="x", room_prefix="", max_rooms=1, run_bot=broken)
    d.handle(event(event_body("room_started", "call-1")))
    await settle()
    assert d.jobs == {}


async def test_reconcile_joins_rooms_with_a_waiting_caller() -> None:
    bots = FakeBots()
    rooms = FakeRooms(
        rooms=[
            models.Room(name="call-waiting", num_participants=1),
            models.Room(name="call-empty", num_participants=0),
            models.Room(name="call-served", num_participants=2),
            models.Room(name="call-recorder", num_participants=1),
            models.Room(name="other-waiting", num_participants=1),
        ],
        participants={
            "call-waiting": [participant("caller", K.SIP)],
            "call-served": [participant("caller", K.STANDARD), participant("a", K.AGENT)],
            "call-recorder": [participant("rec", K.EGRESS)],
            "other-waiting": [participant("caller", K.STANDARD)],
        },
    )
    d = make(bots, rooms, room_prefix="call-", max_rooms=5)
    assert await d.reconcile() == 1
    await settle()
    assert [j.room for j in bots.started] == ["call-waiting"]
    assert bots.started[0].reason == "caller waiting at start-up"


async def test_shutdown_asks_every_bot_to_leave() -> None:
    bots = FakeBots()
    d = make(bots)
    d.handle(event(event_body("room_started", "a")))
    d.handle(event(event_body("room_started", "b")))
    await settle()
    await d.shutdown(grace=1.0)
    assert d.jobs == {}
    assert all(j.leave_reason == "worker shutting down" for j in bots.started)
    assert d.handle(event(event_body("room_started", "c"))).reason == "shutting down"


async def test_shutdown_cancels_bots_that_do_not_leave() -> None:
    async def stubborn(job: RoomJob) -> None:
        await asyncio.sleep(3600)

    d = Dispatcher(agent_name="x", room_prefix="", max_rooms=1, run_bot=stubborn)
    d.handle(event(event_body("room_started", "a")))
    await settle()
    await d.shutdown(grace=0.05)
    assert d.jobs == {}
