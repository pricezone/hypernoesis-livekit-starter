import asyncio
from collections.abc import AsyncIterator

import pytest
from aiohttp.test_utils import TestClient, TestServer
from livekit.protocol import models

from helpers import API_KEY, API_SECRET, event_body, participant_joined, sign
from hn_pipecat import __version__
from hn_pipecat.dispatch import RoomJob
from hn_pipecat.livekit_auth import Credentials
from hn_pipecat.server import DISPATCHER, STATE, create_app, pipecat_version, probe_once
from hn_pipecat.worker_config import WorkerConfig

CONFIG = WorkerConfig(
    livekit_url="ws://127.0.0.1:7880",
    credentials=(Credentials(API_KEY, API_SECRET),),
    room_prefix="call-",
    max_rooms=4,
)


class FakeRooms:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.rooms: list[models.Room] = []

    async def list_rooms(self) -> list[models.Room]:
        if self.fail:
            raise ConnectionError("down")
        return self.rooms

    async def list_participants(self, room: str) -> list[models.ParticipantInfo]:
        return []


class Bots:
    def __init__(self) -> None:
        self.jobs: list[RoomJob] = []

    async def __call__(self, job: RoomJob) -> None:
        self.jobs.append(job)
        await job.leave.wait()


@pytest.fixture
def bots() -> Bots:
    return Bots()


@pytest.fixture
def rooms() -> FakeRooms:
    return FakeRooms()


@pytest.fixture
async def client(bots: Bots, rooms: FakeRooms) -> AsyncIterator[TestClient]:
    app = create_app(CONFIG, {}, rooms=rooms, run_bot=bots, probe=False)
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_health_is_503_until_livekit_answers(client: TestClient, rooms: FakeRooms) -> None:
    resp = await client.get("/")
    assert resp.status == 503
    assert await resp.text() == "not connected to LiveKit yet"

    app = client.app
    assert app is not None
    rooms.fail = True
    assert await probe_once(app[DISPATCHER], rooms, app[STATE]) is False
    resp = await client.get("/")
    assert resp.status == 503
    assert "not reachable" in await resp.text()

    rooms.fail = False
    assert await probe_once(app[DISPATCHER], rooms, app[STATE]) is True
    resp = await client.get("/")
    assert resp.status == 200
    assert await resp.text() == "OK"


async def test_worker_status_shape(client: TestClient) -> None:
    resp = await client.get("/worker")
    assert resp.status == 200
    assert resp.content_type == "application/json"
    assert await resp.json() == {
        "agent_name": "hypernoesis-pipecat",
        "worker_type": "JT_ROOM",
        "active_jobs": 0,
        "capacity": 4,
        "load": 0.0,
        "worker_load": 0.0,
        "accepting_jobs": True,
        "sdk_version": pipecat_version(),
        "framework": "pipecat",
        "version": __version__,
    }
    assert pipecat_version() == "1.17.0"


async def test_signed_webhook_starts_a_bot_and_shows_in_status(
    client: TestClient, bots: Bots
) -> None:
    body = event_body("room_started", "call-1")
    resp = await client.post("/webhook", data=body, headers={"Authorization": sign(body)})
    assert resp.status == 200
    await asyncio.sleep(0.01)
    assert [j.room for j in bots.jobs] == ["call-1"]
    status = await (await client.get("/worker")).json()
    assert (status["active_jobs"], status["load"]) == (1, 0.25)

    body = event_body("room_finished", "call-1")
    resp = await client.post("/webhook", data=body, headers={"Authorization": sign(body)})
    assert resp.status == 200
    await asyncio.sleep(0.01)
    status = await (await client.get("/worker")).json()
    assert status["active_jobs"] == 0


async def test_sip_participant_webhook_starts_a_bot(client: TestClient, bots: Bots) -> None:
    body = participant_joined("call-9", "SIP", identity="sip_+302100000000")
    resp = await client.post("/webhook", data=body, headers={"Authorization": sign(body)})
    assert resp.status == 200
    await asyncio.sleep(0.01)
    assert [j.room for j in bots.jobs] == ["call-9"]


async def test_unsigned_or_tampered_webhooks_are_refused(client: TestClient, bots: Bots) -> None:
    body = event_body("room_started", "call-1")
    resp = await client.post("/webhook", data=body)
    assert resp.status == 401
    resp = await client.post(
        "/webhook",
        data=event_body("room_started", "call-2"),
        headers={"Authorization": sign(body)},
    )
    assert resp.status == 401
    await asyncio.sleep(0.01)
    assert bots.jobs == []


async def test_room_outside_the_prefix_is_not_joined(client: TestClient, bots: Bots) -> None:
    body = event_body("room_started", "meeting-1")
    resp = await client.post("/webhook", data=body, headers={"Authorization": sign(body)})
    assert resp.status == 200
    await asyncio.sleep(0.01)
    assert bots.jobs == []


async def test_unconfigured_worker_refuses_webhooks_and_is_unhealthy(bots: Bots) -> None:
    config = WorkerConfig(livekit_url=None, credentials=())
    app = create_app(config, {}, run_bot=bots, probe=False)
    async with TestClient(TestServer(app)) as c:
        resp = await c.get("/")
        assert resp.status == 503
        assert "LIVEKIT_URL is not set" in await resp.text()
        body = event_body("room_started", "call-1")
        resp = await c.post("/webhook", data=body, headers={"Authorization": sign(body)})
        assert resp.status == 503


async def test_first_successful_probe_joins_waiting_callers(bots: Bots) -> None:
    class Waiting(FakeRooms):
        async def list_participants(self, room: str) -> list[models.ParticipantInfo]:
            return [models.ParticipantInfo(identity="caller", kind=models.ParticipantInfo.SIP)]

    rooms = Waiting()
    rooms.rooms = [models.Room(name="call-5", num_participants=1)]
    app = create_app(CONFIG, {}, rooms=rooms, run_bot=bots, probe=False)
    assert await probe_once(app[DISPATCHER], rooms, app[STATE]) is True
    assert await probe_once(app[DISPATCHER], rooms, app[STATE]) is True  # only once
    await asyncio.sleep(0.01)
    assert [j.room for j in bots.jobs] == ["call-5"]
    await app[DISPATCHER].shutdown(grace=1.0)


async def test_shutdown_turns_health_red(client: TestClient, rooms: FakeRooms) -> None:
    app = client.app
    assert app is not None
    await probe_once(app[DISPATCHER], rooms, app[STATE])
    await app[DISPATCHER].shutdown()
    resp = await client.get("/")
    assert resp.status == 503
    assert await resp.text() == "shutting down"
