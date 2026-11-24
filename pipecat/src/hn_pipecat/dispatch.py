"""Which rooms the bot joins, and the bookkeeping of the rooms it is in.

Pipecat has no LiveKit agent dispatch, so this worker dispatches itself from
LiveKit webhooks (see README.md):

* `room_started`, or a caller (a SIP participant or a standard participant)
  joining a room, starts a bot in that room if the room name matches
  `HN_ROOM_PREFIX`, the bot is not in it yet, no other agent serves it and
  there is a free slot (`HN_MAX_ROOMS`). One bot per room.
* `room_finished` makes the bot leave. The bot also leaves on its own when the
  last caller leaves (see bot.py).
* At start-up, rooms that already have a caller and no agent are joined too, so
  callers are not left alone after a restart.

`decide()` is pure; `Dispatcher` keeps the room table and runs one asyncio task
per room.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Collection, Iterable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from livekit.protocol import models
from livekit.protocol.webhook import WebhookEvent

logger = logging.getLogger("hn_pipecat.dispatch")

WORKER_TYPE: Final = "JT_ROOM"
_KIND = models.ParticipantInfo.Kind
CALLER_KINDS: Final = frozenset({_KIND.STANDARD, _KIND.SIP})
SEEN_EVENT_IDS: Final = 1024

Action = Literal["join", "leave", "ignore"]


@dataclass(frozen=True)
class Decision:
    action: Action
    room: str
    reason: str


def decide(
    event: WebhookEvent,
    *,
    prefix: str,
    served: Collection[str],
    capacity_left: int,
    accepting: bool = True,
) -> Decision:
    """What to do about one webhook event."""
    room = event.room.name if event.HasField("room") else ""
    kind = event.event

    if kind == "room_finished":
        if room in served:
            return Decision("leave", room, "room finished")
        return Decision("ignore", room, "room finished (not in it)")
    if kind == "room_started":
        trigger = "room started"
    elif kind == "participant_joined":
        participant_kind = event.participant.kind
        if participant_kind not in CALLER_KINDS:
            return Decision("ignore", room, "participant is not a caller")
        trigger = "SIP caller joined" if participant_kind == _KIND.SIP else "participant joined"
    else:
        return Decision("ignore", room, f"event {kind or '?'} is not used")

    return _join_decision(room, trigger, prefix, served, capacity_left, accepting)


def _join_decision(
    room: str,
    trigger: str,
    prefix: str,
    served: Collection[str],
    capacity_left: int,
    accepting: bool,
) -> Decision:
    if not room:
        return Decision("ignore", room, "event has no room name")
    if not room.startswith(prefix):
        return Decision("ignore", room, "room name does not match HN_ROOM_PREFIX")
    if room in served:
        return Decision("ignore", room, "already in this room")
    if not accepting:
        return Decision("ignore", room, "shutting down")
    if capacity_left <= 0:
        return Decision("ignore", room, "at capacity (HN_MAX_ROOMS)")
    return Decision("join", room, trigger)


def other_agent(participants: Iterable[models.ParticipantInfo], own_prefix: str) -> str | None:
    """Identity of an agent in the room that is not one of ours, if any."""
    for p in participants:
        if p.kind == _KIND.AGENT and not p.identity.startswith(own_prefix):
            return p.identity
    return None


def has_caller(participants: Iterable[models.ParticipantInfo]) -> bool:
    return any(p.kind in CALLER_KINDS for p in participants)


class RoomDirectory(Protocol):
    """The part of LiveKit's RoomService the dispatcher needs."""

    async def list_rooms(self) -> list[models.Room]: ...

    async def list_participants(self, room: str) -> list[models.ParticipantInfo]: ...


@dataclass
class RoomJob:
    room: str
    identity: str
    reason: str
    started_at: float
    leave: asyncio.Event = field(default_factory=asyncio.Event)
    leave_reason: str = ""
    task: asyncio.Task[None] | None = None

    def request_leave(self, reason: str) -> None:
        if not self.leave.is_set():
            self.leave_reason = reason
            self.leave.set()


BotRunner = Callable[[RoomJob], Awaitable[None]]


class Dispatcher:
    def __init__(
        self,
        *,
        agent_name: str,
        room_prefix: str,
        max_rooms: int,
        run_bot: BotRunner,
        rooms: RoomDirectory | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.agent_name = agent_name
        self.room_prefix = room_prefix
        self.max_rooms = max_rooms
        self._run_bot = run_bot
        self._rooms = rooms
        self._clock = clock
        self._jobs: dict[str, RoomJob] = {}
        self._seen: OrderedDict[str, None] = OrderedDict()
        self.accepting = True

    @property
    def own_prefix(self) -> str:
        return f"{self.agent_name}-"

    @property
    def jobs(self) -> dict[str, RoomJob]:
        return dict(self._jobs)

    def handle(self, event: WebhookEvent) -> Decision:
        """Act on one verified webhook event; never blocks."""
        if event.id:
            if event.id in self._seen:
                return Decision("ignore", event.room.name, "duplicate delivery")
            self._seen[event.id] = None
            while len(self._seen) > SEEN_EVENT_IDS:
                self._seen.popitem(last=False)

        decision = decide(
            event,
            prefix=self.room_prefix,
            served=self._jobs.keys(),
            capacity_left=self.max_rooms - len(self._jobs),
            accepting=self.accepting,
        )
        if decision.action == "join":
            self._start(decision.room, decision.reason)
        elif decision.action == "leave":
            self.leave(decision.room, decision.reason)
        elif decision.reason.startswith("at capacity"):
            logger.warning("not joining room %s: %s", decision.room, decision.reason)
        return decision

    def leave(self, room: str, reason: str) -> bool:
        job = self._jobs.get(room)
        if job is None:
            return False
        job.request_leave(reason)
        return True

    async def reconcile(self) -> int:
        """Join rooms that already have a caller and no agent (after a restart)."""
        if self._rooms is None:
            return 0
        joined = 0
        for room in await self._rooms.list_rooms():
            if room.num_participants == 0:
                continue
            decision = _join_decision(
                room.name,
                "caller waiting at start-up",
                self.room_prefix,
                self._jobs.keys(),
                self.max_rooms - len(self._jobs),
                self.accepting,
            )
            if decision.action != "join":
                continue
            participants = await self._rooms.list_participants(room.name)
            if has_caller(participants) and other_agent(participants, self.own_prefix) is None:
                self._start(room.name, decision.reason, check_other_agent=False)
                joined += 1
        return joined

    def status(self) -> dict[str, Any]:
        active = len(self._jobs)
        load = round(min(active / self.max_rooms, 1.0), 3)
        return {
            "agent_name": self.agent_name,
            "worker_type": WORKER_TYPE,
            "active_jobs": active,
            "capacity": self.max_rooms,
            "load": load,
            "worker_load": load,
            "accepting_jobs": self.accepting and active < self.max_rooms,
        }

    async def shutdown(self, grace: float = 10.0) -> None:
        """Stop accepting rooms, ask every bot to leave and wait for them."""
        self.accepting = False
        jobs = list(self._jobs.values())
        for job in jobs:
            job.request_leave("worker shutting down")
        tasks = [job.task for job in jobs if job.task is not None]
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=grace)
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # -- internals -----------------------------------------------------------

    def _start(self, room: str, reason: str, *, check_other_agent: bool = True) -> RoomJob:
        identity = f"{self.own_prefix}{secrets.token_hex(4)}"
        job = RoomJob(room=room, identity=identity, reason=reason, started_at=self._clock())
        # Reserve the room before the first await, so a second event for the
        # same room (room_started + participant_joined) cannot start a second bot.
        self._jobs[room] = job
        job.task = asyncio.create_task(self._run(job, check_other_agent), name=f"room:{room}")
        logger.info("joining room %s (%s)", room, reason)
        return job

    async def _run(self, job: RoomJob, check_other_agent: bool) -> None:
        ran = False
        try:
            if check_other_agent and await self._served_by_other_agent(job.room):
                return
            ran = True
            await self._run_bot(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("the bot in room %s stopped with an error", job.room)
        finally:
            if self._jobs.get(job.room) is job:
                del self._jobs[job.room]
            if ran:
                logger.info("left room %s", job.room)

    async def _served_by_other_agent(self, room: str) -> bool:
        if self._rooms is None:
            return False
        try:
            participants = await self._rooms.list_participants(room)
        except Exception as err:
            logger.warning(
                "could not list the participants of room %s (%s): joining anyway",
                room,
                type(err).__name__,
            )
            return False
        agent = other_agent(participants, self.own_prefix)
        if agent is not None:
            logger.info("room %s is already served by agent %s: not joining", room, agent)
            return True
        return False
