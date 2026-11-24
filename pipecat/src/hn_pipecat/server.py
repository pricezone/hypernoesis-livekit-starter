"""HTTP server on port 8081: LiveKit webhooks in, worker status out.

* `POST /webhook` - LiveKit webhooks (signed with the server's API key). Room
  and caller events start bots; see dispatch.py.
* `GET /` - health: 200 once LiveKit answers with the configured key, else 503
  (also while the worker shuts down).
* `GET /worker` - JSON status for the console's agent list, shaped like the
  LiveKit Agents worker status: `agent_name`, `worker_type`, `active_jobs`,
  `load`/`worker_load` (active rooms / `HN_MAX_ROOMS`), `sdk_version`, ...
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import logging
from collections.abc import Mapping
from typing import Any, Final

from aiohttp import web
from livekit import api
from livekit.protocol import models
from livekit.protocol.room import ListParticipantsRequest, ListRoomsRequest

from . import __version__
from .bot import BotContext, run_room_bot
from .dispatch import BotRunner, Dispatcher, RoomDirectory, RoomJob
from .livekit_auth import MAX_WEBHOOK_BYTES, Credentials, WebhookError, verify_webhook
from .remote import loader_from_env
from .worker_config import WorkerConfig

logger = logging.getLogger("hn_pipecat.server")

FRAMEWORK: Final = "pipecat"
PROBE_INTERVAL_OK: Final = 30.0
PROBE_INTERVAL_FAILING: Final = 5.0

DISPATCHER: Final = web.AppKey("dispatcher", Dispatcher)
STATE: Final = web.AppKey("state", dict)


def pipecat_version() -> str:
    try:
        return importlib.metadata.version("pipecat-ai")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class LiveKitRooms:
    """`RoomDirectory` backed by LiveKit's RoomService on this server."""

    def __init__(self, url: str, credentials: Credentials) -> None:
        self._url = url
        self._credentials = credentials
        self._api: api.LiveKitAPI | None = None

    def _client(self) -> api.LiveKitAPI:
        if self._api is None:
            self._api = api.LiveKitAPI(
                self._url,
                self._credentials.api_key,
                self._credentials.api_secret,
                failover=False,
            )
        return self._api

    async def list_rooms(self) -> list[models.Room]:
        response = await self._client().room.list_rooms(ListRoomsRequest())
        return list(response.rooms)

    async def list_participants(self, room: str) -> list[models.ParticipantInfo]:
        response = await self._client().room.list_participants(ListParticipantsRequest(room=room))
        return list(response.participants)

    async def aclose(self) -> None:
        if self._api is not None:
            await self._api.aclose()
            self._api = None


def create_app(
    config: WorkerConfig,
    env: Mapping[str, str],
    *,
    rooms: RoomDirectory | None = None,
    run_bot: BotRunner | None = None,
    probe: bool = True,
) -> web.Application:
    """The aiohttp application. `rooms` and `run_bot` are injectable for tests."""
    problems = config.problems()
    if rooms is None and not problems and config.livekit_url:
        rooms = LiveKitRooms(config.livekit_url, config.credentials[0])
    if run_bot is None:
        run_bot = _default_bot_runner(config, env)

    dispatcher = Dispatcher(
        agent_name=config.agent_name,
        room_prefix=config.room_prefix,
        max_rooms=config.max_rooms,
        run_bot=run_bot,
        rooms=rooms,
    )
    state: dict[str, Any] = {
        "ready": False,
        "reason": "; ".join(problems) or "not connected to LiveKit yet",
        "reconciled": False,
    }

    app = web.Application(client_max_size=MAX_WEBHOOK_BYTES)
    app[DISPATCHER] = dispatcher
    app[STATE] = state

    async def health(_: web.Request) -> web.Response:
        if not dispatcher.accepting:
            return web.Response(status=503, text="shutting down")
        if not state["ready"]:
            return web.Response(status=503, text=state["reason"])
        return web.Response(text="OK")

    async def worker(_: web.Request) -> web.Response:
        body = dispatcher.status()
        body.update(
            {
                "sdk_version": pipecat_version(),
                "framework": FRAMEWORK,
                "version": __version__,
            }
        )
        return web.json_response(body)

    async def webhook(request: web.Request) -> web.Response:
        if problems:
            return web.Response(status=503, text="worker is not configured")
        body = await request.read()
        try:
            event = verify_webhook(
                body, request.headers.get("Authorization"), list(config.credentials)
            )
        except WebhookError as err:
            logger.warning("rejected a webhook: %s", err.reason)
            return web.Response(status=401, text="invalid webhook")
        decision = dispatcher.handle(event)
        if decision.action != "ignore":
            logger.debug("webhook %s: %s %s", event.event, decision.action, decision.room)
        return web.Response(text="ok")

    app.router.add_get("/", health)
    app.router.add_get("/worker", worker)
    app.router.add_post("/webhook", webhook)

    async def on_startup(app: web.Application) -> None:
        if probe and rooms is not None:
            state["probe"] = asyncio.create_task(_probe_loop(dispatcher, rooms, state))

    async def on_shutdown(app: web.Application) -> None:
        await dispatcher.shutdown()

    async def on_cleanup(app: web.Application) -> None:
        task = state.get("probe")
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if isinstance(rooms, LiveKitRooms):
            await rooms.aclose()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.on_cleanup.append(on_cleanup)
    return app


async def probe_once(dispatcher: Dispatcher, rooms: RoomDirectory, state: dict[str, Any]) -> bool:
    """Ask LiveKit for its rooms; the first success also joins waiting callers."""
    try:
        await rooms.list_rooms()
    except Exception as err:
        reason = f"LiveKit is not reachable with the configured key ({type(err).__name__})"
        if state["ready"] or state["reason"] != reason:
            logger.warning("%s", reason)
        state["ready"] = False
        state["reason"] = reason
        return False
    if not state["ready"]:
        logger.info("connected to LiveKit; waiting for webhooks")
    state["ready"] = True
    state["reason"] = ""
    if not state["reconciled"]:
        state["reconciled"] = True
        try:
            joined = await dispatcher.reconcile()
        except Exception as err:
            logger.warning("could not check for waiting callers (%s)", type(err).__name__)
        else:
            if joined:
                logger.info("joined %d room(s) with a waiting caller", joined)
    return True


async def _probe_loop(dispatcher: Dispatcher, rooms: RoomDirectory, state: dict[str, Any]) -> None:
    while True:
        ok = await probe_once(dispatcher, rooms, state)
        await asyncio.sleep(PROBE_INTERVAL_OK if ok else PROBE_INTERVAL_FAILING)


def _default_bot_runner(config: WorkerConfig, env: Mapping[str, str]) -> BotRunner:
    loader = loader_from_env(env)

    async def run(job: RoomJob) -> None:
        if not config.livekit_url or not config.credentials:
            return
        await run_room_bot(
            job,
            BotContext(
                livekit_url=config.livekit_url,
                credentials=config.credentials[0],
                agent_name=config.agent_name,
                loader=loader,
                env=env,
                join_timeout=config.join_timeout,
            ),
        )

    return run
