"""Worker settings from the environment: LiveKit server, name, dispatch, HTTP port."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from .livekit_auth import Credentials, credentials_from_env

DEFAULT_AGENT_NAME: Final = "hypernoesis-pipecat"
DEFAULT_MAX_ROOMS: Final = 8
MAX_ROOMS_LIMIT: Final = 100
DEFAULT_HTTP_HOST: Final = "0.0.0.0"  # noqa: S104 - the container's status/webhook port
DEFAULT_HTTP_PORT: Final = 8081
DEFAULT_JOIN_TIMEOUT: Final = 60.0

# Participant identities are built from the agent name, so keep it identity-safe.
_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


@dataclass(frozen=True)
class WorkerConfig:
    livekit_url: str | None
    credentials: tuple[Credentials, ...]
    agent_name: str = DEFAULT_AGENT_NAME
    room_prefix: str = ""
    max_rooms: int = DEFAULT_MAX_ROOMS
    http_host: str = DEFAULT_HTTP_HOST
    http_port: int = DEFAULT_HTTP_PORT
    join_timeout: float = DEFAULT_JOIN_TIMEOUT
    # Names of variables that were set but invalid (never their values).
    invalid: tuple[str, ...] = field(default_factory=tuple)

    def problems(self) -> list[str]:
        """Why the worker cannot serve rooms; empty when it can."""
        found: list[str] = []
        if not self.livekit_url:
            found.append("LIVEKIT_URL is not set")
        if not self.credentials:
            found.append(
                "no API key/secret: set LIVEKIT_API_KEY and LIVEKIT_API_SECRET "
                "(or HN_AGENT_API_KEY and HN_AGENT_API_SECRET)"
            )
        return found


def worker_config_from_env(env: Mapping[str, str]) -> WorkerConfig:
    invalid: list[str] = []

    url = (env.get("LIVEKIT_URL") or "").strip() or None
    if url is not None and not url.startswith(("ws://", "wss://", "http://", "https://")):
        invalid.append("LIVEKIT_URL")
        url = None

    name = (env.get("HN_AGENT_NAME") or "").strip() or DEFAULT_AGENT_NAME
    if not _NAME_RE.match(name):
        invalid.append("HN_AGENT_NAME")
        name = DEFAULT_AGENT_NAME

    return WorkerConfig(
        livekit_url=url,
        credentials=tuple(credentials_from_env(env)),
        agent_name=name,
        room_prefix=(env.get("HN_ROOM_PREFIX") or "").strip(),
        max_rooms=_int(env, "HN_MAX_ROOMS", DEFAULT_MAX_ROOMS, 1, MAX_ROOMS_LIMIT, invalid),
        http_host=(env.get("HN_HTTP_HOST") or "").strip() or DEFAULT_HTTP_HOST,
        http_port=_int(env, "HN_HTTP_PORT", DEFAULT_HTTP_PORT, 1, 65535, invalid),
        join_timeout=float(
            _int(env, "HN_JOIN_TIMEOUT", int(DEFAULT_JOIN_TIMEOUT), 5, 3600, invalid)
        ),
        invalid=tuple(invalid),
    )


def _int(
    env: Mapping[str, str], name: str, default: int, low: int, high: int, invalid: list[str]
) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        invalid.append(name)
        return default
    if not low <= value <= high:
        invalid.append(name)
        return default
    return value
