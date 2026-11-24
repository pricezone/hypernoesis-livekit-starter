"""Fetch the agent settings from HyperNoesis, with a 60-second cache.

LiveKit Agents runs every session in its own process, so the cache has two
layers: memory (this process) and a small JSON file shared by all processes in
the container. When the settings server cannot be reached, the last good copy is
used if there is one, else the built-in defaults; either way the reason is
logged once per process and the server is not asked again for 60 seconds.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import aiohttp

from .config import DEFAULT_CONFIG, AgentConfig, parse_agent_config

logger = logging.getLogger("hn_agent.config")

CACHE_TTL_SECONDS: Final = 60.0
FETCH_TIMEOUT_SECONDS: Final = 3.0
MAX_RESPONSE_BYTES: Final = 256 * 1024
CACHE_FILE_NAME: Final = "hn-agent-config.json"


class ConfigFetchError(Exception):
    """The settings server did not return a usable document. `reason` is safe to log."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# (url, token) -> parsed JSON. Swappable in tests.
Fetcher = Callable[[str, str], Awaitable[Any]]


@dataclass
class _Entry:
    fetched_at: float
    data: Any  # raw JSON as served (re-parsed on read, so parser fixes apply)


class AgentConfigLoader:
    def __init__(
        self,
        url: str | None,
        token: str | None,
        *,
        cache_dir: Path | None = None,
        ttl: float = CACHE_TTL_SECONDS,
        fetcher: Fetcher | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._url = (url or "").strip() or None
        self._token = (token or "").strip() or None
        self._cache_path = (cache_dir / CACHE_FILE_NAME) if cache_dir else None
        self._ttl = ttl
        self._fetcher = fetcher or fetch_json
        self._clock = clock
        self._memory: _Entry | None = None
        self._failed_at: float | None = None
        self._logged: set[str] = set()

    async def get(self) -> AgentConfig:
        if self._url is None:
            self._log_once("unset", "HN_AGENT_CONFIG_URL is not set: using the built-in defaults")
            return DEFAULT_CONFIG
        if self._token is None:
            self._log_once("no-token", "HN_AGENT_TOKEN is not set: using the built-in defaults")
            return DEFAULT_CONFIG
        problem = url_problem(self._url)
        if problem is not None:
            self._log_once("bad-url", f"HN_AGENT_CONFIG_URL {problem}: using the built-in defaults")
            return DEFAULT_CONFIG

        now = self._clock()
        fresh = self._fresh_entry(now)
        if fresh is not None:
            return self._parse(fresh.data)

        if self._recently_failed(now):
            return self._fallback()

        try:
            data = await self._fetcher(self._url, self._token)
        except ConfigFetchError as err:
            self._record_failure(now)
            self._log_once(
                f"fetch:{err.reason}",
                f"agent settings unavailable ({err.reason}): using the "
                + ("last good copy" if self._stale_entry() else "built-in defaults"),
            )
            return self._fallback()

        entry = _Entry(fetched_at=now, data=data)
        self._memory = entry
        self._failed_at = None
        self._write_file({"fetchedAt": now, "data": data})
        return self._parse(data)

    # -- cache ---------------------------------------------------------------

    def _fresh_entry(self, now: float) -> _Entry | None:
        for entry in (self._memory, self._read_file_entry()):
            if entry is not None and 0 <= now - entry.fetched_at < self._ttl:
                self._memory = entry
                return entry
        return None

    def _stale_entry(self) -> _Entry | None:
        return self._memory or self._read_file_entry()

    def _recently_failed(self, now: float) -> bool:
        failed_at = self._failed_at
        if failed_at is None:
            raw = self._read_file()
            value = raw.get("failedAt") if raw else None
            failed_at = float(value) if isinstance(value, int | float) else None
        return failed_at is not None and 0 <= now - failed_at < self._ttl

    def _record_failure(self, now: float) -> None:
        self._failed_at = now
        raw = self._read_file() or {}
        raw["failedAt"] = now
        self._write_file(raw)

    def _fallback(self) -> AgentConfig:
        stale = self._stale_entry()
        return self._parse(stale.data) if stale is not None else DEFAULT_CONFIG

    def _read_file_entry(self) -> _Entry | None:
        raw = self._read_file()
        if not raw or "data" not in raw:
            return None
        fetched_at = raw.get("fetchedAt")
        if not isinstance(fetched_at, int | float):
            return None
        return _Entry(fetched_at=float(fetched_at), data=raw["data"])

    def _read_file(self) -> dict[str, Any] | None:
        if self._cache_path is None:
            return None
        try:
            with self._cache_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    def _write_file(self, payload: dict[str, Any]) -> None:
        if self._cache_path is None:
            return
        directory = self._cache_path.parent
        try:
            directory.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".hn-agent-config.", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                os.chmod(tmp, 0o600)
                os.replace(tmp, self._cache_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
        except OSError:
            self._log_once("cache-write", "could not write the settings cache file")

    # -- helpers -------------------------------------------------------------

    def _parse(self, data: Any) -> AgentConfig:
        result = parse_agent_config(data)
        if result.invalid:
            fields = ", ".join(sorted(set(result.invalid)))
            self._log_once(f"invalid:{fields}", f"agent settings: ignored invalid fields: {fields}")
        return result.config

    def _log_once(self, key: str, message: str) -> None:
        if key in self._logged:
            return
        self._logged.add(key)
        logger.warning(message)


def url_problem(url: str) -> str | None:
    """Why the settings URL must not receive the bearer token, or None if it may."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "is not a valid URL"
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "is not an http(s) URL"
    if parts.username or parts.password:
        return "must not contain credentials"
    if parts.scheme == "http" and not is_private_host(parts.hostname):
        return "must use https outside a private network"
    return None


def is_private_host(host: str) -> bool:
    """Loopback, private/link-local addresses and single-label container names."""
    if host == "localhost" or ("." not in host and ":" not in host):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


async def fetch_json(url: str, token: str) -> Any:
    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(url, headers=headers, allow_redirects=False) as resp,
        ):
            if resp.status != 200:
                raise ConfigFetchError(f"HTTP {resp.status}")
            if (resp.content_length or 0) > MAX_RESPONSE_BYTES:
                raise ConfigFetchError("response too large")
            body = bytearray()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ConfigFetchError("response too large")
    except TimeoutError as err:
        raise ConfigFetchError("timeout") from err
    except aiohttp.ClientError as err:
        raise ConfigFetchError(type(err).__name__) from err
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise ConfigFetchError("invalid JSON") from err
    if not isinstance(data, dict):
        raise ConfigFetchError("not a JSON object")
    return data


def loader_from_env(env: Mapping[str, str]) -> AgentConfigLoader:
    cache_dir = env.get("HN_CACHE_DIR") or tempfile.gettempdir()
    return AgentConfigLoader(
        env.get("HN_AGENT_CONFIG_URL"),
        env.get("HN_AGENT_TOKEN"),
        cache_dir=Path(cache_dir),
    )
