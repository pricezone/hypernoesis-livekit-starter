import json
import logging
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hn_agent.config import DEFAULT_CONFIG
from hn_agent.remote import (
    CACHE_FILE_NAME,
    AgentConfigLoader,
    ConfigFetchError,
    fetch_json,
    loader_from_env,
    url_problem,
)

URL = "https://settings.example/agent-config"
TOKEN = "test-agent-token"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class FakeFetcher:
    def __init__(self, *results: Any) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, url: str, token: str) -> Any:
        self.calls.append((url, token))
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result


def make_loader(
    fetcher: FakeFetcher, clock: FakeClock, cache_dir: Path | None = None, url: str = URL
) -> AgentConfigLoader:
    return AgentConfigLoader(url, TOKEN, cache_dir=cache_dir, fetcher=fetcher, clock=clock)


async def test_unset_url_uses_defaults_and_logs_once(caplog: pytest.LogCaptureFixture) -> None:
    loader = AgentConfigLoader(None, None)
    with caplog.at_level(logging.WARNING, logger="hn_agent.config"):
        assert await loader.get() == DEFAULT_CONFIG
        assert await loader.get() == DEFAULT_CONFIG
    assert [r.message for r in caplog.records] == [
        "HN_AGENT_CONFIG_URL is not set: using the built-in defaults"
    ]


async def test_missing_token_uses_defaults() -> None:
    fetcher = FakeFetcher({"language": "en"})
    loader = AgentConfigLoader(URL, "", fetcher=fetcher)
    assert await loader.get() == DEFAULT_CONFIG
    assert fetcher.calls == []


async def test_token_is_never_sent_over_public_http() -> None:
    fetcher = FakeFetcher({"language": "en"})
    loader = make_loader(fetcher, FakeClock(), url="http://settings.example/agent-config")
    assert await loader.get() == DEFAULT_CONFIG
    assert fetcher.calls == []


def test_url_rules() -> None:
    assert url_problem(URL) is None
    assert url_problem("http://127.0.0.1:3000/x") is None
    assert url_problem("http://10.1.2.3/x") is None
    assert url_problem("http://console:7890/x") is None  # container name
    assert url_problem("http://settings.example/x") is not None
    assert url_problem("https://u:p@settings.example/x") is not None
    assert url_problem("ftp://settings.example/x") is not None


async def test_fetch_result_is_cached_for_60_seconds() -> None:
    clock = FakeClock()
    fetcher = FakeFetcher({"language": "en"}, {"language": "el"})
    loader = make_loader(fetcher, clock)

    assert (await loader.get()).language == "en"
    clock.now += 59
    assert (await loader.get()).language == "en"
    assert len(fetcher.calls) == 1

    clock.now += 2
    assert (await loader.get()).language == "el"
    assert len(fetcher.calls) == 2
    assert fetcher.calls[0] == (URL, TOKEN)


async def test_unreachable_server_uses_defaults_logs_once_and_backs_off(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    fetcher = FakeFetcher(ConfigFetchError("timeout"))
    loader = make_loader(fetcher, clock)
    with caplog.at_level(logging.WARNING, logger="hn_agent.config"):
        assert await loader.get() == DEFAULT_CONFIG
        clock.now += 30
        assert await loader.get() == DEFAULT_CONFIG
        assert len(fetcher.calls) == 1  # no retry inside the 60 s window
        clock.now += 31
        assert await loader.get() == DEFAULT_CONFIG
        assert len(fetcher.calls) == 2
    messages = [r.message for r in caplog.records]
    assert messages == ["agent settings unavailable (timeout): using the built-in defaults"]
    assert TOKEN not in caplog.text


async def test_unreachable_server_keeps_the_last_good_copy() -> None:
    clock = FakeClock()
    fetcher = FakeFetcher({"language": "en"}, ConfigFetchError("HTTP 502"))
    loader = make_loader(fetcher, clock)
    assert (await loader.get()).language == "en"
    clock.now += 120
    assert (await loader.get()).language == "en"
    assert len(fetcher.calls) == 2


async def test_file_cache_is_shared_between_processes(tmp_path: Path) -> None:
    clock = FakeClock()
    first = FakeFetcher({"language": "en", "greeting": "Hi"})
    assert (await make_loader(first, clock, tmp_path).get()).greeting == "Hi"

    # A second process (new loader, empty memory) reads the file instead of fetching.
    second = FakeFetcher({"language": "el"})
    clock.now += 10
    assert (await make_loader(second, clock, tmp_path).get()).greeting == "Hi"
    assert second.calls == []

    cache_file = tmp_path / CACHE_FILE_NAME
    assert cache_file.stat().st_mode & 0o777 == 0o600
    assert json.loads(cache_file.read_text())["data"]["greeting"] == "Hi"

    clock.now += 60
    assert (await make_loader(second, clock, tmp_path).get()).language == "el"
    assert len(second.calls) == 1


async def test_failure_backoff_is_shared_through_the_file(tmp_path: Path) -> None:
    clock = FakeClock()
    failing = FakeFetcher(ConfigFetchError("timeout"))
    await make_loader(failing, clock, tmp_path).get()
    other = FakeFetcher({"language": "en"})
    clock.now += 5
    assert await make_loader(other, clock, tmp_path).get() == DEFAULT_CONFIG
    assert other.calls == []


async def test_invalid_fields_are_logged_by_name_only(caplog: pytest.LogCaptureFixture) -> None:
    fetcher = FakeFetcher({"language": "xx-secret-value", "greeting": "Hello"})
    loader = make_loader(fetcher, FakeClock())
    with caplog.at_level(logging.WARNING, logger="hn_agent.config"):
        config = await loader.get()
    assert config.greeting == "Hello"
    assert config.language == "auto"
    assert "language" in caplog.text
    assert "xx-secret-value" not in caplog.text


def test_loader_from_env(tmp_path: Path) -> None:
    loader = loader_from_env(
        {"HN_AGENT_CONFIG_URL": URL, "HN_AGENT_TOKEN": TOKEN, "HN_CACHE_DIR": str(tmp_path)}
    )
    assert loader._url == URL
    assert loader._cache_path == tmp_path / CACHE_FILE_NAME


# -- the real HTTP client against a local server -------------------------------


async def _elsewhere(_: web.Request) -> web.Response:
    return web.json_response({"language": "en"})


def _redirect() -> web.Response:
    raise web.HTTPFound("/elsewhere")


async def _serve(handler: Any) -> TestServer:
    app = web.Application()
    app.router.add_get("/agent-config", handler)
    app.router.add_get("/elsewhere", _elsewhere)
    server = TestServer(app)
    await server.start_server()
    return server


async def test_fetch_json_sends_the_bearer_token() -> None:
    seen: dict[str, str] = {}

    async def handler(request: web.Request) -> web.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        return web.json_response({"language": "en"})

    server = await _serve(handler)
    try:
        data = await fetch_json(str(server.make_url("/agent-config")), TOKEN)
    finally:
        await server.close()
    assert data == {"language": "en"}
    assert seen["auth"] == f"Bearer {TOKEN}"


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (lambda: web.Response(status=401), "HTTP 401"),
        (lambda: web.Response(text="{not json", content_type="application/json"), "invalid JSON"),
        (lambda: web.json_response([1, 2]), "not a JSON object"),
        (lambda: web.Response(text="x" * 300_000), "response too large"),
        # A redirect is refused: the token must not follow it to another URL.
        (_redirect, "HTTP 302"),
    ],
)
async def test_fetch_json_failures(response: Any, reason: str) -> None:
    async def handler(_: web.Request) -> web.StreamResponse:
        return response()

    server = await _serve(handler)
    try:
        with pytest.raises(ConfigFetchError) as err:
            await fetch_json(str(server.make_url("/agent-config")), TOKEN)
    finally:
        await server.close()
    assert err.value.reason == reason


async def test_fetch_json_connection_refused() -> None:
    with pytest.raises(ConfigFetchError):
        await fetch_json("http://127.0.0.1:9/agent-config", TOKEN)
