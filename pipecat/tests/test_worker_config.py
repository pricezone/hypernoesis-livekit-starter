from hn_pipecat.livekit_auth import Credentials
from hn_pipecat.worker_config import worker_config_from_env


def test_defaults() -> None:
    config = worker_config_from_env({})
    assert config.livekit_url is None
    assert config.credentials == ()
    assert config.agent_name == "hypernoesis-pipecat"
    assert config.room_prefix == ""
    assert config.max_rooms == 8
    assert (config.http_host, config.http_port) == ("0.0.0.0", 8081)  # noqa: S104
    assert config.join_timeout == 60.0
    assert config.invalid == ()
    assert len(config.problems()) == 2


def test_full_environment() -> None:
    config = worker_config_from_env(
        {
            "LIVEKIT_URL": "wss://lk.example",
            "HN_AGENT_API_KEY": "key",
            "HN_AGENT_API_SECRET": "secret",
            "HN_AGENT_NAME": "front-desk",
            "HN_ROOM_PREFIX": " call- ",
            "HN_MAX_ROOMS": "3",
            "HN_HTTP_PORT": "9000",
            "HN_JOIN_TIMEOUT": "30",
        }
    )
    assert config.livekit_url == "wss://lk.example"
    assert config.credentials == (Credentials("key", "secret"),)
    assert config.agent_name == "front-desk"
    assert config.room_prefix == "call-"
    assert (config.max_rooms, config.http_port, config.join_timeout) == (3, 9000, 30.0)
    assert config.problems() == []


def test_invalid_values_fall_back_and_are_named() -> None:
    config = worker_config_from_env(
        {
            "LIVEKIT_URL": "lk.example",
            "HN_AGENT_NAME": "bad name!",
            "HN_MAX_ROOMS": "0",
            "HN_HTTP_PORT": "eighty",
            "HN_JOIN_TIMEOUT": "1",
        }
    )
    assert sorted(config.invalid) == [
        "HN_AGENT_NAME",
        "HN_HTTP_PORT",
        "HN_JOIN_TIMEOUT",
        "HN_MAX_ROOMS",
        "LIVEKIT_URL",
    ]
    assert config.livekit_url is None
    assert config.agent_name == "hypernoesis-pipecat"
    assert (config.max_rooms, config.http_port, config.join_timeout) == (8, 8081, 60.0)


def test_problems_name_both_key_variables() -> None:
    problems = worker_config_from_env({"LIVEKIT_URL": "ws://127.0.0.1:7880"}).problems()
    assert problems == [
        "no API key/secret: set LIVEKIT_API_KEY and LIVEKIT_API_SECRET "
        "(or HN_AGENT_API_KEY and HN_AGENT_API_SECRET)"
    ]
