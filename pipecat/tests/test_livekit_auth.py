import base64
import datetime
import hashlib

import jwt
import pytest
from livekit import api

from helpers import API_KEY, API_SECRET, OTHER_KEY, OTHER_SECRET, event_body, sign
from hn_pipecat.livekit_auth import (
    MAX_WEBHOOK_BYTES,
    Credentials,
    WebhookError,
    credentials_from_env,
    mint_room_token,
    verify_webhook,
)

CREDS = [Credentials(API_KEY, API_SECRET)]


def test_credentials_from_both_variable_names() -> None:
    assert credentials_from_env({}) == []
    assert credentials_from_env({"LIVEKIT_API_KEY": "k"}) == []  # secret missing
    assert credentials_from_env({"LIVEKIT_API_KEY": "k", "LIVEKIT_API_SECRET": "s"}) == [
        Credentials("k", "s")
    ]
    assert credentials_from_env({"HN_AGENT_API_KEY": "k2", "HN_AGENT_API_SECRET": "s2"}) == [
        Credentials("k2", "s2")
    ]
    both = {
        "LIVEKIT_API_KEY": "k",
        "LIVEKIT_API_SECRET": "s",
        "HN_AGENT_API_KEY": "k2",
        "HN_AGENT_API_SECRET": "s2",
    }
    assert credentials_from_env(both) == [Credentials("k", "s"), Credentials("k2", "s2")]
    same = {**both, "HN_AGENT_API_KEY": "k", "HN_AGENT_API_SECRET": "s"}
    assert credentials_from_env(same) == [Credentials("k", "s")]


def test_credentials_repr_hides_the_secret() -> None:
    assert API_SECRET not in repr(Credentials(API_KEY, API_SECRET))


def test_room_token_joins_one_room_as_an_agent() -> None:
    token = mint_room_token(CREDS[0], room="call-1", identity="hypernoesis-pipecat-1", name="bot")
    claims = api.TokenVerifier(API_KEY, API_SECRET).verify(token)
    assert claims.identity == "hypernoesis-pipecat-1"
    assert claims.kind == "agent"
    assert claims.video is not None
    assert claims.video.room == "call-1"
    assert claims.video.room_join is True
    assert claims.video.agent is True
    assert not claims.video.room_admin
    raw = jwt.decode(token, API_SECRET, algorithms=["HS256"], issuer=API_KEY)
    assert raw["exp"] - raw["nbf"] <= 3600


def test_valid_webhook_is_parsed() -> None:
    body = event_body("room_started", "call-42")
    event = verify_webhook(body, sign(body), CREDS)
    assert event.event == "room_started"
    assert event.room.name == "call-42"


def test_bearer_prefix_is_accepted() -> None:
    body = event_body("room_started")
    assert verify_webhook(body, f"Bearer {sign(body)}", CREDS).event == "room_started"


def test_webhook_signed_with_the_second_key_pair() -> None:
    body = event_body("room_started")
    both = [*CREDS, Credentials(OTHER_KEY, OTHER_SECRET)]
    assert verify_webhook(body, sign(body, OTHER_KEY, OTHER_SECRET), both).event == "room_started"


@pytest.mark.parametrize(
    ("make_header", "reason"),
    [
        (lambda body: None, "missing Authorization header"),
        (lambda body: "", "missing Authorization header"),
        (lambda body: "not-a-jwt", "malformed token"),
        (lambda body: sign(body, OTHER_KEY, OTHER_SECRET), "token issued by an unknown API key"),
        (lambda body: sign(body, API_KEY, "wrong-secret-that-is-long-enough-x"), "invalid token"),
        (lambda body: sign(b"another body"), "body hash mismatch"),
        (lambda body: api.AccessToken(API_KEY, API_SECRET).to_jwt(), "token has no body hash"),
    ],
)
def test_bad_webhooks_are_rejected(make_header, reason: str) -> None:
    body = event_body("room_started")
    with pytest.raises(WebhookError) as err:
        verify_webhook(body, make_header(body), CREDS)
    assert err.value.reason.startswith(reason)


def test_expired_webhook_is_rejected() -> None:
    body = event_body("room_started")
    digest = base64.b64encode(hashlib.sha256(body).digest()).decode()
    token = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_sha256(digest)
        .with_ttl(datetime.timedelta(minutes=-10))
        .to_jwt()
    )
    with pytest.raises(WebhookError) as err:
        verify_webhook(body, token, CREDS)
    assert err.value.reason == "invalid token (ExpiredSignatureError)"


def test_token_without_expiry_is_rejected() -> None:
    body = event_body("room_started")
    digest = base64.b64encode(hashlib.sha256(body).digest()).decode()
    token = jwt.encode({"iss": API_KEY, "sha256": digest}, API_SECRET, algorithm="HS256")
    with pytest.raises(WebhookError, match="invalid token"):
        verify_webhook(body, token, CREDS)


def test_non_event_body_is_rejected() -> None:
    body = b"\xff\xfe not json"
    with pytest.raises(WebhookError, match="not a webhook event"):
        verify_webhook(body, sign(body), CREDS)


def test_oversized_body_and_missing_credentials() -> None:
    body = b" " * (MAX_WEBHOOK_BYTES + 1)
    with pytest.raises(WebhookError, match="too large"):
        verify_webhook(body, sign(body), CREDS)
    small = event_body("room_started")
    with pytest.raises(WebhookError, match="no API key"):
        verify_webhook(small, sign(small), [])
