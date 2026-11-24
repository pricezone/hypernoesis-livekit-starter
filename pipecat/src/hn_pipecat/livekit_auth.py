"""LiveKit credentials: read them, mint room tokens, verify webhooks.

The platform injects the server's API key and secret under one of two names:
`LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` (the LiveKit SDK names) or
`HN_AGENT_API_KEY` / `HN_AGENT_API_SECRET`. Both are read; when both pairs are
set and differ, tokens are minted with the first and webhooks signed with
either are accepted.

A LiveKit webhook carries a JWT in its `Authorization` header, signed (HS256)
with the API secret, issued by the API key, valid for a few minutes and holding
the base64 SHA-256 of the exact request body in its `sha256` claim. The body is
only trusted when all of that checks out.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import jwt
from google.protobuf.json_format import Parse, ParseError
from livekit import api
from livekit.protocol.webhook import WebhookEvent

KEY_ENV_PAIRS: Final = (
    ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"),
    ("HN_AGENT_API_KEY", "HN_AGENT_API_SECRET"),
)
MAX_WEBHOOK_BYTES: Final = 1024 * 1024
TOKEN_TTL: Final = datetime.timedelta(hours=1)


@dataclass(frozen=True)
class Credentials:
    api_key: str
    api_secret: str

    def __repr__(self) -> str:  # never print the secret
        return f"Credentials(api_key={self.api_key!r})"


class WebhookError(Exception):
    """The request is not a valid LiveKit webhook. `reason` is safe to log."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def credentials_from_env(env: Mapping[str, str]) -> list[Credentials]:
    """Every complete key/secret pair in the environment, without duplicates."""
    found: list[Credentials] = []
    for key_name, secret_name in KEY_ENV_PAIRS:
        key = (env.get(key_name) or "").strip()
        secret = (env.get(secret_name) or "").strip()
        if key and secret:
            creds = Credentials(key, secret)
            if creds not in found:
                found.append(creds)
    return found


def mint_room_token(
    creds: Credentials,
    *,
    room: str,
    identity: str,
    name: str,
    ttl: datetime.timedelta = TOKEN_TTL,
) -> str:
    """A token that lets the bot join one room as an agent participant."""
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        agent=True,
    )
    return (
        api.AccessToken(creds.api_key, creds.api_secret)
        .with_identity(identity)
        .with_name(name)
        .with_kind("agent")
        .with_grants(grants)
        .with_ttl(ttl)
        .to_jwt()
    )


def verify_webhook(
    body: bytes, authorization: str | None, credentials: list[Credentials]
) -> WebhookEvent:
    """Check the signature and the body hash, then parse the event."""
    if not credentials:
        raise WebhookError("no API key/secret configured")
    if len(body) > MAX_WEBHOOK_BYTES:
        raise WebhookError("body too large")
    token = (authorization or "").strip()
    if token[:7].lower() == "bearer ":
        token = token[7:].strip()
    if not token:
        raise WebhookError("missing Authorization header")

    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError as err:
        raise WebhookError("malformed token") from err
    issuer = unverified.get("iss") if isinstance(unverified, dict) else None
    creds = next((c for c in credentials if c.api_key == issuer), None)
    if creds is None:
        raise WebhookError("token issued by an unknown API key")

    try:
        claims = api.TokenVerifier(creds.api_key, creds.api_secret).verify(token)
    except (jwt.PyJWTError, ValueError, TypeError) as err:
        raise WebhookError(f"invalid token ({type(err).__name__})") from err
    if not claims.sha256:
        raise WebhookError("token has no body hash")
    try:
        expected = base64.b64decode(claims.sha256, validate=True)
    except ValueError as err:
        raise WebhookError("token has a malformed body hash") from err
    if not hmac.compare_digest(hashlib.sha256(body).digest(), expected):
        raise WebhookError("body hash mismatch")

    try:
        return Parse(body.decode("utf-8"), WebhookEvent(), ignore_unknown_fields=True)
    except (UnicodeDecodeError, ParseError) as err:
        raise WebhookError("body is not a webhook event") from err
