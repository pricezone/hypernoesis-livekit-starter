"""Shared test helpers: sign LiveKit webhooks the way the LiveKit server does."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from livekit import api

API_KEY = "test-api-key"
API_SECRET = "test-api-secret-that-is-long-enough-for-hs256"
OTHER_KEY = "other-api-key"
OTHER_SECRET = "other-api-secret-that-is-long-enough-for-hs256"

STANDARD, INGRESS, EGRESS, SIP, AGENT = 0, 1, 2, 3, 4


def event_body(event: str, room: str = "call-1", **extra: Any) -> bytes:
    payload: dict[str, Any] = {"event": event, "room": {"sid": "RM_x", "name": room}}
    payload.update(extra)
    return json.dumps(payload).encode()


def participant_joined(room: str, kind: str, identity: str = "caller", **extra: Any) -> bytes:
    return event_body(
        "participant_joined",
        room,
        participant={"sid": "PA_x", "identity": identity, "kind": kind},
        **extra,
    )


def sign(body: bytes, key: str = API_KEY, secret: str = API_SECRET) -> str:
    digest = base64.b64encode(hashlib.sha256(body).digest()).decode()
    return api.AccessToken(key, secret).with_sha256(digest).to_jwt()
