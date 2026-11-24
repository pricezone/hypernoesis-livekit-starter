"""`python -m hn_agent start|dev|console` (LiveKit Agents CLI)."""

from __future__ import annotations

from livekit.agents import cli

from .agent import server


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
