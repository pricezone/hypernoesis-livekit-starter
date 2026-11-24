"""`python -m hn_pipecat`: start the webhook receiver and status server on :8081."""

from __future__ import annotations

import logging
import os
import sys

from aiohttp import web
from loguru import logger as pipecat_logger

from .server import create_app
from .worker_config import worker_config_from_env

logger = logging.getLogger("hn_pipecat")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Pipecat logs through loguru, at DEBUG by default (every frame and
    # transcript): keep it at the same level as ours.
    pipecat_logger.remove()
    pipecat_logger.add(sys.stderr, level=level)


def main() -> None:
    level = (os.environ.get("HN_LOG_LEVEL") or "INFO").strip().upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        level = "INFO"
    configure_logging(level)

    config = worker_config_from_env(os.environ)
    for name in config.invalid:
        logger.warning("%s is invalid: using its default", name)
    for problem in config.problems():
        logger.error("%s", problem)
    logger.info(
        "agent %s joins %s, up to %d at a time; listening on %s:%d",
        config.agent_name,
        f"rooms starting with {config.room_prefix!r}" if config.room_prefix else "all rooms",
        config.max_rooms,
        config.http_host,
        config.http_port,
    )
    web.run_app(
        create_app(config, os.environ),
        host=config.http_host,
        port=config.http_port,
        access_log=None,
        print=None,
        shutdown_timeout=15.0,
    )


if __name__ == "__main__":
    main()
