"""Centralized logging configuration for the Redis-Bedrock demo app."""

from __future__ import annotations

import logging
import os
from typing import Optional


DEFAULT_LOG_LEVEL = "INFO"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure application-wide logging in a single place.

    This should be called once during application startup. If logging is already
    configured (for example by the hosting environment), the call becomes a
    no-op.
    """
    if logging.getLogger().handlers:
        # Respect existing logging configuration to avoid duplicate handlers
        return

    log_level = level or os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
