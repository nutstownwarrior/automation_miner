"""Logging configuration matching Home Assistant add-on conventions."""

from __future__ import annotations

import logging
import sys

LEVELS = {
    "trace": logging.DEBUG,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.CRITICAL,
}


def setup_logging(level: str = "info") -> None:
    resolved = LEVELS.get(str(level).lower(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)

    # Third-party libraries are far too chatty at INFO.
    for noisy in ("httpx", "httpcore", "uvicorn.access", "sqlalchemy.engine", "numba"):
        logging.getLogger(noisy).setLevel(max(resolved, logging.WARNING))
    logging.getLogger("amminer").setLevel(resolved)
