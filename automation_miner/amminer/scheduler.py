"""A tiny cron scheduler.

Uses ``croniter`` when available and falls back to a plain daily timer, so a
missing optional dependency cannot stop nightly analysis from happening.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)

DEFAULT_SCHEDULE = "0 3 * * *"


def next_run_at(expression: str, after: float | None = None) -> float:
    """Next fire time for *expression*, as a unix timestamp."""
    after = after if after is not None else time.time()
    try:
        from croniter import croniter

        return float(croniter(expression, dt.datetime.fromtimestamp(after)).get_next(float))
    except Exception as err:  # noqa: BLE001 - bad expression or missing croniter
        _LOGGER.warning(
            "Could not parse schedule %r (%s); falling back to every 24 h", expression, err
        )
        return after + 86400.0


class Scheduler:
    """Calls *callback* on a cron schedule until stopped."""

    def __init__(self, expression: str, callback: Callable[[], object]) -> None:
        self.expression = expression or DEFAULT_SCHEDULE
        self.callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.next_run: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="amminer-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.next_run = next_run_at(self.expression)
            delay = max(self.next_run - time.time(), 1.0)
            _LOGGER.info(
                "Next scheduled analysis at %s (in %.1f h)",
                dt.datetime.fromtimestamp(self.next_run).isoformat(timespec="minutes"),
                delay / 3600.0,
            )
            if self._stop.wait(delay):
                return
            try:
                self.callback()
            except Exception as err:  # noqa: BLE001 - one bad run must not end the loop
                _LOGGER.exception("Scheduled analysis failed: %s", err)
