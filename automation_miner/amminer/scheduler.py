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

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop scheduling and wait for a callback in flight.

        Returns False when the callback is still running, so the caller can
        decide what to do rather than assuming it finished.  The old version
        joined for five seconds and then set ``_thread = None`` regardless -
        which reads as "stopped" for a run that is still writing to the
        database the caller is about to close.
        """
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        if thread.is_alive():
            _LOGGER.warning(
                "Scheduled analysis is still running after %.0fs; not abandoning it", timeout
            )
            return False
        self._thread = None
        return True

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
