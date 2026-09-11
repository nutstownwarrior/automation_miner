"""Add-on entrypoint: start the ingress web UI and the nightly analysis."""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

from .config import Options
from .logging_setup import setup_logging
from .runner import Runner
from .scheduler import Scheduler
from .store import Store
from .version import __version__
from .web.app import create_app

_LOGGER = logging.getLogger(__name__)

DEFAULT_PORT = 8099

#: How long shutdown waits for an analysis to finish writing before giving up.
#: An analysis over a real recorder database takes minutes, not seconds.
SHUTDOWN_GRACE_SECONDS = 120.0


def build_everything(options: Options | None = None):
    """Wire up store, runner, scheduler and app.  Shared with the tests."""
    options = options or Options.load()
    setup_logging(options.log_level)
    _LOGGER.info("Automation Miner %s starting", __version__)

    state_dir = Path(options.state_dir)
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        _LOGGER.warning("Cannot create %s (%s); falling back to /data", state_dir, err)
        state_dir = Path("/data")
        state_dir.mkdir(parents=True, exist_ok=True)

    store = Store(state_dir / "automation_miner.db")
    interrupted = store.close_interrupted_runs()
    if interrupted:
        _LOGGER.info("Marked %d unfinished run(s) as interrupted", interrupted)
    runner = Runner(options, store)
    app = create_app(
        options,
        store,
        runner=runner,
        client=runner.client,
        ingress_only=os.environ.get("AMMINER_ALLOW_ANY_HOST", "") != "1",
    )
    return options, store, runner, app


def main() -> int:
    options, store, runner, app = build_everything()

    scheduler = Scheduler(options.schedule, runner.run_now)
    scheduler.start()

    initial: threading.Thread | None = None
    if options.run_on_start:
        initial = threading.Thread(target=runner.run_now, name="amminer-initial", daemon=True)
        initial.start()

    import uvicorn

    port = int(os.environ.get("AMMINER_PORT", DEFAULT_PORT))
    _LOGGER.info("Serving the ingress UI on port %d", port)
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning", access_log=False)
    finally:
        # Order matters.  Stop scheduling, then let whatever is mid-run finish
        # writing, and only then close the database.  A Supervisor restart is
        # an ordinary event, an analysis takes far longer than a few seconds,
        # and closing the store underneath one loses that run's work and leaves
        # its row saying "running" for good.
        scheduler.stop(timeout=SHUTDOWN_GRACE_SECONDS)
        if initial is not None:
            initial.join(timeout=SHUTDOWN_GRACE_SECONDS)
        if runner.wait_until_idle(SHUTDOWN_GRACE_SECONDS):
            store.close()
        else:
            _LOGGER.warning(
                "An analysis is still running after %.0fs; leaving the database open "
                "so it is not closed mid-write. The run will be marked interrupted "
                "on the next start.",
                SHUTDOWN_GRACE_SECONDS,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
