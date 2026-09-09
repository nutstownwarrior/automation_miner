"""Add-on entrypoint: start the ingress web UI and the nightly analysis."""

from __future__ import annotations

import logging
import os
import sys
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

    if options.run_on_start:
        import threading

        threading.Thread(target=runner.run_now, name="amminer-initial", daemon=True).start()

    import uvicorn

    port = int(os.environ.get("AMMINER_PORT", DEFAULT_PORT))
    _LOGGER.info("Serving the ingress UI on port %d", port)
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning", access_log=False)
    finally:
        scheduler.stop()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
