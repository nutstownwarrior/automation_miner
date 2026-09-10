"""The whole pipeline against history that looks like a real recorder.

The shared fixture is clean: every entity always has a state, nothing ever
flaps, and every human change carries a ``context_user_id``.  Roughly the whole
suite runs on it, so the most heavily exercised path through this project was
the easy case, and the degraded paths the docs describe were covered only by
small hand-built unit tests.

``messy=True`` adds what a real instance is full of - restarts leaving
``unavailable``/``unknown``, flapping contacts, and human changes with no user
id at all - while injecting the same four known patterns, so they can be
asserted to survive it.
"""

from __future__ import annotations

import datetime as dt

import pytest
from amminer.config import Options
from amminer.discovery.ha_config import HAConfig
from amminer.pipeline import run_analysis
from amminer.recorderdb.models import Cause
from amminer.testing.synthetic import build_default_fixture

MESSY_END = dt.datetime(2024, 5, 1, 12, 0, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def messy_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("messy")
    (directory / ".storage").mkdir(parents=True)
    (directory / "configuration.yaml").write_text("recorder:\n  purge_keep_days: 60\n")
    truth = build_default_fixture(
        directory / "home-assistant_v2.db",
        days=45,
        end=MESSY_END,
        seed=20240501,
        tz=dt.UTC,
        messy=True,
    )
    return directory, truth


def _run(directory, store, client=None):
    options = Options(ha_config_dir=str(directory), state_dir=str(directory))
    return run_analysis(options, store, client, HAConfig(directory))


def test_the_generator_really_does_produce_a_mess(messy_dir):
    """Guard the guard: a clean fixture would make every test below vacuous."""
    from amminer.discovery.recorder import (
        RecorderInfo,
        create_recorder_engine,
        probe,
        sqlite_readonly_url,
    )
    from amminer.recorderdb.queries import RecorderQueries

    directory, truth = messy_dir
    engine = create_recorder_engine(
        sqlite_readonly_url(directory / "home-assistant_v2.db")
    )
    try:
        queries = RecorderQueries(engine, probe(engine, RecorderInfo(dialect="sqlite")))
        changes = queries.state_changes(truth.start_ts - 1, truth.end_ts + 1)
    finally:
        engine.dispose()

    states = {c.state.lower() for c in changes}
    assert "unavailable" in states
    assert "unknown" in states
    assert any(c.context_user_id is None for c in changes)

    # And bursts: several changes on one entity inside a few seconds.
    by_entity: dict[str, list[float]] = {}
    for change in changes:
        by_entity.setdefault(change.entity_id, []).append(change.ts)
    bursts = sum(
        1
        for stamps in by_entity.values()
        for a, b in zip(sorted(stamps), sorted(stamps)[1:], strict=False)
        if b - a < 5
    )
    assert bursts > 10, f"only {bursts} rapid changes; the fixture is not messy"


def test_the_injected_habits_survive_a_messy_recorder(messy_dir, store):
    directory, _truth = messy_dir
    report, _candidates = _run(directory, store)

    assert report.status in ("ok", "partial"), report.error
    assert report.error is None
    assert report.surfaced > 0

    targets = {
        action["entity_id"]
        for suggestion in store.list_suggestions(status="new")
        for action in (suggestion["payload"].get("actions") or [])
        if action.get("entity_id")
    }
    assert "light.kitchen" in targets, "the 06:30 habit did not survive the mess"


def test_unavailable_readings_never_become_evidence(messy_dir, store):
    """A dropout is the absence of a reading, not a reading of 'unavailable'."""
    directory, _truth = messy_dir
    _run(directory, store)

    for suggestion in store.list_suggestions(status="new"):
        payload = suggestion["payload"]
        for part in (payload.get("triggers") or []) + (payload.get("conditions") or []):
            for key in ("to_state", "from_state", "state"):
                value = str(part.get(key) or "").lower()
                assert value not in ("unavailable", "unknown"), payload["title"]


def test_changes_without_a_user_id_are_not_claimed_as_human(messy_dir, store):
    """We cannot know who did it, and saying "a person" would be a guess."""
    directory, truth = messy_dir
    report, _candidates = _run(directory, store)

    # The window has both kinds, so this is a real distinction being drawn.
    assert report.causality.get(Cause.HUMAN.value, 0) > 0
    assert report.causality.get(Cause.UNKNOWN.value, 0) > 0
    # And nothing is silently filed under "device" to make it go away.
    assert sum(report.causality.values()) == report.state_rows


def test_a_flapping_entity_does_not_become_a_suggestion(messy_dir, store):
    """Bursts are collapsed for the reader and counted against the rule."""
    directory, _truth = messy_dir
    _run(directory, store)

    for suggestion in store.list_suggestions(status="new"):
        backtest = suggestion["payload"].get("backtest") or {}
        if not backtest:
            continue
        assert backtest["passed"] is True
        # Whatever was surfaced kept its bursts within the limit.
        assert backtest.get("burst_fires", 0) <= 3 * max(backtest.get("total_fires", 0), 1)
