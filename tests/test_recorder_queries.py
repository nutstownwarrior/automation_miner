"""The recorder SQL, against the real normalised schema."""

from __future__ import annotations

import os

import pytest
from amminer.discovery.recorder import RecorderInfo, probe
from amminer.recorderdb.queries import RecorderQueries, decode_context
from sqlalchemy import create_engine


def test_state_changes_are_joined_and_ordered(raw_changes):
    assert raw_changes
    timestamps = [c.ts for c in raw_changes]
    assert timestamps == sorted(timestamps)
    assert all("." in c.entity_id for c in raw_changes)


def test_old_state_is_resolved_from_old_state_id(raw_changes):
    kitchen = [c for c in raw_changes if c.entity_id == "light.kitchen"]
    assert kitchen
    # The very first row has no predecessor; later ones must.
    assert kitchen[0].old_state is None
    assert any(c.old_state is not None for c in kitchen[1:])
    transitions = [c for c in kitchen if c.is_transition]
    assert transitions and all(c.old_state != c.state for c in transitions)


def test_attributes_are_decoded(raw_changes):
    temps = [c for c in raw_changes if c.entity_id == "sensor.outdoor_temperature"]
    assert temps
    assert any(c.attributes.get("unit_of_measurement") == "°C" for c in temps)


def test_binary_context_columns_decoded(raw_changes):
    with_context = [c for c in raw_changes if c.context_id]
    assert with_context
    # Binary ULIDs become stable hex strings, not bytes reprs.
    assert all(isinstance(c.context_id, str) for c in with_context)
    assert any(c.context_user_id for c in raw_changes)


def test_decode_context_handles_all_shapes():
    assert decode_context(None) is None
    assert decode_context("") is None
    assert decode_context("abc") == "abc"
    assert decode_context(b"\x01\x02") == "0102"
    assert decode_context(bytearray(b"\xff")) == "ff"


def test_entity_filter(queries, fixture_db):
    _path, truth = fixture_db
    rows = queries.state_changes(
        truth.start_ts - 1, truth.end_ts + 86400, entity_ids=["light.kitchen"]
    )
    assert rows
    assert {c.entity_id for c in rows} == {"light.kitchen"}


def test_time_window_is_respected(queries, fixture_db):
    _path, truth = fixture_db
    midpoint = (truth.start_ts + truth.end_ts) / 2
    rows = queries.state_changes(midpoint, truth.end_ts + 86400)
    assert rows
    assert all(c.ts >= midpoint for c in rows)


def test_events_are_read_with_data(raw_events):
    assert raw_events
    triggered = [e for e in raw_events if e.event_type == "automation_triggered"]
    assert triggered
    assert triggered[0].entity_id == "automation.bedtime_dim"
    assert triggered[0].context_id


def test_entity_ids_lists_everything(queries):
    ids = set(queries.entity_ids())
    assert {"light.kitchen", "switch.heater", "person.alex"} <= ids


def test_long_term_statistics_readable(queries):
    ids = queries.statistic_ids()
    assert "sensor.outdoor_temperature" in ids
    rows = queries.statistics(["sensor.outdoor_temperature"])
    assert rows
    assert rows[0]["mean"] is not None
    assert rows == sorted(rows, key=lambda r: r["start_ts"])


def test_oversized_attributes_are_dropped(tmp_path):
    """The recorder caps attributes at 16 KiB; a larger blob must not crash us."""
    from amminer.recorderdb.queries import _decode_attributes

    assert _decode_attributes('{"a": 1}') == {"a": 1}
    assert _decode_attributes("[1,2,3]") == {}
    assert _decode_attributes("not json") == {}
    assert _decode_attributes('{"big": "' + "x" * 20000 + '"}') == {}


# --- the same queries against MariaDB, when one is available ---------
MYSQL_URL = os.environ.get("AMMINER_TEST_MYSQL_URL")


@pytest.mark.mariadb
@pytest.mark.skipif(not MYSQL_URL, reason="set AMMINER_TEST_MYSQL_URL to run the MariaDB dialect test")
def test_same_queries_run_on_mariadb(fixture_db):
    """Load the fixture into MariaDB and assert the SQL is dialect-agnostic."""
    from amminer.testing.mysql_loader import load_fixture_into_mysql

    path, truth = fixture_db
    engine = create_engine(MYSQL_URL)
    load_fixture_into_mysql(path, engine)
    info = probe(engine, RecorderInfo(dialect="mysql"))
    assert info.normalised_schema
    queries = RecorderQueries(engine, info)
    rows = queries.state_changes(truth.start_ts - 1, truth.end_ts + 86400)
    assert rows
    assert any(r.context_user_id for r in rows)
    engine.dispose()
