"""Validate the mining logic against public smart-home dataset formats.

The CASAS / ARAS / Kasteren archives are not redistributable, so these tests run
against data generated in exactly those on-disk formats.  Point
``AMMINER_CASAS_FILE`` at a real CASAS ``data`` file to run the same assertions
on the genuine archive.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest
from amminer.config import Options
from amminer.miners import association, time_of_day
from amminer.recorderdb.models import Cause
from amminer.testing.datasets import (
    load_dataset,
    parse_aras,
    parse_casas,
    parse_kasteren,
    synthesise_casas_lines,
)

UTC = dt.UTC


# --- format parsing -----------------------------------------------------
def test_casas_lines_map_to_the_internal_schema():
    lines = [
        "2008-02-27 12:43:27.416392\tM017\tON\tMeal_Preparation begin",
        "2008-02-27 12:44:01.100000\tL003\tON",
        "2008-02-27 12:50:00.000000\tD002\tOPEN",
        "2008-02-27 12:55:00.000000\tT001\t21.5",
        "2008-02-27 13:10:00.000000\tM017\tOFF\tMeal_Preparation end",
    ]
    dataset = parse_casas(lines)
    by_entity = {c.entity_id: c for c in dataset.changes}

    assert "binary_sensor.m017" in by_entity
    assert "light.l003" in by_entity
    assert "binary_sensor.d002" in by_entity
    assert "sensor.t001" in by_entity
    # A door opening is a person acting; motion and temperature are not.
    assert by_entity["binary_sensor.d002"].cause is Cause.HUMAN
    assert by_entity["light.l003"].cause is Cause.HUMAN
    assert by_entity["binary_sensor.m017"].cause is Cause.DEVICE
    assert by_entity["sensor.t001"].cause is Cause.DEVICE
    assert by_entity["binary_sensor.d002"].state == "on"  # OPEN normalised
    assert dataset.activities == [
        (
            dt.datetime(2008, 2, 27, 12, 43, 27, 416392, tzinfo=UTC).timestamp(),
            dt.datetime(2008, 2, 27, 13, 10, tzinfo=UTC).timestamp(),
            "Meal_Preparation",
        )
    ]


def test_casas_ignores_malformed_lines():
    dataset = parse_casas(["", "garbage", "not-a-date M1 ON", "2008-02-27 12:00:00 M1 ON"])
    assert len(dataset.changes) == 1


def test_aras_columns_become_state_changes():
    header = ["Ph1", "Ph2", "Force"]
    rows = [
        "0 0 0 1 1",
        "1 0 0 1 1",  # Ph1 turns on
        "1 1 0 2 1",  # Ph2 turns on, activity changes
        "0 1 0 2 1",  # Ph1 turns off
    ]
    dataset = parse_aras(rows, sensor_names=header)
    transitions = [(c.entity_id, c.state) for c in dataset.changes]
    assert ("binary_sensor.ph1", "on") in transitions
    assert ("binary_sensor.ph2", "on") in transitions
    assert ("binary_sensor.ph1", "off") in transitions
    assert dataset.activities  # the label change was recorded
    assert all(c.cause is Cause.HUMAN for c in dataset.changes)


def test_kasteren_intervals_become_on_off_pairs():
    rows = [
        "start_time\tend_time\tsensor\tvalue",
        "2008-02-25 09:10:00\t2008-02-25 09:12:00\tFrontdoor\t1",
        "2008-02-25 10:00:00\t2008-02-25 10:30:00\tHall-Toilet door\t1",
    ]
    dataset = parse_kasteren(rows)
    assert len(dataset.changes) == 4
    front = [c for c in dataset.changes if c.entity_id == "binary_sensor.frontdoor"]
    assert [c.state for c in front] == ["on", "off"]
    assert "binary_sensor.hall_toilet_door" in dataset.entity_ids


def test_load_dataset_rejects_unknown_kinds(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("")
    with pytest.raises(ValueError):
        load_dataset(path, kind="nonsense")


# --- mining on dataset-shaped data --------------------------------------
def test_time_of_day_miner_recovers_a_casas_habit():
    """A 07:0x kitchen-light habit in CASAS format must be found."""
    dataset = parse_casas(synthesise_casas_lines(days=25, habit_hour=7))
    changes = dataset.changes
    window = dataset.window
    options = Options(min_occurrences=5, min_consistency=0.6)

    candidates = time_of_day.mine(changes, options, window)
    light_on = [
        c for c in candidates
        if c.actions[0].entity_id == "light.l003" and c.actions[0].service == "light.turn_on"
    ]
    assert light_on, "the CASAS light habit was not recovered"
    at = light_on[0].triggers[0].at
    assert at.startswith("07:"), f"expected an 07:xx trigger, got {at}"
    assert light_on[0].evidence.consistency >= 0.8


def test_association_miner_runs_on_casas_data():
    dataset = parse_casas(synthesise_casas_lines(days=30))
    options = Options(min_support=0.05, min_confidence=0.6, min_lift=1.2)
    candidates = association.mine(dataset.changes, options, dataset.window)
    # The generated data pairs motion M017 with light L003 every morning.
    pairs = {(c.triggers[0].entity_id, c.actions[0].entity_id) for c in candidates}
    assert ("binary_sensor.m017", "light.l003") in pairs


def test_dataset_stats_are_reported():
    dataset = parse_casas(synthesise_casas_lines(days=10))
    stats = dataset.stats()
    assert stats["changes"] > 0
    assert stats["human"] > 0
    assert stats["device"] > 0
    assert stats["activities"] == 10
    assert 9 <= stats["days"] <= 11


# --- optional: the real archive -----------------------------------------
CASAS_FILE = os.environ.get("AMMINER_CASAS_FILE")


@pytest.mark.slow
@pytest.mark.skipif(not CASAS_FILE, reason="set AMMINER_CASAS_FILE to a real CASAS data file")
def test_real_casas_archive_parses_and_mines():
    dataset = load_dataset(CASAS_FILE, kind="casas")
    assert dataset.changes
    stats = dataset.stats()
    assert stats["entities"] > 5
    candidates = time_of_day.mine(dataset.changes, Options(), dataset.window)
    # We do not assert *which* habits appear in someone else's home, only that
    # the pipeline produces explainable, evidence-carrying output.
    for candidate in candidates:
        assert candidate.evidence.occurrences >= Options().min_occurrences
        assert candidate.evidence.consistency >= Options().min_consistency


# --- fixture determinism -------------------------------------------------
def test_the_fixture_is_reproducible_from_its_seed_alone(tmp_path):
    """The same seed and the same end date must give byte-identical ground truth."""
    import datetime as dt

    from amminer.testing.synthetic import build_default_fixture

    end = dt.datetime(2024, 5, 1, 12, 0, tzinfo=dt.UTC)
    first = build_default_fixture(tmp_path / "a.db", days=45, end=end, seed=99, tz=dt.UTC)
    second = build_default_fixture(tmp_path / "b.db", days=45, end=end, seed=99, tz=dt.UTC)

    assert first.start_ts == second.start_ts
    assert first.end_ts == second.end_ts
    assert first.time_of_day == second.time_of_day
    assert first.conditional == second.conditional


def test_every_weekday_the_suite_could_start_on_still_recovers_the_habit(tmp_path):
    """The generator's weekday branches consume different amounts of randomness.

    Pinning the end date makes the suite reproducible; this keeps the coverage
    that pinning it would otherwise remove.
    """
    import datetime as dt

    from amminer.config import Options
    from amminer.discovery.recorder import (
        RecorderInfo,
        create_recorder_engine,
        probe,
        sqlite_readonly_url,
    )
    from amminer.miners import time_of_day
    from amminer.recorderdb import causality
    from amminer.recorderdb.queries import ORIGIN_EVENT_TYPES, RecorderQueries
    from amminer.testing.synthetic import build_default_fixture

    for offset in range(7):
        end = dt.datetime(2024, 5, 1, 12, 0, tzinfo=dt.UTC) + dt.timedelta(days=offset)
        path = tmp_path / f"week{offset}.db"
        truth = build_default_fixture(path, days=45, end=end, seed=20240501, tz=dt.UTC)
        engine = create_recorder_engine(sqlite_readonly_url(path))
        try:
            info = probe(engine, RecorderInfo(dialect="sqlite"))
            queries = RecorderQueries(engine, info)
            changes = queries.state_changes(truth.start_ts - 1, truth.end_ts + 1)
            events = queries.events(
                truth.start_ts - 1, truth.end_ts + 1, list(ORIGIN_EVENT_TYPES)
            )
            causality.annotate(changes, events)
            found = time_of_day.mine(
                changes, Options(), (truth.start_ts, truth.end_ts)
            )
        finally:
            engine.dispose()
        targets = {a.entity_id for c in found for a in c.actions}
        assert "light.kitchen" in targets, f"habit lost when the window ends on {end:%A}"
