"""A Home Assistant instance is rarely in UTC.

The rest of the suite pins ``TZ`` so the synthetic history and the miners agree
on what "06:30" means.  That pin would otherwise hide the case the code is
explicitly written for - ``local_tz()`` reads ``TZ`` "because the Supervisor
sets it in the container" - so it is exercised here on purpose.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from zoneinfo import ZoneInfo

import pytest
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
from amminer.util.timeutil import local_tz, minute_of_day

ZONES = ("UTC", "America/Los_Angeles", "Europe/Berlin", "Pacific/Kiritimati")


@pytest.fixture
def in_zone(request):
    """Run one test as though the container were in *request.param*."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = request.param
    time.tzset()
    yield request.param
    if previous is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous
    time.tzset()


@pytest.mark.parametrize("in_zone", ZONES, indirect=True)
def test_local_tz_follows_the_container_setting(in_zone):
    assert local_tz() == ZoneInfo(in_zone)


@pytest.mark.parametrize("in_zone", ZONES, indirect=True)
def test_minute_of_day_is_read_in_the_instance_timezone(in_zone):
    zone = ZoneInfo(in_zone)
    moment = dt.datetime(2024, 5, 1, 6, 30, tzinfo=zone)
    assert minute_of_day(moment.timestamp()) == 6 * 60 + 30


@pytest.mark.parametrize("in_zone", ZONES, indirect=True)
def test_a_non_utc_instance_still_recovers_its_habits(tmp_path, in_zone):
    """History recorded in the instance's own timezone, mined in the same one."""
    zone = ZoneInfo(in_zone)
    end = dt.datetime(2024, 5, 1, 12, 0, tzinfo=zone)
    path = tmp_path / f"{in_zone.replace('/', '_')}.db"
    truth = build_default_fixture(path, days=45, end=end, seed=20240501, tz=zone)

    engine = create_recorder_engine(sqlite_readonly_url(path))
    try:
        info = probe(engine, RecorderInfo(dialect="sqlite"))
        queries = RecorderQueries(engine, info)
        changes = queries.state_changes(truth.start_ts - 1, truth.end_ts + 1)
        events = queries.events(truth.start_ts - 1, truth.end_ts + 1, list(ORIGIN_EVENT_TYPES))
        causality.annotate(changes, events)
        found = time_of_day.mine(changes, Options(), (truth.start_ts, truth.end_ts))
    finally:
        engine.dispose()

    kitchen = [
        candidate
        for candidate in found
        if candidate.actions[0].entity_id == "light.kitchen"
        and candidate.actions[0].service == "light.turn_on"
    ]
    assert kitchen, f"the 06:30 habit was lost in {in_zone}"
    at = str(kitchen[0].triggers[0].at)
    recovered = int(at[:2]) * 60 + int(at[3:5])
    expected = truth.time_of_day["hour"] * 60 + truth.time_of_day["minute"]
    assert abs(recovered - expected) <= 10, f"{in_zone}: recovered {at}, wanted ~06:30"
