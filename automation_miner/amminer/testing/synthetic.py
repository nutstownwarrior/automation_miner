"""Generate a synthetic Home Assistant recorder database.

The generator writes the real post-2023 normalised schema (``states_meta``,
``state_attributes``, ``event_types``, ``event_data``, binary context columns,
``statistics``/``statistics_meta``) so the production SQL is exercised verbatim
- no test-only query path.

It injects *known* patterns so miners can be asserted against ground truth:

* a 06:30 weekday "kitchen light on" habit,
* an arrive-home sequence (presence -> hallway light -> thermostat),
* a temperature-driven heater habit (heater on when outdoor temp < 8 °C),
* automation-caused changes plus deliberate human overrides of them.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE states_meta (
    metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   VARCHAR(255)
);
CREATE UNIQUE INDEX ix_states_meta_entity_id ON states_meta (entity_id);

CREATE TABLE state_attributes (
    attributes_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash          INTEGER,
    shared_attrs  TEXT
);

CREATE TABLE states (
    state_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    metadata_id         INTEGER,
    state               VARCHAR(255),
    attributes_id       INTEGER,
    old_state_id        INTEGER,
    last_updated_ts     FLOAT,
    last_changed_ts     FLOAT,
    last_reported_ts    FLOAT,
    context_id_bin      BLOB,
    context_user_id_bin BLOB,
    context_parent_id_bin BLOB,
    origin_idx          INTEGER
);
CREATE INDEX ix_states_metadata_id_last_updated_ts ON states (metadata_id, last_updated_ts);
CREATE INDEX ix_states_last_updated_ts ON states (last_updated_ts);

CREATE TABLE event_types (
    event_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type    VARCHAR(64)
);
CREATE UNIQUE INDEX ix_event_types_event_type ON event_types (event_type);

CREATE TABLE event_data (
    data_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    hash        INTEGER,
    shared_data TEXT
);

CREATE TABLE events (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type_id       INTEGER,
    data_id             INTEGER,
    origin_idx          INTEGER,
    time_fired_ts       FLOAT,
    context_id_bin      BLOB,
    context_user_id_bin BLOB,
    context_parent_id_bin BLOB
);
CREATE INDEX ix_events_time_fired_ts ON events (time_fired_ts);

CREATE TABLE statistics_meta (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    statistic_id        VARCHAR(255),
    source              VARCHAR(32),
    unit_of_measurement VARCHAR(255),
    has_mean            BOOLEAN,
    has_sum             BOOLEAN,
    name                VARCHAR(255)
);

CREATE TABLE statistics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts  FLOAT,
    metadata_id INTEGER,
    start_ts    FLOAT,
    mean        FLOAT,
    min         FLOAT,
    max         FLOAT,
    last_reset_ts FLOAT,
    state       FLOAT,
    sum         FLOAT
);

CREATE TABLE statistics_short_term (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts  FLOAT,
    metadata_id INTEGER,
    start_ts    FLOAT,
    mean        FLOAT,
    min         FLOAT,
    max         FLOAT,
    last_reset_ts FLOAT,
    state       FLOAT,
    sum         FLOAT
);

CREATE TABLE recorder_runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    start      DATETIME,
    end        DATETIME,
    closed_incorrect BOOLEAN,
    created    DATETIME
);
"""


def new_context_id() -> bytes:
    """16 raw bytes, exactly like Home Assistant's ULID context columns."""
    return uuid.uuid4().bytes


@dataclass
class Injected:
    """Ground truth about what the generator put into the database."""

    time_of_day: dict[str, Any] = field(default_factory=dict)
    sequence: dict[str, Any] = field(default_factory=dict)
    conditional: dict[str, Any] = field(default_factory=dict)
    overrides: list[dict[str, Any]] = field(default_factory=list)
    association: dict[str, Any] = field(default_factory=dict)
    start_ts: float = 0.0
    end_ts: float = 0.0
    user_id: str = ""


class SyntheticRecorder:
    """Builds a recorder-shaped SQLite file and remembers the ground truth."""

    def __init__(self, path: str | Path, tz: dt.tzinfo | None = None, seed: int = 1234) -> None:
        self.path = Path(path)
        self.tz = tz or dt.datetime.now().astimezone().tzinfo or dt.UTC
        self.random = random.Random(seed)
        if self.path.exists():
            self.path.unlink()
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(SCHEMA)
        self._meta: dict[str, int] = {}
        self._event_types: dict[str, int] = {}
        self._last_state_id: dict[str, int] = {}
        self.truth = Injected()

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> SyntheticRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _metadata_id(self, entity_id: str) -> int:
        if entity_id not in self._meta:
            cursor = self.conn.execute(
                "INSERT INTO states_meta(entity_id) VALUES(?)", (entity_id,)
            )
            self._meta[entity_id] = int(cursor.lastrowid)
        return self._meta[entity_id]

    def _event_type_id(self, event_type: str) -> int:
        if event_type not in self._event_types:
            cursor = self.conn.execute(
                "INSERT INTO event_types(event_type) VALUES(?)", (event_type,)
            )
            self._event_types[event_type] = int(cursor.lastrowid)
        return self._event_types[event_type]

    def _attributes_id(self, attributes: dict[str, Any] | None) -> int | None:
        if not attributes:
            return None
        payload = json.dumps(attributes, sort_keys=True)
        cursor = self.conn.execute(
            "INSERT INTO state_attributes(hash, shared_attrs) VALUES(?, ?)",
            (hash(payload) & 0x7FFFFFFF, payload),
        )
        return int(cursor.lastrowid)

    # ------------------------------------------------------------------
    def add_state(
        self,
        entity_id: str,
        state: str,
        ts: float,
        *,
        user_id: bytes | None = None,
        context_id: bytes | None = None,
        parent_id: bytes | None = None,
        attributes: dict[str, Any] | None = None,
        no_context: bool = False,
    ) -> bytes | None:
        """Insert one ``states`` row, wiring up ``old_state_id`` automatically.

        ``no_context`` writes a row with no context columns at all, which is
        what a state restored after a restart, or a row from before Home
        Assistant recorded contexts, actually looks like.
        """
        metadata_id = self._metadata_id(entity_id)
        context_id = None if no_context else (context_id or new_context_id())
        cursor = self.conn.execute(
            "INSERT INTO states(metadata_id, state, attributes_id, old_state_id,"
            " last_updated_ts, last_changed_ts, context_id_bin, context_user_id_bin,"
            " context_parent_id_bin, origin_idx) VALUES(?,?,?,?,?,?,?,?,?,0)",
            (
                metadata_id,
                state,
                self._attributes_id(attributes),
                self._last_state_id.get(entity_id),
                ts,
                ts,
                context_id,
                user_id,
                parent_id,
            ),
        )
        self._last_state_id[entity_id] = int(cursor.lastrowid)
        return context_id

    def add_event(
        self,
        event_type: str,
        ts: float,
        data: dict[str, Any] | None = None,
        *,
        context_id: bytes | None = None,
        user_id: bytes | None = None,
        parent_id: bytes | None = None,
    ) -> bytes:
        context_id = context_id or new_context_id()
        data_id = None
        if data:
            payload = json.dumps(data, sort_keys=True)
            cursor = self.conn.execute(
                "INSERT INTO event_data(hash, shared_data) VALUES(?, ?)",
                (hash(payload) & 0x7FFFFFFF, payload),
            )
            data_id = int(cursor.lastrowid)
        self.conn.execute(
            "INSERT INTO events(event_type_id, data_id, origin_idx, time_fired_ts,"
            " context_id_bin, context_user_id_bin, context_parent_id_bin) VALUES(?,?,0,?,?,?,?)",
            (self._event_type_id(event_type), data_id, ts, context_id, user_id, parent_id),
        )
        return context_id

    def add_statistic(
        self,
        statistic_id: str,
        rows: Sequence[tuple[float, float]],
        unit: str = "°C",
        short_term: bool = False,
    ) -> None:
        cursor = self.conn.execute(
            "INSERT INTO statistics_meta(statistic_id, source, unit_of_measurement, has_mean,"
            " has_sum, name) VALUES(?,?,?,1,0,NULL)",
            (statistic_id, "recorder", unit),
        )
        metadata_id = int(cursor.lastrowid)
        table = "statistics_short_term" if short_term else "statistics"
        self.conn.executemany(
            f"INSERT INTO {table}(created_ts, metadata_id, start_ts, mean, min, max)"
            " VALUES(?,?,?,?,?,?)",
            [(start, metadata_id, start, value, value - 0.5, value + 0.5) for start, value in rows],
        )

    # ------------------------------------------------------------------
    def _at(self, day: dt.date, hour: int, minute: int, jitter: int = 0) -> float:
        minute_offset = self.random.randint(-jitter, jitter) if jitter else 0
        moment = dt.datetime.combine(day, dt.time(hour, minute), tzinfo=self.tz) + dt.timedelta(
            minutes=minute_offset
        )
        return moment.timestamp()


def build_default_fixture(
    path: str | Path,
    days: int = 30,
    end: dt.datetime | None = None,
    seed: int = 1234,
    tz: dt.tzinfo | None = None,
    messy: bool = False,
) -> Injected:
    """Write a fixture DB with four known patterns and return the ground truth.

    ``messy`` adds what a real recorder is full of and this generator otherwise
    never produced: entities dropping to ``unavailable``/``unknown`` around
    restarts, sensors that flap, and human changes that carry no
    ``context_user_id`` at all.  The same four patterns are still in there, so a
    test can assert they survive contact with it - without which the most
    heavily exercised path through the whole pipeline is the easy case.
    """
    tz = tz or dt.datetime.now().astimezone().tzinfo or dt.UTC
    end = end or dt.datetime.now(tz).replace(hour=12, minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(days=days)

    gen = SyntheticRecorder(path, tz=tz, seed=seed)
    user_bytes = uuid.uuid4().bytes
    truth = gen.truth
    # The day loop below writes rows from midnight of the first day to late
    # evening of the last, so the declared window has to span whole days. Using
    # `start`/`end` directly would put the earliest and latest rows OUTSIDE the
    # window the ground truth advertises, which silently changes what a
    # window-bounded query returns depending on the day of the week.
    truth.start_ts = dt.datetime.combine(start.date(), dt.time(0, 0), tzinfo=tz).timestamp()
    truth.end_ts = dt.datetime.combine(
        end.date(), dt.time(23, 59, 59), tzinfo=tz
    ).timestamp()
    truth.user_id = user_bytes.hex()

    outdoor_series: list[tuple[float, float]] = []
    day = start.date()
    last_day = end.date()

    heater_conditional_hits = 0
    heater_total_days = 0
    tod_hits = 0
    tod_days = 0
    sequence_hits = 0
    assoc_hits = 0
    override_records: list[dict[str, Any]] = []

    def maybe_messy(entity_id: str, ts: float) -> None:
        """A restart, a dropout, or a flapping sensor - the ordinary mess."""
        if not messy:
            return
        roll = gen.random.random()
        if roll < 0.04:
            # The integration dropped out and came back.  The restored state
            # afterwards carries no context at all, which is what makes it
            # genuinely unknown rather than merely un-attributed.
            gen.add_state(entity_id, "unavailable", ts - 30)
            gen.add_state(entity_id, "unknown", ts - 20)
            gen.add_state(entity_id, "off", ts - 10, no_context=True)
        elif roll < 0.08:
            # A flapping contact: several changes inside a few seconds.
            for step in range(4):
                gen.add_state(entity_id, "on" if step % 2 == 0 else "off", ts - 12 + step * 3)

    def user_context(ts: float) -> bytes | None:
        """Some real human changes carry no user id: physical switches, scenes.

        The recorder records the change and nothing about who caused it.
        """
        if messy and gen.random.random() < 0.15:
            return None
        return user_bytes

    while day <= last_day:
        weekday = day.weekday()

        # --- background noise: outdoor temperature every hour --------
        for hour in range(0, 24, 1):
            base = 12.0 if weekday % 2 == 0 else 9.0
            temp = base + 6.0 * (0.5 - abs(12 - hour) / 24.0) + gen.random.uniform(-2.5, 2.5)
            ts = gen._at(day, hour, 0)
            gen.add_state("sensor.outdoor_temperature", f"{temp:.1f}", ts,
                          attributes={"unit_of_measurement": "°C", "device_class": "temperature"})
            outdoor_series.append((ts, temp))

        # --- PATTERN 1: 06:30 weekday kitchen light habit ------------
        if weekday < 5:
            tod_days += 1
            if gen.random.random() < 0.9:  # 90 % consistency
                ts = gen._at(day, 6, 30, jitter=6)
                maybe_messy("light.kitchen", ts)
                gen.add_state("light.kitchen", "on", ts, user_id=user_context(ts))
                tod_hits += 1
                # ... and off again mid-morning (noise for the miner to ignore)
                gen.add_state("light.kitchen", "off", ts + 3600, user_id=user_bytes)

        # --- PATTERN 2: leave in the morning, arrive-home sequence ---
        gen.add_state("person.alex", "not_home", gen._at(day, 8, 15, jitter=20))
        gen.add_state("light.hallway", "off", gen._at(day, 8, 20, jitter=20), user_id=user_bytes)
        if gen.random.random() < 0.85:
            base_ts = gen._at(day, 17, 45, jitter=25)
            gen.add_state("person.alex", "home", base_ts)
            maybe_messy("light.hallway", base_ts + 40)
            gen.add_state("light.hallway", "on", base_ts + 40, user_id=user_context(base_ts))
            gen.add_state("climate.living_room", "heat", base_ts + 95, user_id=user_bytes,
                          attributes={"temperature": 21.0})
            gen.add_state("climate.living_room", "off", base_ts + 5 * 3600, user_id=user_bytes)
            sequence_hits += 1
            assoc_hits += 1

        # --- PATTERN 3: heater on when it is cold --------------------
        heater_total_days += 1
        evening_ts = gen._at(day, 19, 0, jitter=40)
        evening_temp = 5.0 + gen.random.uniform(-2.0, 2.0) if weekday % 3 == 0 else 14.0 + gen.random.uniform(-2.0, 2.0)
        gen.add_state("sensor.outdoor_temperature", f"{evening_temp:.1f}", evening_ts - 120,
                      attributes={"unit_of_measurement": "°C", "device_class": "temperature"})
        outdoor_series.append((evening_ts - 120, evening_temp))
        if evening_temp < 8.0:
            maybe_messy("sensor.outdoor_temperature", evening_ts - 200)
            gen.add_state("switch.heater", "on", evening_ts, user_id=user_context(evening_ts))
            gen.add_state("switch.heater", "off", evening_ts + 7200, user_id=user_bytes)
            heater_conditional_hits += 1

        # --- PATTERN 4: an automation the user keeps overriding ------
        auto_ts = gen._at(day, 22, 15, jitter=4)
        auto_ctx = gen.add_event(
            "automation_triggered",
            auto_ts,
            {"entity_id": "automation.bedtime_dim", "name": "Bedtime dim"},
        )
        gen.add_state("light.bedroom", "off", auto_ts + 1, context_id=auto_ctx)
        if gen.random.random() < 0.7:  # the user usually puts it straight back on
            human_ts = auto_ts + gen.random.uniform(15, 90)
            gen.add_state("light.bedroom", "on", human_ts, user_id=user_bytes)
            override_records.append(
                {
                    "entity_id": "light.bedroom",
                    "ts": human_ts,
                    "automation_entity_id": "automation.bedtime_dim",
                    "automation_state": "off",
                    "human_state": "on",
                }
            )

        # --- a stale automation that never fires ---------------------
        gen.add_state("automation.holiday_mode", "on", gen._at(day, 3, 0),
                      attributes={"last_triggered": None, "friendly_name": "Holiday mode"})

        day += dt.timedelta(days=1)

    # Long-term statistics survive purge_keep_days - mirror the temperature.
    hourly = [(ts, value) for ts, value in outdoor_series if int(ts) % 3600 < 60]
    gen.add_statistic("sensor.outdoor_temperature", hourly or outdoor_series[:100])

    truth.time_of_day = {
        "entity_id": "light.kitchen",
        "state": "on",
        "hour": 6,
        "minute": 30,
        "day_filter": "weekdays",
        "hits": tod_hits,
        "opportunities": tod_days,
        "expected_consistency": tod_hits / max(tod_days, 1),
    }
    truth.sequence = {
        "steps": ["person.alex=home", "light.hallway=on", "climate.living_room=heat"],
        "occurrences": sequence_hits,
    }
    truth.association = {
        "antecedent": "person.alex=home",
        "consequent": "light.hallway=on",
        "occurrences": assoc_hits,
    }
    truth.conditional = {
        "entity_id": "switch.heater",
        "state": "on",
        "condition_entity": "sensor.outdoor_temperature",
        "below": 8.0,
        "occurrences": heater_conditional_hits,
        "opportunities": heater_total_days,
    }
    truth.overrides = override_records
    gen.close()
    return truth


def main() -> None:  # pragma: no cover - developer convenience
    import argparse

    parser = argparse.ArgumentParser(description="Generate a synthetic recorder database")
    parser.add_argument("path", help="output .db path")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    truth = build_default_fixture(args.path, days=args.days, seed=args.seed)
    print(json.dumps(dict(truth.__dict__), indent=2, default=str))
    print(f"wrote {args.path} ({os.path.getsize(args.path)} bytes)")


if __name__ == "__main__":  # pragma: no cover
    main()
