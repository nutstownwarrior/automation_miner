"""Map public smart-home datasets onto the internal event schema.

The miners only ever see :class:`~amminer.recorderdb.models.StateChange`, so any
dataset that can be expressed as "(timestamp, sensor, value, who caused it)" can
validate the mining logic without a Home Assistant instance.

Supported layouts:

CASAS (casas.wsu.edu/datasets)
    ``2008-02-27 12:43:27.416392  M17  ON``, optionally followed by an activity
    label (``Meal_Preparation begin`` / ``end``).  Sensor prefixes carry the
    type: ``M`` motion, ``D`` door, ``L`` light, ``T`` temperature, ``AD``
    analog, ``I`` item, ``P`` power.

ARAS (House A/B)
    Space-separated binary columns, one row per second, with two trailing
    activity-label columns.  A header file names the sensors.

Kasteren
    ``start_time  end_time  sensor_id  value`` interval rows; each interval
    becomes an ON change at the start and an OFF change at the end.

Actuator-like channels (lights, doors, items, power) are attributed to a
human, matching the datasets' semantics of a resident acting; passive sensors
(motion, temperature) are attributed to the device.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..recorderdb.models import Cause, StateChange

_LOGGER = logging.getLogger(__name__)

#: CASAS sensor-id prefix -> (HA domain, treated as a human action?)
CASAS_PREFIXES: tuple[tuple[str, str, bool], ...] = (
    ("AD", "sensor", False),
    ("M", "binary_sensor", False),
    ("MA", "binary_sensor", False),
    ("D", "binary_sensor", True),
    ("L", "light", True),
    ("LS", "sensor", False),
    ("T", "sensor", False),
    ("I", "binary_sensor", True),
    ("P", "sensor", False),
    ("E", "switch", True),
)

_CASAS_LINE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>[\d:.]+)\s+(?P<sensor>\S+)\s+(?P<value>\S+)"
    r"(?:\s+(?P<label>.+))?$"
)


@dataclass
class DatasetEvents:
    """A dataset mapped into the internal schema."""

    name: str
    changes: list[StateChange] = field(default_factory=list)
    activities: list[tuple[float, float, str]] = field(default_factory=list)
    entity_ids: set[str] = field(default_factory=set)

    @property
    def window(self) -> tuple[float, float]:
        if not self.changes:
            return (0.0, 0.0)
        return (self.changes[0].ts, self.changes[-1].ts)

    def stats(self) -> dict[str, int | float]:
        human = sum(1 for c in self.changes if c.cause is Cause.HUMAN)
        return {
            "changes": len(self.changes),
            "entities": len(self.entity_ids),
            "human": human,
            "device": len(self.changes) - human,
            "activities": len(self.activities),
            "days": round((self.window[1] - self.window[0]) / 86400.0, 2),
        }


def _classify_casas(sensor: str) -> tuple[str, bool]:
    """Return ``(domain, is_human_action)`` for a CASAS sensor id."""
    letters = re.match(r"^[A-Za-z]+", sensor)
    prefix = (letters.group(0) if letters else "M").upper()
    for candidate, domain, human in sorted(CASAS_PREFIXES, key=lambda x: -len(x[0])):
        if prefix == candidate:
            return domain, human
    return "binary_sensor", False


def _normalise_value(value: str) -> str:
    lowered = value.strip().lower()
    return {
        "on": "on",
        "off": "off",
        "open": "on",
        "close": "off",
        "closed": "off",
        "present": "on",
        "absent": "off",
        "1": "on",
        "0": "off",
    }.get(lowered, lowered)


def parse_casas(lines: Iterable[str], name: str = "casas") -> DatasetEvents:
    """Parse CASAS ``data`` files into the internal schema."""
    dataset = DatasetEvents(name=name)
    last_state: dict[str, str] = {}
    open_activities: dict[str, float] = {}

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        match = _CASAS_LINE.match(line)
        if not match:
            continue
        timestamp = f"{match['date']} {match['time']}"
        try:
            moment = dt.datetime.fromisoformat(timestamp).replace(tzinfo=dt.UTC)
        except ValueError:
            continue
        ts = moment.timestamp()
        sensor = match["sensor"]
        value = _normalise_value(match["value"])
        domain, human = _classify_casas(sensor)
        entity_id = f"{domain}.{sensor.lower()}"

        change = StateChange(
            entity_id=entity_id,
            state=value,
            ts=ts,
            old_state=last_state.get(entity_id),
            cause=Cause.HUMAN if human else Cause.DEVICE,
            context_id=f"casas-{len(dataset.changes)}",
            context_user_id="resident" if human else None,
        )
        last_state[entity_id] = value
        dataset.entity_ids.add(entity_id)
        dataset.changes.append(change)

        label = (match["label"] or "").strip()
        if label:
            parts = label.split()
            activity = parts[0]
            marker = parts[-1].lower() if len(parts) > 1 else ""
            if marker == "begin":
                open_activities[activity] = ts
            elif marker == "end" and activity in open_activities:
                dataset.activities.append((open_activities.pop(activity), ts, activity))

    dataset.changes.sort(key=lambda c: c.ts)
    _LOGGER.info("Parsed CASAS dataset %s: %s", name, dataset.stats())
    return dataset


def parse_aras(
    lines: Iterable[str],
    sensor_names: Sequence[str] | None = None,
    start: dt.datetime | None = None,
    name: str = "aras",
    step_seconds: float = 1.0,
) -> DatasetEvents:
    """Parse an ARAS ``DAY_*.txt`` file (binary columns, one row per second)."""
    dataset = DatasetEvents(name=name)
    start = start or dt.datetime(2013, 1, 1, tzinfo=dt.UTC)
    base = start.timestamp()
    previous: list[str] | None = None
    activity_start: float | None = None
    activity_label: str | None = None

    for index, raw in enumerate(lines):
        columns = raw.split()
        if len(columns) < 3:
            continue
        sensors = columns[:-2]
        label = columns[-2]
        ts = base + index * step_seconds

        if previous is not None:
            # ARAS rows can be ragged; zip's shortest-wins behaviour is deliberate here.
            for position, (old, new) in enumerate(zip(previous, sensors)):  # noqa: B905
                if old == new:
                    continue
                sensor_name = (
                    sensor_names[position]
                    if sensor_names and position < len(sensor_names)
                    else f"sensor_{position:02d}"
                )
                entity_id = f"binary_sensor.{sensor_name.lower()}"
                change = StateChange(
                    entity_id=entity_id,
                    state="on" if new == "1" else "off",
                    ts=ts,
                    old_state="on" if old == "1" else "off",
                    cause=Cause.HUMAN,
                    context_id=f"aras-{index}-{position}",
                    context_user_id="resident1",
                )
                dataset.entity_ids.add(entity_id)
                dataset.changes.append(change)
        previous = sensors

        if label != activity_label:
            if activity_label is not None and activity_start is not None:
                dataset.activities.append((activity_start, ts, activity_label))
            activity_label, activity_start = label, ts

    dataset.changes.sort(key=lambda c: c.ts)
    _LOGGER.info("Parsed ARAS dataset %s: %s", name, dataset.stats())
    return dataset


def parse_kasteren(lines: Iterable[str], name: str = "kasteren") -> DatasetEvents:
    """Parse Kasteren interval rows (``start end sensor value``)."""
    dataset = DatasetEvents(name=name)
    for raw in lines:
        line = raw.strip()
        if not line or line.lower().startswith(("start", "#")):
            continue
        parts = re.split(r"\t+|\s{2,}", line)
        if len(parts) < 3:
            parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = dt.datetime.fromisoformat(parts[0].strip()).replace(tzinfo=dt.UTC)
            end = dt.datetime.fromisoformat(parts[1].strip()).replace(tzinfo=dt.UTC)
        except ValueError:
            continue
        sensor = parts[2].strip()
        entity_id = f"binary_sensor.{re.sub(r'[^a-z0-9]+', '_', sensor.lower()).strip('_')}"
        dataset.entity_ids.add(entity_id)
        dataset.changes.append(
            StateChange(entity_id, "on", start.timestamp(), old_state="off",
                        cause=Cause.HUMAN, context_user_id="resident")
        )
        dataset.changes.append(
            StateChange(entity_id, "off", end.timestamp(), old_state="on",
                        cause=Cause.HUMAN, context_user_id="resident")
        )
    dataset.changes.sort(key=lambda c: c.ts)
    _LOGGER.info("Parsed Kasteren dataset %s: %s", name, dataset.stats())
    return dataset


def load_dataset(path: str | Path, kind: str = "casas", **kwargs) -> DatasetEvents:
    """Load a dataset file from disk."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    parsers = {"casas": parse_casas, "aras": parse_aras, "kasteren": parse_kasteren}
    parser = parsers.get(kind.lower())
    if parser is None:
        raise ValueError(f"unknown dataset kind '{kind}' (expected one of {sorted(parsers)})")
    return parser(lines, name=path.stem, **kwargs)


def synthesise_casas_lines(
    days: int = 20, start: dt.datetime | None = None, habit_hour: int = 7
) -> Iterator[str]:
    """Emit CASAS-formatted lines with a known habit, for offline testing.

    The public archives cannot be redistributed, so the parser and the miners
    are exercised on generated data in exactly the archive's on-disk format.
    """
    start = start or dt.datetime(2011, 6, 1, tzinfo=dt.UTC)
    for day in range(days):
        base = start + dt.timedelta(days=day)
        morning = base.replace(hour=habit_hour, minute=2 + (day % 5))
        yield f"{morning:%Y-%m-%d %H:%M:%S.%f}\tM017\tON\tMeal_Preparation begin"
        yield f"{morning + dt.timedelta(minutes=1):%Y-%m-%d %H:%M:%S.%f}\tL003\tON"
        yield f"{morning + dt.timedelta(minutes=25):%Y-%m-%d %H:%M:%S.%f}\tL003\tOFF"
        yield f"{morning + dt.timedelta(minutes=26):%Y-%m-%d %H:%M:%S.%f}\tM017\tOFF\tMeal_Preparation end"
        evening = base.replace(hour=19, minute=30 + (day % 7))
        yield f"{evening:%Y-%m-%d %H:%M:%S.%f}\tD002\tOPEN"
        yield f"{evening + dt.timedelta(seconds=45):%Y-%m-%d %H:%M:%S.%f}\tD002\tCLOSE"
        yield f"{base.replace(hour=13):%Y-%m-%d %H:%M:%S.%f}\tT001\t21.5"
