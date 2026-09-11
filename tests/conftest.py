"""Shared fixtures.  Every test runs against a real recorder-shaped database."""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

import pytest
from amminer.config import Options
from amminer.discovery.recorder import (
    RecorderInfo,
    create_recorder_engine,
    probe,
    sqlite_readonly_url,
)
from amminer.recorderdb import causality
from amminer.recorderdb.queries import ORIGIN_EVENT_TYPES, RecorderQueries
from amminer.store import Store
from amminer.testing.synthetic import build_default_fixture

# Pinned before any test module is imported, because several of them compute
# timestamps at module scope.  The synthetic history is built in UTC and the
# miners read the *local* timezone (TZ, which the Supervisor sets in the
# container); with the two disagreeing, a habit injected at 06:30 is recovered
# at 20:30 and the suite fails for a reason that has nothing to do with the
# code.  test_timezones.py covers the non-UTC case this pin would otherwise
# hide.
os.environ["TZ"] = "UTC"
time.tzset()

FIXTURE_DAYS = 45
FIXTURE_TZ = dt.UTC

#: The fixture ends on a fixed date, not on "today".
#:
#: The seed alone does not make it reproducible: several injected patterns are
#: conditioned on the weekday, and the weekday branches consume different
#: numbers of random() calls, so the same seed run on a Tuesday and on a Friday
#: produced different ground truth - measured at hits 28-30 and consistency
#: 0.82-0.91 for the same habit.  Today's margins happen to be wide enough not
#: to flip an assertion, which is not the same as the suite being deterministic.
#: A Wednesday, so the 45-day window covers whole weeks plus a remainder.
FIXTURE_END = dt.datetime(2024, 5, 1, 12, 0, tzinfo=FIXTURE_TZ)


@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory) -> tuple[Path, object]:
    """A synthetic recorder database plus the ground truth used to build it."""
    directory = tmp_path_factory.mktemp("recorder")
    path = directory / "home-assistant_v2.db"
    truth = build_default_fixture(
        path, days=FIXTURE_DAYS, end=FIXTURE_END, seed=20240501, tz=FIXTURE_TZ
    )
    return path, truth


@pytest.fixture(scope="session")
def recorder_engine(fixture_db):
    path, _truth = fixture_db
    engine = create_recorder_engine(sqlite_readonly_url(path))
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def recorder_info(recorder_engine) -> RecorderInfo:
    return probe(recorder_engine, RecorderInfo(dialect="sqlite"))


@pytest.fixture(scope="session")
def queries(recorder_engine, recorder_info) -> RecorderQueries:
    return RecorderQueries(recorder_engine, recorder_info)


@pytest.fixture(scope="session")
def raw_changes(queries, fixture_db):
    _path, truth = fixture_db
    return queries.state_changes(truth.start_ts - 1, truth.end_ts + 86400, with_attributes=True)


@pytest.fixture(scope="session")
def raw_events(queries, fixture_db):
    _path, truth = fixture_db
    return queries.events(truth.start_ts - 1, truth.end_ts + 86400, ORIGIN_EVENT_TYPES)


@pytest.fixture
def classified(raw_changes, raw_events):
    """Freshly classified copies, so mutating tests cannot leak between cases."""
    import copy

    changes = copy.deepcopy(list(raw_changes))
    changes, index = causality.annotate(changes, raw_events)
    return changes, index


@pytest.fixture
def options() -> Options:
    return Options()


@pytest.fixture
def store(tmp_path) -> Store:
    with Store(tmp_path / "state.db") as store:
        yield store


@pytest.fixture
def ha_config_dir(tmp_path, fixture_db) -> Path:
    """A minimal but realistic Home Assistant config directory."""
    path, _truth = fixture_db
    directory = tmp_path / "homeassistant"
    (directory / ".storage").mkdir(parents=True)
    (directory / "configuration.yaml").write_text(
        "homeassistant:\n"
        "  name: Test Home\n"
        "  time_zone: UTC\n"
        "recorder:\n"
        "  purge_keep_days: 60\n"
        "automation: !include automations.yaml\n",
        encoding="utf-8",
    )
    (directory / "automations.yaml").write_text("[]\n", encoding="utf-8")
    import shutil

    shutil.copy(path, directory / "home-assistant_v2.db")
    return directory


@pytest.fixture
def registry_files(ha_config_dir) -> Path:
    """Registry files including entities with and without a unique_id."""
    storage = ha_config_dir / ".storage"
    entity_registry = {
        "version": 1,
        "minor_version": 15,
        "key": "core.entity_registry",
        "data": {
            "entities": [
                {
                    "entity_id": "light.kitchen",
                    "unique_id": "abc-kitchen",
                    "platform": "hue",
                    "device_id": "dev1",
                    "area_id": None,
                    "name": None,
                    "original_name": "Kitchen ceiling",
                    "labels": ["lbl_downstairs"],
                    "categories": {},
                    "aliases": [],
                    "disabled_by": None,
                    "hidden_by": None,
                    "entity_category": None,
                    "config_entry_id": "cfg1",
                },
                {
                    "entity_id": "switch.heater",
                    "unique_id": "abc-heater",
                    "platform": "shelly",
                    "device_id": "dev2",
                    "area_id": "area_living",
                    "name": "Heater",
                    "original_name": "Relay 0",
                    "labels": [],
                    "categories": {},
                    "aliases": ["the heater"],
                    "disabled_by": None,
                    "hidden_by": None,
                    "entity_category": None,
                    "config_entry_id": "cfg2",
                },
                {
                    "entity_id": "sensor.disabled_thing",
                    "unique_id": "abc-disabled",
                    "platform": "hue",
                    "device_id": "dev1",
                    "area_id": None,
                    "disabled_by": "user",
                    "hidden_by": None,
                    "labels": [],
                },
            ],
            # A long deleted_entities section must be ignored entirely.
            "deleted_entities": [
                {"entity_id": f"light.ghost_{i}", "unique_id": f"ghost{i}", "platform": "hue"}
                for i in range(500)
            ],
        },
    }
    device_registry = {
        "version": 1,
        "key": "core.device_registry",
        "data": {
            "devices": [
                {
                    "id": "dev1",
                    "name": "Hue bridge lamp",
                    "name_by_user": None,
                    "manufacturer": "Signify",
                    "model": "LCT015",
                    "area_id": "area_kitchen",
                    "via_device_id": None,
                    "labels": [],
                    "config_entries": ["cfg1"],
                    "identifiers": [["hue", "1"]],
                },
                {
                    "id": "dev2",
                    "name": "Shelly Plug",
                    "manufacturer": "Shelly",
                    "model": "PlugS",
                    "area_id": "area_living",
                    "labels": ["lbl_power"],
                    "config_entries": ["cfg2"],
                },
            ],
            "deleted_devices": [{"id": f"gone{i}"} for i in range(200)],
        },
    }
    area_registry = {
        "version": 1,
        "key": "core.area_registry",
        "data": {
            "areas": [
                {"id": "area_kitchen", "name": "Kitchen", "floor_id": "floor_ground", "labels": []},
                {"id": "area_living", "name": "Living room", "floor_id": "floor_ground", "labels": []},
            ]
        },
    }
    label_registry = {
        "version": 1,
        "key": "core.label_registry",
        "data": {"labels": [{"label_id": "lbl_downstairs", "name": "Downstairs"},
                            {"label_id": "lbl_power", "name": "Power hungry"}]},
    }
    floor_registry = {
        "version": 1,
        "key": "core.floor_registry",
        "data": {"floors": [{"floor_id": "floor_ground", "name": "Ground floor"}]},
    }
    for name, payload in (
        ("core.entity_registry", entity_registry),
        ("core.device_registry", device_registry),
        ("core.area_registry", area_registry),
        ("core.label_registry", label_registry),
        ("core.floor_registry", floor_registry),
    ):
        (storage / name).write_text(json.dumps(payload), encoding="utf-8")
    return ha_config_dir


@pytest.fixture
def states_payload() -> list[dict]:
    """A ``/api/states`` snapshot including entities absent from the registry."""
    return [
        {
            "entity_id": "light.kitchen",
            "state": "off",
            "attributes": {"friendly_name": "Kitchen ceiling", "supported_color_modes": ["hs"]},
        },
        {"entity_id": "switch.heater", "state": "off", "attributes": {"friendly_name": "Heater"}},
        # No unique_id -> never in the registry, only recoverable via states.
        {
            "entity_id": "sensor.outdoor_temperature",
            "state": "7.4",
            "attributes": {
                "friendly_name": "Outdoor temperature",
                "unit_of_measurement": "°C",
                "device_class": "temperature",
            },
        },
        {
            "entity_id": "binary_sensor.workday_sensor",
            "state": "on",
            "attributes": {"friendly_name": "Workday sensor"},
        },
        {"entity_id": "person.alex", "state": "home", "attributes": {"friendly_name": "Alex"}},
        {
            "entity_id": "light.hallway",
            "state": "off",
            "attributes": {"friendly_name": "Hallway"},
        },
        {
            "entity_id": "climate.living_room",
            "state": "off",
            "attributes": {"friendly_name": "Living room thermostat"},
        },
        {
            "entity_id": "light.bedroom",
            "state": "on",
            "attributes": {"friendly_name": "Bedroom"},
        },
        {
            "entity_id": "automation.holiday_mode",
            "state": "on",
            "attributes": {"friendly_name": "Holiday mode", "id": "holiday1", "last_triggered": None},
        },
        {
            "entity_id": "automation.bedtime_dim",
            "state": "on",
            "attributes": {
                "friendly_name": "Bedtime dim",
                "id": "bedtime1",
                "last_triggered": "2099-01-01T22:15:00+00:00",
            },
        },
    ]


class FakeHAClient:
    """A stand-in for :class:`amminer.ha_api.HAClient` with no network."""

    def __init__(self, states=None, services=None, check_config_result="valid"):
        self._states = states or []
        self._services = services or {
            "light.turn_on", "light.turn_off", "switch.turn_on", "switch.turn_off",
            "climate.set_hvac_mode", "climate.turn_off", "automation.reload",
            "homeassistant.turn_on", "homeassistant.turn_off", "scene.turn_on",
            "script.turn_on", "cover.open_cover", "cover.close_cover",
        }
        self.check_config_result = check_config_result
        self.configured = True
        self.last_error = None
        self.written: dict[str, dict] = {}
        self.reloaded = False
        #: Every service call, so tests can assert what reached Home Assistant.
        self.service_calls: list[tuple[str, dict]] = []
        #: When set, every service call fails the way the real client reports it.
        self.service_fails = False

    def get_states(self):
        return list(self._states)

    def service_index(self):
        return set(self._services)

    def check_config(self):
        return {"result": self.check_config_result, "errors": None}

    def upsert_automation(self, automation_id, config):
        self.written[automation_id] = config
        return True

    def call_service(self, domain, service, data=None):
        self.service_calls.append((f"{domain}.{service}", dict(data or {})))
        if self.service_fails:
            self.last_error = "service unavailable"
            return None
        return []

    def reload_automations(self):
        # Through call_service, as the real client does, so a test watching
        # service calls sees this one too.
        if self.call_service("automation", "reload") is None:
            return False
        self.reloaded = True
        return True

    def ws_registry_lists(self):
        return None

    def get_services(self):
        return []

    def close(self):
        pass


@pytest.fixture
def fake_client(states_payload) -> FakeHAClient:
    return FakeHAClient(states_payload)
