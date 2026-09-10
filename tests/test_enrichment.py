"""External-signal detection, signal series, and gap suggestions."""

from __future__ import annotations

import pytest
from amminer.discovery.recorder import RecorderInfo
from amminer.enrich.detect import detect_signals
from amminer.enrich.signals import SignalSeries, build_signal_store
from amminer.entities import EntityResolver
from amminer.gaps import suggest
from amminer.recorderdb.models import Cause, StateChange


def resolver_from(states) -> EntityResolver:
    resolver = EntityResolver()
    resolver.merge_states(states)
    return resolver


def state(entity_id, value="on", **attributes):
    return {"entity_id": entity_id, "state": value, "attributes": attributes}


# --- detection ----------------------------------------------------------
def test_detects_core_signals():
    signals = detect_signals(
        resolver_from(
            [
                state("sun.sun", "above_horizon"),
                state("weather.home", "sunny", temperature=12.0),
                state("binary_sensor.workday_sensor", friendly_name="Workday sensor"),
                state("calendar.family"),
                state("person.alex", "home"),
                state("device_tracker.alex_phone", "home"),
                state("climate.living_room", "heat"),
            ]
        )
    )
    assert signals.sun == ["sun.sun"]
    assert signals.weather == ["weather.home"]
    assert signals.workday == ["binary_sensor.workday_sensor"]
    assert signals.calendar == ["calendar.family"]
    assert signals.person == ["person.alex"]
    assert signals.device_tracker == ["device_tracker.alex_phone"]
    assert signals.thermostat == ["climate.living_room"]


@pytest.mark.parametrize(
    ("entity_id", "attribute", "role"),
    [
        ("sensor.tibber_electricity_price_home", {}, "energy_price"),
        ("sensor.nordpool_kwh_se3", {}, "energy_price"),
        ("sensor.current_electricity_market_price", {}, "energy_price"),
        ("sensor.electricity_price_level", {}, "price_level"),
        ("sensor.solcast_pv_forecast_today", {}, "solar_forecast"),
        ("sensor.energy_production_today", {}, "solar_forecast"),
        ("sensor.electricity_maps_carbon_intensity", {}, "carbon_intensity"),
        ("sensor.dwd_weather_warnings_current_warning_level", {}, "weather_warning"),
    ],
)
def test_detects_energy_and_weather_signals(entity_id, attribute, role):
    signals = detect_signals(resolver_from([state(entity_id, "1.0", **attribute)]))
    assert entity_id in getattr(signals, role), signals.as_dict()


def test_detects_outdoor_temperature_by_device_class_and_by_name():
    by_class = detect_signals(
        resolver_from(
            [state("sensor.garden_temp", "8.1", device_class="temperature", friendly_name="Garden")]
        )
    )
    assert by_class.outdoor_temperature == ["sensor.garden_temp"]

    by_name = detect_signals(resolver_from([state("sensor.outdoor_temperature", "8.1")]))
    assert by_name.outdoor_temperature == ["sensor.outdoor_temperature"]


def test_detects_deferrable_loads_and_ev_chargers():
    signals = detect_signals(
        resolver_from(
            [
                state("switch.dishwasher"),
                state("switch.washing_machine"),
                state("switch.easee_charger"),
                state("switch.desk_lamp"),
            ]
        )
    )
    assert "switch.dishwasher" in signals.deferrable_loads
    assert "switch.washing_machine" in signals.deferrable_loads
    assert "switch.easee_charger" in signals.ev_charger
    assert "switch.desk_lamp" not in signals.deferrable_loads


def test_detects_mmwave_as_room_presence():
    signals = detect_signals(
        resolver_from(
            [
                state("binary_sensor.bedroom_presence", device_class="occupancy",
                      friendly_name="Bedroom FP2 presence"),
                state("binary_sensor.hall_motion", device_class="motion"),
            ]
        )
    )
    assert "binary_sensor.bedroom_presence" in signals.room_presence
    assert "binary_sensor.hall_motion" in signals.motion
    assert "binary_sensor.hall_motion" not in signals.room_presence


def test_no_signals_is_a_valid_outcome():
    signals = detect_signals(resolver_from([state("light.a"), state("switch.b")]))
    assert signals.present() == set()
    assert signals.summary() == "none"


# --- signal series ------------------------------------------------------
def test_value_at_is_a_step_function():
    series = SignalSeries("sensor.t", numeric=True)
    for ts, value in ((100.0, 5.0), (200.0, 10.0), (300.0, 15.0)):
        series.add(ts, value)
    series.finalise()
    assert series.value_at(50.0) is None       # before the first sample
    assert series.value_at(100.0) == 5.0
    assert series.value_at(250.0) == 10.0      # last value at or before
    assert series.value_at(999.0) == 15.0


def test_stale_values_are_refused():
    series = SignalSeries("sensor.t", numeric=True)
    series.add(100.0, 5.0)
    series.finalise()
    assert series.value_at(100_000.0, max_staleness=3600) is None
    assert series.value_at(3600.0, max_staleness=7200) == 5.0


def test_build_store_splits_numeric_and_categorical():
    changes = [
        StateChange("sensor.t", "21.5", 100.0),
        StateChange("sensor.t", "22.0", 200.0),
        StateChange("person.alex", "home", 150.0),
        StateChange("sensor.t", "unavailable", 250.0),  # must be skipped
    ]
    store = build_signal_store(changes, ["sensor.t", "person.alex"], None)
    assert store.numeric_entities() == ["sensor.t"]
    assert store.categorical_entities() == ["person.alex"]
    assert store.numeric_at("sensor.t", 300.0) == 22.0
    assert store.value_at("person.alex", 300.0) == "home"


def test_attribute_series_are_extracted():
    change = StateChange("weather.home", "sunny", 100.0, attributes={"temperature": 9.5})
    store = build_signal_store([change], ["weather.home#temperature"], None)
    assert store.numeric_at("weather.home#temperature", 200.0) == 9.5


def test_long_term_statistics_fill_in_short_raw_history(queries, fixture_db):
    """LTS survives purge_keep_days - it must be used when raw history is thin."""
    _path, truth = fixture_db
    # Deliberately pass no raw changes at all for this entity.
    store = build_signal_store(
        [], ["sensor.outdoor_temperature"], queries, (truth.start_ts, truth.end_ts)
    )
    series = store.get("sensor.outdoor_temperature")
    assert series is not None
    assert series.source == "statistics"
    assert len(series) > 20
    assert store.stats()["from_statistics"] == ["sensor.outdoor_temperature"]


# --- gap suggestions ----------------------------------------------------
def _human_changes(entity_id, count=25):
    out = []
    for i in range(count):
        change = StateChange(entity_id, "on" if i % 2 == 0 else "off", 1000.0 + i * 3600,
                             old_state="off" if i % 2 == 0 else "on")
        change.cause = Cause.HUMAN
        out.append(change)
    return out


def test_suggests_mmwave_when_only_pir_exists():
    resolver = resolver_from(
        [state("light.kitchen"), state("binary_sensor.hall_motion", device_class="motion")]
    )
    signals = detect_signals(resolver)
    gaps = suggest(resolver, signals, _human_changes("light.kitchen"))
    assert any("mmWave" in g.title for g in gaps)
    finding = next(g for g in gaps if "mmWave" in g.title)
    assert "PIR" in finding.gap
    assert finding.evidence


def test_suggests_a_price_sensor_for_deferrable_loads():
    resolver = resolver_from([state("switch.dishwasher")])
    gaps = suggest(resolver, detect_signals(resolver), [])
    assert any("price sensor" in g.title for g in gaps)


def test_suggests_solar_forecast_when_inverter_present():
    resolver = resolver_from([state("sensor.solaredge_pv_power", "1200", device_class="power")])
    gaps = suggest(resolver, detect_signals(resolver), [])
    assert any("forecast" in g.title.lower() for g in gaps)


def test_suggests_outdoor_temperature_for_climate():
    resolver = resolver_from([state("climate.living_room", "heat")])
    gaps = suggest(resolver, detect_signals(resolver), [])
    assert any("outdoor temperature" in g.title.lower() for g in gaps)


def test_suggests_presence_detection_when_absent():
    resolver = resolver_from([state("light.a")])
    gaps = suggest(resolver, detect_signals(resolver), [])
    assert any("presence" in g.title.lower() for g in gaps)


def test_suggests_mariadb_on_short_sqlite_history():
    info = RecorderInfo(dialect="sqlite", purge_keep_days=10)
    info.oldest_state_ts, info.newest_state_ts = 0.0, 8 * 86400.0
    resolver = resolver_from([state("light.a")])
    gaps = suggest(resolver, detect_signals(resolver), [], recorder_info=info)
    mariadb = next((g for g in gaps if "MariaDB" in g.title), None)
    assert mariadb is not None
    assert "core-mariadb" in mariadb.recommendation
    assert mariadb.score >= 0.85  # this is the highest-value fix


def test_no_mariadb_suggestion_with_long_history():
    info = RecorderInfo(dialect="sqlite", purge_keep_days=90)
    info.oldest_state_ts, info.newest_state_ts = 0.0, 60 * 86400.0
    resolver = resolver_from([state("light.a")])
    gaps = suggest(resolver, detect_signals(resolver), [], recorder_info=info)
    assert not any("MariaDB" in g.title for g in gaps)


def test_gap_ids_are_stable():
    resolver = resolver_from([state("climate.a", "heat")])
    signals = detect_signals(resolver)
    first = {g.id for g in suggest(resolver, signals, [])}
    second = {g.id for g in suggest(resolver, signals, [])}
    assert first == second


def test_statistics_are_stamped_when_their_hour_closes(queries, fixture_db):
    """An hourly mean stamped at the hour it opens lets a backtest see ahead."""
    _path, truth = fixture_db
    store = build_signal_store(
        [], ["sensor.outdoor_temperature"], queries, (truth.start_ts, truth.end_ts)
    )
    series = store.get("sensor.outdoor_temperature")
    assert series is not None
    rows = queries.statistics(["sensor.outdoor_temperature"], truth.start_ts, truth.end_ts)
    assert rows, "fixture has no statistics to check against"
    opens = min(float(row["start_ts"]) for row in rows)
    # Nothing is readable at the instant the first hour begins; the mean of that
    # hour is only known once it is over.
    assert series.numeric_at(opens) is None
    assert series.numeric_at(opens + 3600) is not None


def test_statistics_never_replace_real_readings(queries, fixture_db):
    """A year of hourly means must not swallow ten days of actual values."""
    _path, truth = fixture_db
    raw = [
        change
        for change in queries.state_changes(truth.start_ts - 1, truth.end_ts + 86400)
        if change.entity_id == "sensor.outdoor_temperature"
    ]
    assert raw, "fixture has no raw readings to protect"
    recent = [c for c in raw if c.ts >= truth.end_ts - 2 * 86400][:10]
    assert recent

    store = build_signal_store(
        recent, ["sensor.outdoor_temperature"], queries, (truth.start_ts, truth.end_ts)
    )
    series = store.get("sensor.outdoor_temperature")
    assert series is not None
    # The exact readings survive at their own timestamps ...
    for change in recent:
        if change.numeric is not None:
            assert series.numeric_at(change.ts) == pytest.approx(change.numeric)
    # ... and the statistics still cover the period before them.
    assert series.numeric_at(truth.start_ts + 10 * 86400) is not None
    assert "statistics" in series.source
