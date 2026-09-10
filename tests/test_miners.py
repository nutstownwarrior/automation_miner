"""Miners must recover exactly the patterns the generator injected."""

from __future__ import annotations

import datetime as dt

import pytest
from amminer.enrich.detect import SignalSet
from amminer.enrich.signals import build_signal_store
from amminer.miners import association, conditional, energy, motif, sequence, stale, time_of_day
from amminer.miners.time_of_day import service_for
from amminer.recorderdb.models import Cause, StateChange

UTC = dt.UTC


@pytest.fixture
def window(fixture_db):
    _path, truth = fixture_db
    return truth.start_ts, truth.end_ts


# --- miner A -----------------------------------------------------------
def test_time_of_day_recovers_the_injected_0630_habit(classified, options, window, fixture_db):
    _path, truth = fixture_db
    changes, _index = classified
    candidates = time_of_day.mine(changes, options, window)

    kitchen_on = [
        c
        for c in candidates
        if c.actions[0].entity_id == "light.kitchen" and c.actions[0].service == "light.turn_on"
    ]
    assert kitchen_on, "the 06:30 kitchen habit was not recovered"
    found = kitchen_on[0]

    at = found.triggers[0].at
    minutes = int(at[:2]) * 60 + int(at[3:5])
    expected = truth.time_of_day["hour"] * 60 + truth.time_of_day["minute"]
    assert abs(minutes - expected) <= 10, f"expected ~06:30, got {at}"

    assert found.evidence.consistency >= options.min_consistency
    assert found.evidence.occurrences >= options.min_occurrences
    # The generator only fires this on weekdays, so the miner must say so.
    assert found.evidence.extra["day_filter"] == "weekdays"
    assert found.conditions and found.conditions[0].weekday == ["mon", "tue", "wed", "thu", "fri"]


def test_time_of_day_ignores_automation_caused_changes(options, window):
    """The whole point: an entity an automation already drives is not a habit."""
    changes = []
    for day in range(30):
        base = window[0] + day * 86400
        change = StateChange("light.auto", "on", base + 6 * 3600, old_state="off")
        change.cause = Cause.AUTOMATION
        changes.append(change)
    assert time_of_day.mine(changes, options, window) == []


def test_time_of_day_needs_enough_occurrences(options, window):
    changes = []
    for day in range(3):
        change = StateChange("light.rare", "on", window[0] + day * 86400 + 21600, old_state="off")
        change.cause = Cause.HUMAN
        changes.append(change)
    assert time_of_day.mine(changes, options, window) == []


def test_time_of_day_rejects_too_few_clustered_occurrences(options, window):
    """Twenty actions at random hours is not a habit, however many there are."""
    changes = []
    for day in range(40):
        hour = (day * 7) % 24
        change = StateChange(
            "light.random", "on", window[0] + day * 86400 + hour * 3600, old_state="off"
        )
        change.cause = Cause.HUMAN
        changes.append(change)
    candidates = time_of_day.mine(changes, options, window)
    assert not [c for c in candidates if c.actions[0].entity_id == "light.random"]


def test_circular_clustering_handles_midnight(options):
    """23:59 and 00:05 are six minutes apart, not twenty-three hours."""
    from amminer.util.timeutil import local_tz

    tz = local_tz()
    first = dt.datetime(2024, 3, 1, 0, 0, tzinfo=tz)
    changes = []
    for day in range(30):
        midnight = first + dt.timedelta(days=day)
        # One action per calendar day, alternating either side of midnight.
        offset = dt.timedelta(minutes=1439) if day % 2 else dt.timedelta(minutes=5)
        moment = midnight + offset
        change = StateChange("light.night", "on", moment.timestamp(), old_state="off")
        change.cause = Cause.HUMAN
        changes.append(change)
    window = (first.timestamp(), (first + dt.timedelta(days=30)).timestamp())
    candidates = [
        c for c in time_of_day.mine(changes, options, window)
        if c.actions[0].entity_id == "light.night"
    ]
    assert candidates, "a habit straddling midnight must still be found"
    at = candidates[0].triggers[0].at
    minutes = int(at[:2]) * 60 + int(at[3:5])
    # The circular mean must land near midnight, not near noon.
    assert min(minutes, 1440 - minutes) <= 30


@pytest.mark.parametrize(
    ("entity_id", "state", "expected"),
    [
        ("light.a", "on", "light.turn_on"),
        ("light.a", "off", "light.turn_off"),
        ("cover.a", "open", "cover.open_cover"),
        ("cover.a", "closed", "cover.close_cover"),
        ("lock.a", "locked", "lock.lock"),
        ("climate.a", "heat", "climate.set_hvac_mode"),
        ("climate.a", "off", "climate.turn_off"),
        ("input_select.a", "night", "input_select.select_option"),
    ],
)
def test_service_mapping(entity_id, state, expected):
    service, _data = service_for(entity_id, state)
    assert service == expected


def test_service_mapping_rejects_unknowns():
    assert service_for("sensor.temp", "21.5") is None


# --- miner B -----------------------------------------------------------
def test_stale_automation_detected(classified, options, window, fake_client):
    from amminer.entities import EntityResolver

    changes, _index = classified
    resolver = EntityResolver()
    resolver.merge_states(fake_client.get_states())
    findings = stale.mine_stale_automations(resolver, options, now=window[1])
    titles = {f.title for f in findings}
    assert any("Holiday mode" in t for t in titles)  # never triggered
    assert not any("Bedtime dim" in t for t in titles)  # fired recently


def test_unused_entity_detected(options, window, fake_client):
    from amminer.entities import EntityResolver

    resolver = EntityResolver()
    resolver.merge_states(fake_client.get_states())
    changes = [StateChange("light.kitchen", "on", window[0] + 100, old_state="off")]
    changes[0].cause = Cause.HUMAN
    findings = stale.mine_unused_entities(changes, resolver, options, window)
    unused = {f.entities[0] for f in findings}
    assert "light.hallway" in unused
    assert "light.kitchen" not in unused


# --- miner C -----------------------------------------------------------
def test_association_recovers_presence_rule(classified, options, window):
    changes, _index = classified
    candidates = association.mine(changes, options, window)
    pairs = {(c.triggers[0].entity_id, c.actions[0].entity_id) for c in candidates}
    assert ("person.alex", "light.hallway") in pairs
    rule = next(c for c in candidates if c.triggers[0].entity_id == "person.alex")
    assert rule.evidence.lift >= options.min_lift
    assert rule.evidence.confidence >= options.min_confidence


def test_association_ignores_numeric_sensor_values(classified, options, window):
    """A numeric reading is not a symbol; every value would be its own item."""
    changes, _index = classified
    candidates = association.mine(changes, options, window)
    assert not any(
        t.entity_id == "sensor.outdoor_temperature" for c in candidates for t in c.triggers
    )


# --- miner D -----------------------------------------------------------
def test_sequence_recovers_the_arrive_home_routine(classified, options, window):
    changes, _index = classified
    candidates = sequence.mine(changes, options, window)
    assert candidates
    routine = next(
        (c for c in candidates if c.triggers[0].entity_id == "person.alex"), None
    )
    assert routine is not None, "the arrive-home routine was not recovered"
    targets = [a.entity_id for a in routine.actions]
    assert "light.hallway" in targets
    assert "climate.living_room" in targets
    # Order matters: lights before the thermostat, as injected.
    assert targets.index("light.hallway") < targets.index("climate.living_room")


def test_prefixspan_finds_ordered_subsequences():
    sequences = [["a", "b", "c"], ["a", "b", "d"], ["a", "b", "c"], ["x", "y"]]
    patterns = dict(sequence.prefixspan(sequences, min_support=3))
    assert patterns[("a", "b")] == 3
    assert ("b", "a") not in patterns  # order is respected


# --- miner F -----------------------------------------------------------
def test_conditional_recovers_the_temperature_driven_heater(
    classified, options, window, queries, fixture_db
):
    _path, truth = fixture_db
    changes, _index = classified
    signals = SignalSet(
        outdoor_temperature=["sensor.outdoor_temperature"], person=["person.alex"]
    )
    store = build_signal_store(changes, signals.all_entities, queries, window)
    candidates = conditional.mine(changes, options, signals, store, window)

    heater = next(
        (c for c in candidates if c.actions[0].entity_id == "switch.heater"
         and c.actions[0].service == "switch.turn_on"),
        None,
    )
    assert heater is not None, "the temperature-driven heater habit was not recovered"
    trigger = heater.triggers[0]
    assert trigger.entity_id == "sensor.outdoor_temperature"
    assert trigger.below is not None
    # The generator's threshold is 8.0 C.
    assert abs(trigger.below - truth.conditional["below"]) <= 2.5
    # Not `lift > MIN_LIFT`, which the miner guarantees by construction and so
    # asserts nothing.  This is the lift the injected pattern actually has.
    assert heater.evidence.lift > 3.0


def test_conditional_skips_when_no_signals(classified, options, window):
    changes, _index = classified
    from amminer.enrich.signals import SignalStore

    assert conditional.mine(changes, options, SignalSet(), SignalStore(), window) == []


# --- miner E -----------------------------------------------------------
def test_motif_falls_back_without_stumpy(classified, options, window, queries):
    changes, _index = classified
    store = build_signal_store(
        changes, ["sensor.outdoor_temperature", "switch.heater"], queries, window
    )
    candidates = motif.mine(changes, options, store, window)
    # With or without stumpy this must not raise, and any candidate it does
    # produce must carry the honest note about which path ran.
    for candidate in candidates:
        assert "stumpy" in candidate.evidence.extra
        assert candidate.triggers[0].kind == "numeric_state"


# --- energy ------------------------------------------------------------
def test_energy_miner_skips_without_a_price_signal(classified, options, window):
    from amminer.enrich.signals import SignalStore

    changes, _index = classified
    signals = SignalSet(deferrable_loads=["switch.dishwasher"])
    assert energy.mine(changes, options, signals, SignalStore(), window) == []


def test_energy_miner_suggests_shifting_expensive_runs(options):
    """A dishwasher always started in the expensive evening must be flagged."""
    start = 1_700_000_000.0
    window = (start, start + 20 * 86400)
    changes: list[StateChange] = []
    for day in range(20):
        base = start + day * 86400
        for hour in range(24):
            price = 0.45 if 17 <= hour <= 21 else 0.10
            change = StateChange(
                "sensor.electricity_price", f"{price:.2f}", base + hour * 3600, old_state="0"
            )
            change.cause = Cause.DEVICE
            changes.append(change)
        run = StateChange("switch.dishwasher", "on", base + 19 * 3600 + 600, old_state="off")
        run.cause = Cause.HUMAN
        changes.append(run)

    signals = SignalSet(
        energy_price=["sensor.electricity_price"], deferrable_loads=["switch.dishwasher"]
    )
    store = build_signal_store(changes, signals.all_entities, None, window)
    candidates = energy.mine(changes, options, signals, store, window)
    assert candidates
    assert candidates[0].actions[0].entity_id == "switch.dishwasher"
    assert candidates[0].evidence.confidence >= 0.35
    cheapest_hour = candidates[0].evidence.extra["cheapest_block"][0]
    assert not (17 <= cheapest_hour <= 21)


# --- association direction ----------------------------------------------
def _ordered_pair_history(days: int = 80) -> list:
    """The pump always comes on 60s before the light, and off 60s before it."""
    from amminer.recorderdb.models import Cause, StateChange

    def change(entity_id: str, state: str, ts: float, old: str) -> StateChange:
        item = StateChange(entity_id=entity_id, state=state, ts=ts,
                           old_state=old, last_changed_ts=ts)
        item.cause = Cause.HUMAN
        return item

    base = 1_700_000_000.0
    history = []
    for day in range(days):
        ts = base + day * 86400
        history.append(change("switch.pump", "on", ts, "off"))
        history.append(change("light.kitchen", "on", ts + 60, "off"))
        history.append(change("switch.pump", "off", ts + 3600, "on"))
        history.append(change("light.kitchen", "off", ts + 3660, "on"))
    return history


def test_association_does_not_propose_the_rule_backwards():
    """A basket is a set, so FP-Growth yields both arrows with equal evidence."""
    from amminer.config import Options as MinerOptions
    from amminer.miners import association

    history = _ordered_pair_history()
    candidates = association.mine(
        history, MinerOptions(), (history[0].ts, history[-1].ts)
    )
    arrows = {(c.triggers[0].entity_id, c.actions[0].entity_id) for c in candidates}

    assert ("switch.pump", "light.kitchen") in arrows
    # Same support, same confidence, same lift - and the pump does not come on
    # because the light did.  Two cards with identical numbers is how a user
    # accepts both and builds a loop.
    assert ("light.kitchen", "switch.pump") not in arrows


def test_a_sensor_that_merely_follows_is_not_offered_as_a_trigger():
    """The power draw rises *because* the heater came on."""
    from amminer.config import Options as MinerOptions
    from amminer.miners import association
    from amminer.recorderdb.models import Cause, StateChange

    def change(entity_id, state, ts, old, cause=Cause.HUMAN):
        item = StateChange(entity_id=entity_id, state=state, ts=ts,
                           old_state=old, last_changed_ts=ts)
        item.cause = cause
        return item

    base = 1_700_000_000.0
    history = []
    for day in range(80):
        ts = base + day * 86400
        history.append(change("switch.heater", "on", ts, "off"))
        history.append(
            change("binary_sensor.high_draw", "on", ts + 30, "off", Cause.DEVICE)
        )
        history.append(change("switch.heater", "off", ts + 7200, "on"))
        history.append(
            change("binary_sensor.high_draw", "off", ts + 7230, "on", Cause.DEVICE)
        )

    candidates = association.mine(history, MinerOptions(), (base, history[-1].ts))
    arrows = {(c.triggers[0].entity_id, c.actions[0].entity_id) for c in candidates}
    assert ("binary_sensor.high_draw", "switch.heater") not in arrows


# --- confounding --------------------------------------------------------
def test_conditional_does_not_mistake_a_shared_schedule_for_a_cause():
    """A pure clock habit and an unrelated evening dip are not cause and effect."""
    import datetime as dt

    from amminer.config import Options as MinerOptions
    from amminer.enrich.detect import SignalSet
    from amminer.enrich.signals import SignalSeries, SignalStore
    from amminer.miners import conditional
    from amminer.recorderdb.models import Cause, StateChange

    base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC).timestamp()
    days = 60
    changes = []
    for day in range(days):
        on = base + day * 86400 + 18 * 3600
        for ts, state, old in ((on, "on", "off"), (on + 3600, "off", "on")):
            change = StateChange(entity_id="light.kitchen", state=state, ts=ts,
                                 old_state=old, last_changed_ts=ts)
            change.cause = Cause.HUMAN
            changes.append(change)

    # A nightly backup job, nothing to do with the light.
    load = SignalSeries("sensor.server_load", numeric=True)
    for day in range(days):
        for hour in range(24):
            load.add(base + day * 86400 + hour * 3600, 30.0 if 17 <= hour <= 21 else 70.0)
    store = SignalStore()
    store.add(load.finalise())

    signals = SignalSet()
    signals.power = ["sensor.server_load"]

    found = conditional.mine(
        changes, MinerOptions(), signals, store, (base, base + days * 86400)
    )
    assert found == [], [c.title for c in found]


def test_sequence_confidence_is_conditional_on_the_trigger():
    """A 10-of-10 routine reported 20% because it divided by everything."""
    import datetime as dt

    from amminer.config import Options as MinerOptions
    from amminer.miners import sequence
    from amminer.recorderdb.models import Cause, StateChange

    def change(entity_id, state, ts, old):
        item = StateChange(entity_id=entity_id, state=state, ts=ts,
                           old_state=old, last_changed_ts=ts)
        item.cause = Cause.HUMAN
        return item

    base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC).timestamp()
    changes = []
    for day in range(10):
        ts = base + day * 86400
        changes.append(change("binary_sensor.front_door", "on", ts, "off"))
        changes.append(change("light.hall", "on", ts + 20, "off"))
        changes.append(change("light.porch", "on", ts + 40, "off"))
    # Forty unrelated sessions on other days, which must not dilute anything.
    for index in range(40):
        ts = base + (20 + index) * 86400
        changes.append(change("light.bedroom", "on", ts, "off"))
        changes.append(change("light.bedroom", "off", ts + 60, "on"))

    found = sequence.mine(changes, MinerOptions(), (base, base + 70 * 86400))
    assert found, "the routine must be recovered"
    routine = found[0]
    assert routine.evidence.occurrences == 10
    assert routine.evidence.opportunities == 10  # times the trigger happened
    assert routine.evidence.confidence == pytest.approx(1.0)
    assert routine.score == pytest.approx(1.0)


def test_a_day_restricted_habit_must_beat_every_day_by_a_margin():
    """Best-of-three correlated hypotheses, reported as if it were the only one."""
    import datetime as dt
    import random

    from amminer.config import Options as MinerOptions
    from amminer.miners import time_of_day
    from amminer.recorderdb.models import Cause, StateChange

    def trial(seed: int) -> bool:
        rng = random.Random(seed)
        base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC).timestamp()
        changes = []
        for day in range(40):
            if rng.random() >= 0.55:  # day-INDEPENDENT, so no day claim is true
                continue
            ts = base + day * 86400 + 18 * 3600 + rng.randint(-120, 120)
            for at, state, old in ((ts, "on", "off"), (ts + 3600, "off", "on")):
                item = StateChange(entity_id="light.kitchen", state=state, ts=at,
                                   old_state=old, last_changed_ts=at)
                item.cause = Cause.HUMAN
                changes.append(item)
        found = time_of_day.mine(changes, MinerOptions(), (base, base + 40 * 86400))
        return any(c.conditions for c in found)

    accepted = sum(trial(seed) for seed in range(200))
    # Before the margin this was over half of them.
    assert accepted < 40, f"{accepted}/200 day-independent habits called day-conditioned"
