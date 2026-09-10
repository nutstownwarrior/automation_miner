"""The thresholds that decide what a user is shown.

Each of these was found by deleting the check and running the suite: every one
could be removed, or moved by orders of magnitude, without a single test
failing.  A threshold nothing tests is a threshold nothing is holding.
"""

from __future__ import annotations

import datetime as dt

import pytest
from amminer.config import Options
from amminer.enrich.detect import SignalSet
from amminer.enrich.signals import SignalSeries, SignalStore
from amminer.entities import EntityResolver
from amminer.miners import association, conditional, stale, time_of_day
from amminer.recorderdb.models import Cause, StateChange

BASE = dt.datetime(2024, 1, 1, tzinfo=dt.UTC).timestamp()


def human(entity_id: str, state: str, ts: float, old: str | None = None) -> StateChange:
    change = StateChange(
        entity_id=entity_id,
        state=state,
        ts=ts,
        old_state=old if old is not None else ("off" if state == "on" else "on"),
        last_changed_ts=ts,
    )
    change.cause = Cause.HUMAN
    return change


def automated(entity_id: str, state: str, ts: float) -> StateChange:
    change = human(entity_id, state, ts)
    change.cause = Cause.AUTOMATION
    return change


def _toggles(entity_id: str, days, hour_minute_for) -> list[StateChange]:
    out = []
    for day in days:
        minutes = hour_minute_for(day)
        if minutes is None:
            continue
        ts = BASE + day * 86400 + minutes * 60
        out.append(human(entity_id, "on", ts))
        out.append(human(entity_id, "off", ts + 3600))
    return out


# --- time_of_day: min_consistency ---------------------------------------
def test_a_habit_scattered_across_a_long_window_fails_on_consistency():
    """Enough occurrences to pass min_occurrences, far too few days to mean it.

    This is the case the README sells: "6 of 7 weekdays scores far above 6 hits
    scattered over 60 days".  Deleting the consistency check left every test
    green, because the scattered case was being rejected earlier, on the
    occurrence count.
    """
    days = 60
    # Eight hits, all at exactly 18:00, spread over sixty days: the cluster is
    # tight, the count clears min_occurrences, and it happens on one day in
    # eight.
    changes = _toggles(
        "light.kitchen", range(days), lambda d: 18 * 60 if d % 8 == 0 else None
    )
    options = Options()
    found = time_of_day.mine(changes, options, (BASE, BASE + days * 86400))
    kitchen = [c for c in found if c.actions[0].entity_id == "light.kitchen"]
    assert kitchen == [], "a rule that fires one day in eight is not a habit"


def test_the_same_number_of_hits_on_consecutive_days_is_a_habit():
    """The contrast case: same count, same cluster, but it happens every day."""
    changes = _toggles("light.kitchen", range(8), lambda d: 18 * 60)
    found = time_of_day.mine(changes, Options(), (BASE, BASE + 8 * 86400))
    kitchen = [
        c for c in found
        if c.actions[0].entity_id == "light.kitchen"
        and c.actions[0].service == "light.turn_on"
    ]
    assert kitchen, "eight for eight is exactly what this miner is for"
    assert kitchen[0].evidence.consistency >= Options().min_consistency


# --- conditional: MIN_LIFT, MIN_PURITY and MAX_SIGNAL_STALENESS ---------
DAYS = 60
ACTION_DAYS = [day for day in range(DAYS) if day % 2 == 0]
WINDOW = (BASE, BASE + DAYS * 86400)


def _evening_actions() -> list[StateChange]:
    """The heater goes on at 18:00, every other day."""
    out = []
    for day in ACTION_DAYS:
        ts = BASE + day * 86400 + 18 * 3600
        out.append(human("switch.heater", "on", ts))
        out.append(human("switch.heater", "off", ts + 3600))
    return out


def _temperature(value_for, hours=range(24)) -> SignalStore:
    series = SignalSeries("sensor.outdoor_temperature", numeric=True)
    for day in range(DAYS):
        for hour in hours:
            series.add(BASE + day * 86400 + hour * 3600, value_for(day))
    store = SignalStore()
    store.add(series.finalise())
    return store


def _signals() -> SignalSet:
    signals = SignalSet()
    signals.outdoor_temperature = ["sensor.outdoor_temperature"]
    return signals


def _mine(store: SignalStore):
    return conditional.mine(_evening_actions(), Options(), _signals(), store, WINDOW)


def test_a_driver_that_is_just_as_true_when_you_did_not_act_has_no_lift():
    """Cold on the days you switched it on - and on almost every other day too."""
    # Only two of the thirty non-action days differ, so the temperature says
    # almost nothing about whether the heater went on.
    warm = {day for day in range(DAYS) if day % 2 == 1 and day <= 3}
    assert _mine(_temperature(lambda d: 18.0 if d in warm else 2.0)) == []


def test_a_driver_that_only_covers_a_third_of_the_actions_is_not_pure_enough():
    """Real lift, far too little coverage: it explains a sixth of what you did."""
    store = _temperature(lambda d: 2.0 if (d in ACTION_DAYS and d % 6 == 0) else 18.0)
    assert _mine(store) == []


def test_a_driver_that_is_both_pure_and_lifted_is_found():
    """The contrast case, so the two tests above are not passing for free."""
    found = _mine(_temperature(lambda d: 2.0 if d in ACTION_DAYS else 18.0))
    assert found, "a perfectly discriminative driver must be found"
    assert found[0].evidence.lift is not None and found[0].evidence.lift > 1.4


def test_a_reading_fifteen_hours_old_does_not_explain_anything():
    """'Below 8 degrees' has to mean now, not at three o'clock this morning."""
    # Perfectly discriminative - but only ever reported at 03:00, and the
    # actions are at 18:00.
    store = _temperature(lambda d: 2.0 if d in ACTION_DAYS else 18.0, hours=(3,))
    assert _mine(store) == []


# --- association: min_lift and the human-consequent filter ---------------
def test_two_things_that_both_happen_all_day_are_not_a_rule():
    """Perfect confidence, no lift: the hall light is in nearly every basket."""
    changes = []
    for day in range(60):
        start = BASE + day * 86400
        for hour in (7, 9, 11, 13, 15, 17, 19, 21):
            changes.append(human("light.hall", "on", start + hour * 3600))
            changes.append(human("light.hall", "off", start + hour * 3600 + 120))
        changes.append(human("switch.fan", "on", start + 13 * 3600 + 30))
        changes.append(human("switch.fan", "off", start + 13 * 3600 + 90))

    found = association.mine(changes, Options(), (BASE, BASE + 60 * 86400))
    arrows = {(c.triggers[0].entity_id, c.actions[0].entity_id) for c in found}
    assert ("switch.fan", "light.hall") not in arrows


def test_an_action_no_human_ever_takes_is_not_proposed():
    """The consequent must be something the user does by hand.

    An entity that only ever moves because an automation moved it is already
    automated; proposing it back is proposing what they have.
    """
    changes = []
    for day in range(100):
        start = BASE + day * 86400
        if day < 25:
            changes.append(human("binary_sensor.motion", "on", start))
            changes.append(automated("light.kitchen", "on", start + 60))
            changes.append(human("binary_sensor.motion", "off", start + 300))
            changes.append(automated("light.kitchen", "off", start + 360))
        else:
            # Unrelated human activity, so the pair above is not in every basket.
            changes.append(human("switch.fan", "on", start + 3600))
            changes.append(human("light.bedroom", "on", start + 3660))
            changes.append(human("switch.fan", "off", start + 7200))
            changes.append(human("light.bedroom", "off", start + 7260))

    found = association.mine(changes, Options(), (BASE, BASE + 100 * 86400))
    arrows = {(c.triggers[0].entity_id, c.actions[0].entity_id) for c in found}
    # The genuinely human pair is still found, so this is not passing for free.
    assert ("switch.fan", "light.bedroom") in arrows
    assert ("binary_sensor.motion", "light.kitchen") not in arrows


# --- stale: the age cutoff ----------------------------------------------
def _automation_resolver(last_triggered: str | None) -> EntityResolver:
    resolver = EntityResolver()
    resolver.merge_states([
        {
            "entity_id": "automation.dusty",
            "state": "on",
            "attributes": {"friendly_name": "Dusty", "last_triggered": last_triggered},
        }
    ])
    return resolver


@pytest.mark.parametrize(
    "days_ago, expected_stale",
    [
        (5, False),    # fired last week
        (29, False),   # just inside the default 30-day cutoff
        (31, True),    # just outside it
        (400, True),   # long gone
    ],
)
def test_the_stale_cutoff_is_where_it_says_it_is(days_ago, expected_stale):
    """The whole aging branch could be moved three years and nothing noticed."""
    now = dt.datetime(2024, 6, 1, 12, 0, tzinfo=dt.UTC)
    last = (now - dt.timedelta(days=days_ago)).isoformat()
    found = stale.mine_stale_automations(
        _automation_resolver(last), Options(), now=now.timestamp()
    )
    is_stale = any(c.miner == "stale_automation" for c in found)
    assert is_stale is expected_stale


def test_the_cutoff_follows_the_option():
    now = dt.datetime(2024, 6, 1, 12, 0, tzinfo=dt.UTC)
    last = (now - dt.timedelta(days=45)).isoformat()
    resolver = _automation_resolver(last)
    lenient = stale.mine_stale_automations(
        resolver, Options(stale_automation_days=90), now=now.timestamp()
    )
    strict = stale.mine_stale_automations(
        resolver, Options(stale_automation_days=30), now=now.timestamp()
    )
    assert not any(c.miner == "stale_automation" for c in lenient)
    assert any(c.miner == "stale_automation" for c in strict)
