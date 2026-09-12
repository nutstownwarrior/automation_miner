"""The backtester must produce exact precision/recall/false-fire counts."""

from __future__ import annotations

import datetime as dt

import pytest
from amminer.backtest import (
    backtest,
    backtest_all,
    ground_truth_actions,
    simulate_fires,
    split_window,
)
from amminer.config import Options
from amminer.enrich.signals import SignalSeries, SignalStore
from amminer.miners.base import Action, Candidate, Condition, Trigger
from amminer.recorderdb.models import Cause, OverrideEvent, StateChange
from amminer.util.timeutil import local_tz

TZ = local_tz()
START = dt.datetime(2024, 3, 1, 0, 0, tzinfo=TZ)
DAYS = 28
WINDOW = (START.timestamp(), (START + dt.timedelta(days=DAYS)).timestamp())


def human(entity_id: str, state: str, ts: float) -> StateChange:
    change = StateChange(entity_id, state, ts, old_state="off" if state == "on" else "on")
    change.cause = Cause.HUMAN
    return change


def daily_candidate(at: str = "06:30:00") -> Candidate:
    return Candidate(
        miner="test",
        title="test rule",
        triggers=[Trigger(kind="time", at=at)],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )


def test_perfect_rule_scores_100_percent():
    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=6, minutes=30)).timestamp())
        for d in range(DAYS)
    ]
    result = backtest(daily_candidate(), changes, SignalStore(), Options(), WINDOW)
    assert result.true_fires == DAYS
    assert result.false_fires == 0
    assert result.missed == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.passed is True


def test_rule_that_mostly_misfires_is_rejected():
    """The user only did it three times; a daily rule would fire 28 times."""
    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=6, minutes=30)).timestamp())
        for d in (0, 5, 9)
    ]
    options = Options()
    result = backtest(daily_candidate(), changes, SignalStore(), options, WINDOW)
    assert result.true_fires == 3
    assert result.false_fires == DAYS - 3
    assert result.precision == pytest.approx(3 / DAYS)
    assert result.passed is False
    assert "precision" in result.reason


def test_false_fires_per_week_gate():
    """High precision is not enough if the nuisance rate is still too high."""
    changes = []
    for day in range(DAYS):
        if day % 5 != 0:  # user acts on 80 % of days -> precision 0.8
            changes.append(
                human("light.kitchen", "on",
                      (START + dt.timedelta(days=day, hours=6, minutes=30)).timestamp())
            )
    options = Options(backtest_min_precision=0.5, backtest_max_false_fires_per_week=0.5)
    result = backtest(daily_candidate(), changes, SignalStore(), options, WINDOW)
    assert result.precision > 0.5
    assert result.false_fires_per_week > 0.5
    assert result.passed is False
    assert "per week" in result.reason


def test_missed_actions_are_counted():
    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=h)).timestamp())
        for d in range(DAYS)
        for h in (6.5, 20)  # the rule only covers the morning one
    ]
    result = backtest(daily_candidate(), changes, SignalStore(), Options(), WINDOW)
    assert result.true_fires == DAYS
    assert result.missed == DAYS
    assert result.recall == pytest.approx(0.5)


def test_weekday_condition_reduces_false_fires():
    changes = []
    for day in range(DAYS):
        moment = START + dt.timedelta(days=day, hours=6, minutes=30)
        if moment.weekday() < 5:
            changes.append(human("light.kitchen", "on", moment.timestamp()))

    unconditioned = backtest(daily_candidate(), changes, SignalStore(), Options(), WINDOW)
    conditioned_candidate = daily_candidate()
    conditioned_candidate.conditions = [
        Condition(kind="time", weekday=["mon", "tue", "wed", "thu", "fri"])
    ]
    conditioned = backtest(conditioned_candidate, changes, SignalStore(), Options(), WINDOW)

    assert unconditioned.false_fires == 8  # the eight weekend days
    assert conditioned.false_fires == 0
    assert conditioned.precision == 1.0
    assert conditioned.precision > unconditioned.precision
    assert conditioned.passed is True

    # Under a stricter nuisance budget only the conditioned rule survives.
    strict = Options(backtest_max_false_fires_per_week=1.0)
    assert backtest(daily_candidate(), changes, SignalStore(), strict, WINDOW).passed is False
    assert backtest(conditioned_candidate, changes, SignalStore(), strict, WINDOW).passed is True


def test_numeric_condition_is_evaluated_from_the_signal_store():
    """A rule conditioned on a signal must only fire when the signal agrees."""
    store = SignalStore()
    series = SignalSeries("sensor.outdoor_temperature", numeric=True)
    changes = []
    for day in range(DAYS):
        cold = day % 2 == 0
        moment = START + dt.timedelta(days=day, hours=6)
        series.add(moment.timestamp(), 4.0 if cold else 18.0)
        if cold:
            changes.append(
                human("switch.heater", "on",
                      (START + dt.timedelta(days=day, hours=6, minutes=30)).timestamp())
            )
    store.add(series)

    candidate = Candidate(
        miner="test",
        title="heater",
        triggers=[Trigger(kind="time", at="06:30:00")],
        conditions=[Condition(kind="numeric_state", entity_id="sensor.outdoor_temperature", below=8.0)],
        actions=[Action(service="switch.turn_on", entity_id="switch.heater")],
    )
    result = backtest(candidate, changes, store, Options(), WINDOW)
    assert result.false_fires == 0
    assert result.precision == 1.0
    assert result.passed is True


def test_state_trigger_simulation_uses_a_tight_tolerance():
    store = SignalStore()
    presence = SignalSeries("person.alex")
    changes = []
    for day in range(DAYS):
        arrival = START + dt.timedelta(days=day, hours=17, minutes=45)
        presence.add((arrival - dt.timedelta(hours=8)).timestamp(), "not_home")
        presence.add(arrival.timestamp(), "home")
        changes.append(
            human("light.hallway", "on", (arrival + dt.timedelta(seconds=40)).timestamp())
        )
    store.add(presence)

    candidate = Candidate(
        miner="test",
        title="arrive home",
        triggers=[Trigger(kind="state", entity_id="person.alex", to_state="home")],
        actions=[Action(service="light.turn_on", entity_id="light.hallway")],
    )
    result = backtest(candidate, changes, store, Options(), WINDOW)
    assert result.true_fires == DAYS
    assert result.false_fires == 0
    assert result.passed is True


def test_an_unevaluable_time_bound_fails_closed():
    """A gate must never read a bound it cannot parse as 'no constraint'."""
    from amminer.backtest import _condition_holds

    moment = (START + dt.timedelta(days=1, hours=12)).timestamp()
    assert _condition_holds(Condition(kind="time", after="06:00:00"), moment, SignalStore(), TZ)
    for bogus in ("evening", "25:00", "12:70", "half past six"):
        assert not _condition_holds(
            Condition(kind="time", after=bogus), moment, SignalStore(), TZ
        ), bogus
        assert not _condition_holds(
            Condition(kind="time", before=bogus), moment, SignalStore(), TZ
        ), bogus


def test_a_rule_with_an_unevaluable_condition_never_fires():
    """The end-to-end consequence: it cannot be surfaced by accident."""
    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=6, minutes=30)).timestamp())
        for d in range(DAYS)
    ]
    candidate = daily_candidate()
    candidate.conditions = [Condition(kind="time", after="breakfast")]
    result = backtest(candidate, changes, SignalStore(), Options(), WINDOW)
    assert result.true_fires == 0
    assert result.passed is False


def test_time_of_day_parsing_is_strict():
    from amminer.util.timeutil import parse_time_of_day, time_of_day_minutes

    assert time_of_day_minutes("06:30") == 390
    assert time_of_day_minutes("06:30:45") == 390
    assert parse_time_of_day("6:5") == "06:05:00"
    for bogus in ("evening", "25:00", "12:70", "6", "", None, 630, "06:30:99"):
        assert time_of_day_minutes(bogus) is None, bogus
        assert parse_time_of_day(bogus) is None, bogus


def test_unsimulatable_trigger_is_rejected_not_crashed():
    candidate = daily_candidate()
    candidate.triggers = [Trigger(kind="webhook")]
    result = backtest(candidate, [], SignalStore(), Options(), WINDOW)
    assert result.simulated is False
    assert result.passed is False
    assert "cannot be simulated" in result.reason


def test_candidate_without_ground_truth_is_rejected():
    result = backtest(daily_candidate(), [], SignalStore(), Options(), WINDOW)
    assert result.passed is False
    assert "no manual occurrences" in result.reason


def test_audit_findings_are_not_gated():
    """A stale-automation finding has no action and must pass through."""
    candidate = Candidate(miner="stale_automation", title="stale", entities=["automation.x"])
    result = backtest(candidate, [], SignalStore(), Options(), WINDOW)
    assert result.passed is True
    assert result.simulated is False


def test_burst_fires_are_collapsed():
    """Twenty crossings in a minute are one event to a human."""
    store = SignalStore()
    series = SignalSeries("sensor.lux", numeric=True)
    base = (START + dt.timedelta(days=1, hours=20)).timestamp()
    for i in range(20):
        series.add(base + i * 2, 700 if i % 2 == 0 else 500)  # flapping around 600
    store.add(series)
    candidate = Candidate(
        miner="test",
        title="blinds",
        triggers=[Trigger(kind="numeric_state", entity_id="sensor.lux", below=600)],
        actions=[Action(service="cover.close_cover", entity_id="cover.blinds")],
    )
    fires, error = simulate_fires(candidate, store, WINDOW)
    assert error is None
    assert len(fires) == 1


def test_ground_truth_matches_service_target_states():
    changes = [
        human("light.kitchen", "on", 100.0),
        human("light.kitchen", "off", 200.0),
        human("light.other", "on", 300.0),
    ]
    candidate = daily_candidate()
    assert ground_truth_actions(candidate, changes) == [100.0]


def test_backtest_all_splits_and_rescores():
    good = daily_candidate()
    good.score = 0.5
    bad = daily_candidate(at="03:00:00")
    bad.score = 0.9
    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=6, minutes=30)).timestamp())
        for d in range(DAYS)
    ]
    passed, rejected = backtest_all([good, bad], changes, SignalStore(), Options(), WINDOW)
    assert [c.triggers[0].at for c in passed] == ["06:30:00"]
    assert [c.triggers[0].at for c in rejected] == ["03:00:00"]
    # A perfect backtest lifts the score.
    assert passed[0].score > 0.5
    assert passed[0].backtest["passed"] is True


def test_nuisance_fires_counted_against_overrides():
    from amminer.recorderdb.models import OverrideEvent

    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=6, minutes=30)).timestamp())
        for d in range(0, DAYS, 2)
    ]
    # An override right after each day the rule would have fired wrongly.
    overrides = [
        OverrideEvent(
            entity_id="light.kitchen",
            ts=(START + dt.timedelta(days=d, hours=6, minutes=32)).timestamp(),
            automation_entity_id="automation.x",
            automation_state="on",
            human_state="off",
            delay_seconds=120,
        )
        for d in range(1, DAYS, 2)
    ]
    result = backtest(
        daily_candidate(), changes, SignalStore(), Options(), WINDOW, overrides
    )
    assert result.false_fires > 0
    assert result.nuisance_fires > 0


# --- the evidence floor -------------------------------------------------
def test_a_single_correct_fire_is_not_evidence():
    """100% precision, zero nuisance, one observation."""
    once = (START + dt.timedelta(days=3, hours=6, minutes=30)).timestamp()
    candidate = Candidate(
        miner="test",
        title="one-off",
        triggers=[Trigger(kind="state", entity_id="binary_sensor.door", to_state="on")],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )
    store = SignalStore()
    series = SignalSeries("binary_sensor.door")
    series.add(once - 30, "on")
    store.add(series)

    result = backtest(candidate, [human("light.kitchen", "on", once)], store, Options(), WINDOW)

    assert result.true_fires == 1
    assert result.false_fires == 0
    assert result.precision == 1.0
    assert result.false_fires_per_week == 0.0
    # Every ratio is perfect and there is still nothing here.
    assert result.passed is False
    assert "too little to judge" in result.reason


def test_the_floor_is_cleared_by_a_habit_that_repeats():
    candidate = Candidate(
        miner="test",
        title="repeats",
        triggers=[Trigger(kind="state", entity_id="binary_sensor.door", to_state="on")],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )
    store = SignalStore()
    series = SignalSeries("binary_sensor.door")
    changes = []
    for day in range(10):
        ts = (START + dt.timedelta(days=day, hours=18)).timestamp()
        series.add(ts - 30, "on")
        series.add(ts - 20, "off")
        changes.append(human("light.kitchen", "on", ts))
    store.add(series)

    result = backtest(candidate, changes, store, Options(), WINDOW)

    assert result.true_fires == 10
    assert result.passed is True, result.reason
    assert "right 10 times" in result.reason


def test_a_precise_rule_that_covers_almost_nothing_is_rejected():
    """Firing correctly 3 times out of 20 is precise and useless."""
    candidate = Candidate(
        miner="test",
        title="rare",
        triggers=[Trigger(kind="state", entity_id="binary_sensor.door", to_state="on")],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )
    store = SignalStore()
    series = SignalSeries("binary_sensor.door")
    changes = []
    for day in range(20):
        ts = (START + dt.timedelta(days=day % 20, hours=18)).timestamp()
        changes.append(human("light.kitchen", "on", ts))
        if day < 3:  # the door only explains the first three
            series.add(ts - 30, "on")
            series.add(ts - 20, "off")
    store.add(series)

    result = backtest(candidate, changes, store, Options(), WINDOW)

    assert result.precision == 1.0
    assert result.recall == pytest.approx(0.15)
    assert result.passed is False
    assert "would only have covered" in result.reason


def test_false_fires_next_to_an_override_are_disqualifying():
    """Where the user has already reached over and undone an automation."""
    candidate = Candidate(
        miner="test",
        title="unwanted",
        triggers=[Trigger(kind="state", entity_id="binary_sensor.door", to_state="on")],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )
    store = SignalStore()
    series = SignalSeries("binary_sensor.door")
    changes, overrides = [], []
    for day in range(10):
        ts = (START + dt.timedelta(days=day, hours=18)).timestamp()
        series.add(ts - 30, "on")
        series.add(ts - 20, "off")
        changes.append(human("light.kitchen", "on", ts))
    # One extra opening the user did not follow with the light, and did
    # actively undo.
    stray = (START + dt.timedelta(days=11, hours=3)).timestamp()
    series.add(stray, "on")
    series.add(stray + 10, "off")
    store.add(series)
    overrides.append(
        OverrideEvent(
            entity_id="light.kitchen",
            ts=stray + 60,
            automation_entity_id="automation.x",
            automation_state="on",
            human_state="off",
            delay_seconds=60.0,
        )
    )

    options = Options()
    baseline = backtest(candidate, changes, store, options, WINDOW)
    assert baseline.nuisance_fires == 0
    assert baseline.passed is True, baseline.reason

    result = backtest(candidate, changes, store, options, WINDOW, overrides=overrides)
    assert result.nuisance_fires == 1
    assert result.passed is False
    assert "previously overridden" in result.reason


def test_a_state_trigger_is_not_judged_by_the_clock_tolerance():
    """A mixed-trigger rule must not borrow the loosest tolerance it has."""
    door_ts = (START + dt.timedelta(days=1, hours=18)).timestamp()
    candidate = Candidate(
        miner="test",
        title="mixed",
        triggers=[
            Trigger(kind="time", at="06:30:00"),
            Trigger(kind="state", entity_id="binary_sensor.door", to_state="on"),
        ],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )
    store = SignalStore()
    series = SignalSeries("binary_sensor.door")
    series.add(door_ts, "on")
    store.add(series)

    # 10 minutes after the door: inside the 15-minute clock tolerance, well
    # outside the 5-minute one a state trigger is held to.
    result = backtest(
        candidate, [human("light.kitchen", "on", door_ts + 600)], store, Options(), WINDOW
    )
    assert door_ts in result.false_fire_samples
    assert result.true_fires == 0


# --- Home Assistant semantics -------------------------------------------
def _minute(day: int, hour: int, minute: int = 0) -> float:
    return (START + dt.timedelta(days=day, hours=hour, minutes=minute)).timestamp()


@pytest.mark.parametrize(
    "hour, expected",
    [
        (23, True),   # late evening: inside the wrapped window
        (2, True),    # small hours: still inside it
        (5, True),    # just before the end
        (12, False),  # midday: outside
        (21, False),  # just before the start
    ],
)
def test_an_overnight_window_wraps_around_midnight(hour, expected):
    """'after 22:00 and before 06:00' is an evening, not an empty set."""
    condition = Condition(kind="time", after="22:00:00", before="06:00:00")
    from amminer.backtest import _condition_holds

    holds = _condition_holds(condition, _minute(2, hour), SignalStore(), local_tz())
    assert holds is expected


def test_a_daytime_window_still_reads_normally():
    from amminer.backtest import _condition_holds

    condition = Condition(kind="time", after="09:00:00", before="17:00:00")
    tz = local_tz()
    assert _condition_holds(condition, _minute(2, 12), SignalStore(), tz) is True
    assert _condition_holds(condition, _minute(2, 3), SignalStore(), tz) is False
    assert _condition_holds(condition, _minute(2, 20), SignalStore(), tz) is False


def test_an_overnight_rule_can_actually_fire():
    """The wrap bug made every night-time candidate unsimulatable, so every
    one of them was rejected for never having fired."""
    candidate = Candidate(
        miner="test",
        title="late night",
        triggers=[Trigger(kind="state", entity_id="binary_sensor.motion", to_state="on")],
        conditions=[Condition(kind="time", after="22:00:00", before="06:00:00")],
        actions=[Action(service="light.turn_on", entity_id="light.hallway")],
    )
    store = SignalStore()
    series = SignalSeries("binary_sensor.motion")
    changes = []
    for day in range(8):
        ts = _minute(day, 23, 30)
        series.add(ts, "on")
        series.add(ts + 60, "off")
        changes.append(human("light.hallway", "on", ts + 5))
    store.add(series)

    result = backtest(candidate, changes, store, Options(), WINDOW)
    assert result.true_fires == 8
    assert result.passed is True, result.reason


def test_numeric_range_entry_only():
    trigger = Trigger(kind="numeric_state", entity_id="sensor.temp", above=10, below=20)
    store = SignalStore()
    series = SignalSeries("sensor.temp")
    base = _minute(1, 12)
    # 5 -> 25 jumps clean over the band; 25 -> 15 enters it; 15 -> 25 leaves it;
    # 25 -> 5 crosses both bounds downward without ever being inside.
    for offset, value in enumerate([5, 25, 15, 25, 5]):
        series.add(base + offset * 60, value)
    store.add(series)

    from amminer.backtest import _numeric_trigger_fires

    fires = _numeric_trigger_fires(trigger, store)
    assert fires == [base + 2 * 60]


def test_a_single_bound_numeric_trigger_is_unchanged():
    trigger = Trigger(kind="numeric_state", entity_id="sensor.lux", below=600)
    store = SignalStore()
    series = SignalSeries("sensor.lux")
    base = _minute(1, 12)
    for offset, value in enumerate([700, 500, 400, 800, 300]):
        series.add(base + offset * 60, value)
    store.add(series)

    from amminer.backtest import _numeric_trigger_fires

    assert _numeric_trigger_fires(trigger, store) == [base + 60, base + 4 * 60]


def test_an_unavailable_gap_re_arms_a_numeric_trigger():
    """Coming back into range after a restart is a fire in Home Assistant."""
    trigger = Trigger(kind="numeric_state", entity_id="sensor.temp", below=10)
    store = SignalStore()
    series = SignalSeries("sensor.temp")
    base = _minute(1, 12)
    series.add(base, 20)
    series.add(base + 60, 5)  # enters
    series.add(base + 120, "unavailable")  # restart
    series.add(base + 180, 5)  # back, still in range
    store.add(series)

    from amminer.backtest import _numeric_trigger_fires

    assert _numeric_trigger_fires(trigger, store) == [base + 60, base + 180]


def test_a_flapping_trigger_is_not_smoothed_into_a_pass():
    """Collapsing a burst is a courtesy to the reader; HA runs the action each time."""
    candidate = Candidate(
        miner="test",
        title="flapper",
        triggers=[Trigger(kind="state", entity_id="binary_sensor.motion", to_state="on")],
        actions=[Action(service="light.turn_on", entity_id="light.hallway")],
    )
    store = SignalStore()
    series = SignalSeries("binary_sensor.motion")
    changes = []
    for day in range(8):
        ts = _minute(day, 18)
        # One "event" to a person: twelve service calls to Home Assistant.
        for i in range(12):
            series.add(ts + i * 4, "on")
            series.add(ts + i * 4 + 2, "off")
        changes.append(human("light.hallway", "on", ts + 5))
    store.add(series)

    result = backtest(candidate, changes, store, Options(), WINDOW)

    assert result.total_fires == 8  # what a person would say happened
    assert result.burst_fires == 8 * 11  # what would really have run
    assert result.passed is False
    assert "flaps" in result.reason


def test_an_ordinary_rule_reports_no_bursts():
    candidate = Candidate(
        miner="test",
        title="calm",
        triggers=[Trigger(kind="state", entity_id="binary_sensor.motion", to_state="on")],
        actions=[Action(service="light.turn_on", entity_id="light.hallway")],
    )
    store = SignalStore()
    series = SignalSeries("binary_sensor.motion")
    changes = []
    for day in range(8):
        ts = _minute(day, 18)
        series.add(ts, "on")
        series.add(ts + 300, "off")
        changes.append(human("light.hallway", "on", ts + 5))
    store.add(series)

    result = backtest(candidate, changes, store, Options(), WINDOW)
    assert result.burst_fires == 0
    assert result.passed is True, result.reason


# --- risk tiering --------------------------------------------------------
def _repeating(service: str, entity_id: str, days: int = 10):
    """A clean, high-precision habit - the kind a light suggestion is made of."""
    candidate = Candidate(
        miner="test",
        title=f"{service} {entity_id}",
        triggers=[Trigger(kind="state", entity_id="binary_sensor.motion", to_state="on")],
        actions=[Action(service=service, entity_id=entity_id)],
    )
    store = SignalStore()
    series = SignalSeries("binary_sensor.motion")
    changes = []
    target = {"lock.lock": "locked", "lock.unlock": "unlocked",
              "cover.close_cover": "closed", "cover.open_cover": "open"}.get(service, "on")
    for day in range(days):
        ts = _minute(day, 18)
        series.add(ts - 30, "on")
        series.add(ts - 20, "off")
        changes.append(human(entity_id, target, ts))
    store.add(series)
    return candidate, changes, store


def test_a_lamp_habit_passes_on_the_ordinary_bar():
    candidate, changes, store = _repeating("light.turn_on", "light.hallway")
    result = backtest(candidate, changes, store, Options(), WINDOW)
    assert result.passed is True, result.reason
    assert result.risky_domains == []


def test_the_same_evidence_is_not_enough_for_a_lock():
    """Identical statistics, a different thing being controlled."""
    candidate, changes, store = _repeating("lock.lock", "lock.front_door")
    result = backtest(candidate, changes, store, Options(), WINDOW)
    assert result.precision == 1.0
    assert result.true_fires == 10
    assert result.risky_domains == ["lock"]
    assert result.passed is False
    assert "at least 12 needed for a lock" in result.reason


def test_a_lock_passes_once_the_evidence_is_there():
    candidate, changes, store = _repeating("lock.lock", "lock.front_door", days=20)
    result = backtest(candidate, changes, store, Options(), WINDOW)
    assert result.risky_domains == ["lock"]
    assert result.passed is True, result.reason


@pytest.mark.parametrize(
    "service, entity_id",
    [
        ("lock.unlock", "lock.front_door"),
        ("cover.open_cover", "cover.garage"),
        ("valve.open_valve", "valve.mains"),
    ],
)
def test_unlocking_and_opening_are_never_proposed_from_a_correlation(service, entity_id):
    """No precision makes a correlation a reason to unsecure a house."""
    candidate, changes, store = _repeating(service, entity_id, days=60)
    result = backtest(candidate, changes, store, Options(), WINDOW)
    assert result.passed is False
    assert result.simulated is False  # refused before the statistics are consulted
    assert "less secured" in result.reason


def test_the_refusal_can_be_lifted_deliberately():
    candidate, changes, store = _repeating("lock.unlock", "lock.front_door", days=20)
    options = Options(allow_security_actions=True)
    result = backtest(candidate, changes, store, options, WINDOW)
    assert result.simulated is True
    assert result.risky_domains == ["lock"]  # still on the stricter thresholds
    assert result.passed is True, result.reason


# --- temporal holdout validation -----------------------------------------
def test_split_window_is_a_wall_clock_split_of_the_final_fraction():
    """A row-count split would let a habit that got denser near the end of the
    window buy itself a bigger holdout; this must not move with the data."""
    split = split_window(WINDOW, Options(backtest_holdout_fraction=0.25))
    assert split.train == (WINDOW[0], WINDOW[0] + 21 * 86400.0)
    assert split.holdout == (WINDOW[0] + 21 * 86400.0, WINDOW[1])
    assert split.train_days == pytest.approx(21.0)
    assert split.holdout_days == pytest.approx(7.0)


def test_plain_backtest_never_validates_on_a_holdout():
    """Without asking for it, backtest() is exactly what it always was."""
    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=6, minutes=30)).timestamp())
        for d in range(DAYS)
    ]
    result = backtest(daily_candidate(), changes, SignalStore(), Options(), WINDOW)
    assert result.validation == "in_sample"
    assert result.holdout_evaluated is False
    assert result.segments is None


def test_a_habit_that_only_held_up_in_sample_is_rejected_by_holdout_gating():
    """The point of the whole feature: precision measured on the data a rule
    was mined from is not evidence it would hold up on data it was not - and
    a rule that only looks good in-sample must not be surfaced as if it does."""
    # The training slice (the first 21 days): a perfect daily habit.
    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=6, minutes=30)).timestamp())
        for d in range(21)
    ]
    # The holdout (the final 7 days): the habit has all but stopped.  The rule
    # itself is an unconditional daily trigger, so it keeps firing every one
    # of those 7 days regardless; the user only actually did it once.
    changes.append(
        human("light.kitchen", "on", (START + dt.timedelta(days=21, hours=6, minutes=30)).timestamp())
    )

    options = Options()
    in_sample = backtest(daily_candidate(), changes, SignalStore(), options, WINDOW)
    assert in_sample.passed is True, in_sample.reason
    assert in_sample.precision == pytest.approx(22 / 28)

    validated = backtest(
        daily_candidate(), changes, SignalStore(), options, WINDOW, validate_holdout=True
    )
    assert validated.validation == "holdout"
    assert validated.holdout_days == pytest.approx(7.0)
    assert validated.train_days == pytest.approx(21.0)
    assert validated.true_fires == 1  # only the one real occurrence in the holdout
    assert validated.false_fires == 6
    assert validated.passed is False
    assert "too little to judge" in validated.reason
    # The train/holdout/full breakdown is there for anyone who wants to see
    # that this did look fine in-sample - it just is not what gated it.
    assert validated.segments["full"]["passed"] is True
    assert validated.segments["holdout"]["true_fires"] == 1


def test_too_little_history_falls_back_to_in_sample_and_says_so():
    """A window too short for a trustworthy holdout must not silently pass
    (nor silently fail) on one - it falls back, and the fallback is labelled."""
    short_window = (START.timestamp(), (START + dt.timedelta(days=10)).timestamp())
    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=6, minutes=30)).timestamp())
        for d in range(10)
    ]
    options = Options()  # backtest_holdout_fraction=0.25 -> 2.5 days, below the 7-day floor
    plain = backtest(daily_candidate(), changes, SignalStore(), options, short_window)
    validated = backtest(
        daily_candidate(), changes, SignalStore(), options, short_window, validate_holdout=True
    )
    assert validated.validation == "in_sample"
    assert validated.holdout_evaluated is True
    assert validated.holdout_days == pytest.approx(2.5)
    assert validated.train_days == pytest.approx(7.5)
    # Falling back means exactly that: the same verdict a plain backtest gives.
    assert validated.passed == plain.passed
    assert validated.true_fires == plain.true_fires
    assert "not enough held-out history" in validated.validation_note()


def test_a_real_habit_with_enough_history_is_validated_on_the_holdout():
    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=6, minutes=30)).timestamp())
        for d in range(DAYS)
    ]
    result = backtest(
        daily_candidate(), changes, SignalStore(), Options(), WINDOW, validate_holdout=True
    )
    assert result.validation == "holdout"
    assert result.holdout_days == pytest.approx(7.0)
    assert result.passed is True, result.reason
    assert result.true_fires == 7  # only the holdout week counts now
    assert "Validated on 7 days" in result.validation_note()


def test_backtest_all_defaults_to_holdout_validation():
    good = daily_candidate()
    changes = [
        human("light.kitchen", "on", (START + dt.timedelta(days=d, hours=6, minutes=30)).timestamp())
        for d in range(DAYS)
    ]
    passed, _rejected = backtest_all([good], changes, SignalStore(), Options(), WINDOW)
    assert passed[0].backtest["validation"] == "holdout"
