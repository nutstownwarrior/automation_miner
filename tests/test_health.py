"""Post-deployment health of applied automations (amminer.health).

Every scenario here builds its own tiny, hand-controlled history rather than
reusing the big synthetic fixture: the point of each test is a specific,
checkable number (predicted fires, actual fires, override rate), and the
synthetic fixture's randomness would make that number a moving target instead
of a fact the test can assert on.
"""

from __future__ import annotations

from amminer import health
from amminer.automations import normalise_automation
from amminer.config import Options
from amminer.enrich.signals import SignalSeries, SignalStore
from amminer.miners.base import Action, Candidate, Trigger
from amminer.recorderdb.models import OverrideEvent, RecorderEvent
from amminer.runner import candidate_from_payload

DAY = 86400.0
AUTOMATION_ID = "amminer_kitchen1"
ENTITY_ID = "automation.kitchen_light"


def _candidate() -> Candidate:
    return Candidate(
        miner="time_of_day",
        title="Kitchen light at dusk",
        triggers=[Trigger(kind="state", entity_id="binary_sensor.motion", to_state="on")],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )


SHIPPED_CONFIG = {
    "id": AUTOMATION_ID,
    "alias": "Kitchen light at dusk",
    "trigger": [{"platform": "state", "entity_id": "binary_sensor.motion", "to": "on"}],
    "action": [{"service": "light.turn_on", "entity_id": "light.kitchen"}],
    "mode": "single",
}


def _applied_row(applied_ts: float, candidate: Candidate | None = None) -> dict:
    candidate = candidate or _candidate()
    return {
        "automation_id": AUTOMATION_ID,
        "suggestion_id": "sugg1",
        "title": candidate.title,
        "applied_ts": applied_ts,
        "candidate_payload": candidate.as_dict(),
        "shipped_config": dict(SHIPPED_CONFIG),
    }


def _existing(entity_id: str | None = ENTITY_ID, enabled: bool = True, edited: bool = False):
    raw = dict(SHIPPED_CONFIG)
    if edited:
        raw["action"] = [{"service": "light.turn_on", "entity_id": "light.bedroom"}]
    automation = normalise_automation(raw)
    automation.entity_id = entity_id
    automation.enabled = enabled
    return automation


def _fire_series(entity_id: str, fire_times: list[float]) -> SignalSeries:
    """A signal series that fires (transitions to 'on') at exactly *fire_times*."""
    series = SignalSeries(entity_id)
    for ts in fire_times:
        series.add(ts - 1.0, "off")
        series.add(ts, "on")
    return series.finalise()


def _store(fire_times: list[float]) -> SignalStore:
    store = SignalStore()
    store.add(_fire_series("binary_sensor.motion", fire_times))
    return store


def _firings(times: list[float]) -> list[RecorderEvent]:
    """``automation_triggered`` events - what amminer.health counts as "it
    really fired", exactly as amminer.recorderdb.queries loads them for
    causality classification."""
    return [
        RecorderEvent(
            event_type="automation_triggered",
            ts=ts,
            data={"entity_id": ENTITY_ID},
        )
        for ts in times
    ]


def _override(ts: float) -> OverrideEvent:
    return OverrideEvent(
        entity_id="light.kitchen",
        ts=ts,
        automation_entity_id=ENTITY_ID,
        automation_state="on",
        human_state="off",
        delay_seconds=15.0,
    )


OPTIONS = Options()


def test_candidate_payload_round_trips_through_the_store_shape():
    """The snapshot saved at apply time must be exactly what
    amminer.runner.candidate_from_payload can rebuild - health.evaluate relies
    on this without re-checking it."""
    payload = _candidate().as_dict()
    rebuilt = candidate_from_payload(payload)
    assert rebuilt.triggers[0].entity_id == "binary_sensor.motion"
    assert rebuilt.actions[0].entity_id == "light.kitchen"


# --- a healthy automation ------------------------------------------------
def test_an_automation_that_performs_well_is_healthy():
    window = (0.0, 20 * DAY)
    fire_times = [d * DAY + 100 for d in range(15)]  # 15 predicted fires
    existing = {AUTOMATION_ID: _existing()}
    events = _firings(fire_times)  # it really fired every one of those times
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, events, _store(fire_times), [], OPTIONS, window
    )
    assert result.status == health.STATUS_ACTIVE
    assert result.verdict == health.VERDICT_HEALTHY
    assert result.predicted_fires == 15
    assert result.actual_fires == 15
    assert result.overrides == 0
    assert result.override_rate == 0.0
    assert result.evidence  # never blank
    assert "no action needed" in result.recommendation.lower()


# --- an automation the user keeps overriding ------------------------------
def test_an_automation_the_user_keeps_overriding_is_flagged():
    window = (0.0, 20 * DAY)
    fire_times = [d * DAY + 100 for d in range(10)]
    overrides = [_override(ts + 30.0) for ts in fire_times[:7]]  # 7 of 10 undone
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0),
        existing,
        _firings(fire_times),
        _store(fire_times),
        overrides,
        OPTIONS,
        window,
    )
    assert result.actual_fires == 10
    assert result.overrides == 7
    assert result.override_rate == 0.7
    assert result.verdict == health.VERDICT_OVERRIDDEN
    assert "retire" in result.recommendation.lower()


def test_a_moderately_overridden_automation_is_noisy_not_overridden():
    window = (0.0, 20 * DAY)
    fire_times = [d * DAY + 100 for d in range(10)]
    overrides = [_override(ts + 30.0) for ts in fire_times[:4]]  # 4 of 10 -> 40%
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0),
        existing,
        _firings(fire_times),
        _store(fire_times),
        overrides,
        OPTIONS,
        window,
    )
    assert result.override_rate == 0.4
    assert result.verdict == health.VERDICT_NOISY
    assert "retun" in result.recommendation.lower()


# --- BLOCKER 1: an override rate over a handful of fires is not a verdict --
def test_one_fire_and_one_revert_is_not_enough_to_say_retire_it():
    """override_rate=1.0 from n=1 must never reach VERDICT_OVERRIDDEN - that
    is the most severe, most actionable verdict in this feature, and it must
    rest on more than a single coincidence."""
    window = (0.0, 20 * DAY)
    fire_times = [5 * DAY]
    overrides = [_override(fire_times[0] + 30.0)]  # the only fire, undone
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, _firings(fire_times), _store(fire_times),
        overrides, OPTIONS, window,
    )
    assert result.actual_fires == 1
    assert result.verdict == health.VERDICT_INSUFFICIENT
    assert result.verdict != health.VERDICT_OVERRIDDEN
    assert "too few" in result.evidence.lower()


def test_one_fire_never_overridden_is_also_not_yet_healthy():
    """The same floor protects the reassuring verdict, not only the alarming
    one - "no action needed" from a single fire is exactly as overconfident
    as "retire it" from one."""
    window = (0.0, 20 * DAY)
    fire_times = [5 * DAY]
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, _firings(fire_times), _store(fire_times),
        [], OPTIONS, window,
    )
    assert result.verdict == health.VERDICT_INSUFFICIENT
    assert result.verdict != health.VERDICT_HEALTHY


def test_just_below_the_fires_floor_is_still_insufficient():
    assert OPTIONS.health_min_fires_for_verdict == 5
    window = (0.0, 20 * DAY)
    fire_times = [d * DAY for d in range(4)]  # one short of the floor
    overrides = [_override(ts + 30.0) for ts in fire_times]  # every one undone
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, _firings(fire_times), _store(fire_times),
        overrides, OPTIONS, window,
    )
    assert result.actual_fires == 4
    assert result.verdict == health.VERDICT_INSUFFICIENT


def test_at_the_fires_floor_a_verdict_is_shown():
    window = (0.0, 20 * DAY)
    fire_times = [d * DAY for d in range(5)]  # exactly the floor
    overrides = [_override(ts + 30.0) for ts in fire_times]  # every one undone
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, _firings(fire_times), _store(fire_times),
        overrides, OPTIONS, window,
    )
    assert result.actual_fires == 5
    assert result.verdict == health.VERDICT_OVERRIDDEN


def test_overridden_never_rests_on_less_evidence_than_noisy():
    """Both verdicts are read off the same actual_fires count, gated by the
    same floor - overridden cannot fire on a smaller n than noisy needs."""
    window = (0.0, 20 * DAY)
    fire_times = [d * DAY for d in range(5)]
    # Exactly at the floor, at the overridden threshold.
    overrides = [_override(ts + 30.0) for ts in fire_times[:3]]  # 3 of 5 -> 60%
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, _firings(fire_times), _store(fire_times),
        overrides, OPTIONS, window,
    )
    assert result.actual_fires == OPTIONS.health_min_fires_for_verdict
    assert result.verdict == health.VERDICT_OVERRIDDEN


# --- GAP: a large predicted-vs-actual shortfall is its own verdict --------
def test_a_severe_shortfall_with_no_overrides_is_underperforming_not_healthy():
    """predicted=50, actual=6, override_rate=0 used to read as healthy -
    there was so little happening there was nothing to override. The
    shortfall itself must be surfaced."""
    window = (0.0, 60 * DAY)
    predicted_times = [d * DAY + 100 for d in range(50)]  # the habit still happens daily
    actual_times = [d * DAY + 100 for d in (5, 15, 25, 30, 40, 50)]  # only 6 of it
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, _firings(actual_times),
        _store(predicted_times), [], OPTIONS, window,
    )
    assert result.predicted_fires == 50
    assert result.actual_fires == 6
    assert result.override_rate == 0.0
    assert result.verdict == health.VERDICT_UNDERPERFORMING
    assert result.verdict != health.VERDICT_HEALTHY
    assert "50" in result.evidence and "6" in result.evidence


def test_firing_nearly_as_often_as_predicted_is_still_healthy():
    """The shortfall floor must not turn ordinary variance into a verdict -
    firing 80% as often as predicted, with nothing overridden, is healthy."""
    window = (0.0, 60 * DAY)
    predicted_times = [d * DAY + 100 for d in range(50)]
    actual_times = predicted_times[:40]  # 80% of predicted
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, _firings(actual_times),
        _store(predicted_times), [], OPTIONS, window,
    )
    assert result.verdict == health.VERDICT_HEALTHY


def test_zero_predicted_fires_cannot_divide_by_zero_or_read_as_shortfall():
    """The rule never would have fired by our own simulation, yet Home
    Assistant ran it anyway (a manual trigger, or a real-world condition our
    simulator does not model) - predicted=0 must not raise, and must not be
    read as a shortfall (there is nothing to have fallen short of)."""
    window = (0.0, 20 * DAY)
    actual_times = [d * DAY for d in range(6)]  # clears the fires-for-verdict floor
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, _firings(actual_times),
        _store([]), [], OPTIONS, window,  # nothing in the store -> predicted == 0
    )
    assert result.predicted_fires == 0
    assert result.verdict != health.VERDICT_UNDERPERFORMING


# --- a dormant automation --------------------------------------------------
def test_an_automation_that_stopped_firing_is_dormant():
    window = (0.0, 20 * DAY)
    predicted_times = [d * DAY + 100 for d in range(5)]  # the habit is still there...
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0),
        existing,
        [],  # ...but the automation never recorded a firing
        _store(predicted_times),
        [],
        OPTIONS,
        window,
    )
    assert result.actual_fires == 0
    assert result.predicted_fires == 5
    assert result.verdict == health.VERDICT_DORMANT
    assert "retire" in result.recommendation.lower() or "check why" in result.recommendation.lower()


def test_near_silence_on_both_sides_is_insufficient_not_dormant():
    """One predicted fire against zero actual ones is not evidence of
    dormancy - it is a quiet couple of weeks. The floor exists precisely so
    this is not misreported as a confident, wrong verdict."""
    window = (0.0, 20 * DAY)
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0),
        existing,
        [],
        _store([10 * DAY]),  # a single predicted fire
        [],
        OPTIONS,
        window,
    )
    assert result.verdict == health.VERDICT_INSUFFICIENT
    assert "not enough" in result.evidence.lower() or "too quiet" in result.evidence.lower()


# --- applied too recently to judge -----------------------------------------
def test_an_automation_applied_two_days_ago_says_not_enough_data():
    window = (0.0, 20 * DAY)
    applied_ts = window[1] - 2 * DAY
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=applied_ts), existing, [], _store([]), [], OPTIONS, window
    )
    assert result.verdict == health.VERDICT_INSUFFICIENT
    assert result.status == health.STATUS_ACTIVE
    assert "not enough" in result.evidence.lower()
    # Never a confident number resting on two days of history.
    assert result.predicted_fires is None
    assert result.actual_fires is None


# --- deleted -----------------------------------------------------------
def test_a_deleted_automation_is_reported_as_gone():
    result = health.evaluate(
        _applied_row(applied_ts=0.0), {}, [], _store([]), [], OPTIONS, (0.0, 20 * DAY)
    )
    assert result.status == health.STATUS_DELETED
    assert result.verdict == health.VERDICT_NA
    assert "deleted" in result.evidence.lower() or "no automation" in result.evidence.lower()
    assert "nothing to do" in result.recommendation.lower()


# --- edited beyond recognition ---------------------------------------------
def test_an_edited_automation_is_reported_as_edited_not_scored():
    existing = {AUTOMATION_ID: _existing(edited=True)}
    result = health.evaluate(
        _applied_row(applied_ts=0.0),
        existing,
        _firings([1 * DAY, 2 * DAY]),
        _store([1 * DAY, 2 * DAY]),
        [],
        OPTIONS,
        (0.0, 20 * DAY),
    )
    assert result.status == health.STATUS_EDITED
    assert result.verdict == health.VERDICT_NA
    assert result.predicted_fires is None  # never scored against the wrong rule
    assert "changed" in result.evidence.lower()


def test_a_states_only_automation_is_not_mistaken_for_edited():
    """When the YAML cannot be read, an empty ``raw`` must not look like an
    edit - it is missing information, not a detected change."""
    existing = _existing()
    existing.source = "states-only"
    existing.raw = {}
    result = health.evaluate(
        _applied_row(applied_ts=0.0),
        {AUTOMATION_ID: existing},
        _firings([1 * DAY] * 5),
        _store([1 * DAY] * 1),
        [],
        OPTIONS,
        (0.0, 20 * DAY),
    )
    assert result.status != health.STATUS_EDITED
    assert any("could not be read" in note for note in result.notes)


# --- disabled ---------------------------------------------------------
def test_a_disabled_automation_is_reported_as_disabled():
    existing = {AUTOMATION_ID: _existing(enabled=False)}
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, [], _store([]), [], OPTIONS, (0.0, 20 * DAY)
    )
    assert result.status == health.STATUS_DISABLED
    assert result.verdict == health.VERDICT_NA
    assert "switched off" in result.evidence.lower()


def test_an_unsimulatable_trigger_reports_why_instead_of_a_number():
    unsimulatable = Candidate(
        miner="time_of_day",
        title="odd rule",
        triggers=[Trigger(kind="sun", event="sunset")],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )
    existing = {AUTOMATION_ID: _existing()}
    result = health.evaluate(
        _applied_row(applied_ts=0.0, candidate=unsimulatable),
        existing, [], _store([]), [], OPTIONS, (0.0, 20 * DAY),
    )
    assert result.verdict == health.VERDICT_INSUFFICIENT
    assert "could not replay" in result.evidence.lower()


def test_an_automation_with_no_live_entity_is_unmeasurable_not_dormant():
    existing = {AUTOMATION_ID: _existing(entity_id=None)}
    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing, [], _store([]), [], OPTIONS, (0.0, 20 * DAY)
    )
    assert result.verdict == health.VERDICT_INSUFFICIENT
    assert result.actual_fires is None


# --- evaluate_all persists to the store -------------------------------
def test_evaluate_all_persists_a_health_row_per_applied_automation(store, monkeypatch):
    # record_applied_automation always stamps "now" as applied_ts - pin what
    # that means so this test's window can put it safely in the past, the
    # same way a real automation applied three weeks ago would be.
    applied_at = 1_000_000.0
    monkeypatch.setattr("amminer.store.db.time.time", lambda: applied_at)
    store.record_applied_automation(
        AUTOMATION_ID, "sugg1", "Kitchen light at dusk",
        _candidate().as_dict(), SHIPPED_CONFIG,
    )
    window = (applied_at, applied_at + 20 * DAY)
    existing = [_existing()]
    fire_times = [applied_at + d * DAY + 100 for d in range(10)]
    results = health.evaluate_all(
        store, existing, _firings(fire_times), _store(fire_times), [], OPTIONS, window
    )
    assert len(results) == 1
    assert results[0].verdict == health.VERDICT_HEALTHY
    stored = store.get_applied_automation(AUTOMATION_ID)
    assert stored["health"]["verdict"] == health.VERDICT_HEALTHY
    assert stored["health"]["payload"]["actual_fires"] == 10


# --- BLOCKER 2: one bad row must not blank every other automation's health -
def test_a_corrupt_payload_costs_only_its_own_row(store, monkeypatch):
    """A legacy or corrupted candidate_payload must not raise its way out of
    evaluate_all and take every other automation's verdict down with it."""
    applied_at = 1_000_000.0
    monkeypatch.setattr("amminer.store.db.time.time", lambda: applied_at)
    store.record_applied_automation(
        AUTOMATION_ID, "sugg1", "Kitchen light at dusk", _candidate().as_dict(), SHIPPED_CONFIG,
    )
    # A shape candidate_from_payload cannot rebuild: an unknown field on a
    # trigger dict, e.g. left over from a schema this add-on no longer
    # writes. Trigger(**t) raises TypeError on this, not something caught
    # inside candidate_from_payload itself.
    corrupt_payload = {
        "miner": "time_of_day",
        "title": "Broken",
        "triggers": [{"kind": "state", "no_longer_a_field": "x"}],
        "actions": [{"service": "light.turn_on", "entity_id": "light.bedroom"}],
    }
    broken_shipped_config = {
        "id": "broken1",
        "trigger": [{"platform": "state", "entity_id": "sensor.x", "to": "on"}],
        "action": [{"service": "light.turn_on", "entity_id": "light.bedroom"}],
    }
    store.record_applied_automation(
        "broken1", "sugg2", "Broken automation", corrupt_payload, broken_shipped_config,
    )
    window = (applied_at, applied_at + 20 * DAY)
    fire_times = [applied_at + d * DAY + 100 for d in range(10)]
    # The live automation matches what was shipped (not "edited"), so
    # evaluate() gets far enough to attempt candidate_from_payload - that is
    # the call this test means to prove is isolated, not something an
    # earlier, unrelated "edited" return would mask.
    broken_existing = normalise_automation(broken_shipped_config)
    broken_existing.entity_id = "automation.broken"
    existing = [_existing(), broken_existing]

    results = health.evaluate_all(
        store, existing, _firings(fire_times), _store(fire_times), [], OPTIONS, window
    )

    assert len(results) == 2
    by_id = {r.automation_id: r for r in results}
    # The good row is judged normally - not degraded, not skipped.
    assert by_id[AUTOMATION_ID].verdict == health.VERDICT_HEALTHY
    # The bad row reports its own failure honestly - never silently dropped,
    # never mislabelled as "not enough data" (a different claim).
    broken = by_id["broken1"]
    assert broken.status == health.STATUS_ERROR
    assert broken.verdict == health.VERDICT_NA
    assert "could not check" in broken.evidence.lower()

    # And both are actually persisted - the good one is not somehow lost too.
    assert store.get_applied_automation(AUTOMATION_ID)["health"]["verdict"] == health.VERDICT_HEALTHY
    assert store.get_applied_automation("broken1")["health"]["status"] == health.STATUS_ERROR


def test_health_as_dict_never_omits_evidence_or_recommendation():
    """Every verdict path must leave a plain sentence behind, never a blank
    one a UI would render as an empty line."""
    scenarios = [
        health.evaluate(
            _applied_row(applied_ts=0.0), {}, [], _store([]), [], OPTIONS, (0.0, 20 * DAY)
        ),
        health.evaluate(
            _applied_row(applied_ts=0.0),
            {AUTOMATION_ID: _existing(enabled=False)},
            [],
            _store([]),
            [],
            OPTIONS,
            (0.0, 20 * DAY),
        ),
    ]
    for result in scenarios:
        data = result.as_dict()
        assert data["evidence"].strip()
        assert data["recommendation"].strip()


# --- MINOR: the marker's failure modes --------------------------------
def test_stripping_the_id_field_degrades_to_deleted_not_a_crash():
    """A user who deletes the ``id:`` line from an automation's YAML by hand
    breaks the one thing this add-on recognises it by. load_existing_automations
    (amminer.automations) gives such an automation ``id=None``, and
    evaluate_all's own id-keyed lookup (``if a.id``) already excludes falsy
    ids - so this marker-less automation is simply never in existing_by_id,
    the same as if it had never existed. That is a safe failure: no crash,
    and a plain, honest status rather than a wrong one."""
    existing = _existing()
    existing.id = None  # what normalise_automation produces for a missing "id:"
    existing_by_id = {a.id: a for a in [existing] if a.id}  # evaluate_all's own filter
    assert existing_by_id == {}  # the stripped automation is invisible, not mismatched

    result = health.evaluate(
        _applied_row(applied_ts=0.0), existing_by_id, [], _store([]), [], OPTIONS,
        (0.0, 20 * DAY),
    )
    assert result.status == health.STATUS_DELETED
    assert result.verdict == health.VERDICT_NA


def test_a_duplicated_marker_does_not_crash_evaluate_all(store, monkeypatch):
    """Two live automations sharing one id (a hand copy-pasted YAML block,
    id included) is a Home Assistant misconfiguration this add-on did not
    cause and cannot fully resolve - dict-keyed existing_by_id can only hold
    one of them. The bar here is that evaluate_all still completes and
    persists a result, rather than raising or silently losing the row."""
    applied_at = 1_000_000.0
    monkeypatch.setattr("amminer.store.db.time.time", lambda: applied_at)
    store.record_applied_automation(
        AUTOMATION_ID, "sugg1", "Kitchen light at dusk", _candidate().as_dict(), SHIPPED_CONFIG,
    )
    window = (applied_at, applied_at + 20 * DAY)
    # Two ExistingAutomation entries, same id - as if the same YAML block
    # (id included) had been pasted into two separate files.
    duplicate = _existing(entity_id="automation.kitchen_light_copy")
    existing = [_existing(), duplicate]

    results = health.evaluate_all(
        store, existing, [], _store([]), [], OPTIONS, window
    )

    assert len(results) == 1
    assert results[0].automation_id == AUTOMATION_ID
    # Whichever of the two duplicates survived the id collision, evaluate()
    # ran to completion against it (both are enabled, unedited) rather than
    # raising.
    assert results[0].status == health.STATUS_ACTIVE
    stored = store.get_applied_automation(AUTOMATION_ID)
    assert stored["health"] is not None
