"""Causality classification and override detection - the key differentiator."""

from __future__ import annotations

from amminer.recorderdb import causality
from amminer.recorderdb.causality import build_index, detect_overrides
from amminer.recorderdb.models import Cause, RecorderEvent, StateChange


def _change(entity_id, state, ts, **kwargs) -> StateChange:
    kwargs.setdefault("old_state", "off" if state == "on" else "on")
    return StateChange(entity_id=entity_id, state=state, ts=ts, **kwargs)


def test_user_context_means_human():
    change = _change("light.a", "on", 100.0, context_id="c1", context_user_id="u1")
    causality.classify([change], build_index([change]))
    assert change.cause is Cause.HUMAN


def test_automation_event_context_means_automation():
    event = RecorderEvent("automation_triggered", 99.0, {"entity_id": "automation.x"}, context_id="c1")
    change = _change("light.a", "on", 100.0, context_id="c1")
    causality.classify([change], build_index([change], [event]))
    assert change.cause is Cause.AUTOMATION
    assert change.origin_entity_id == "automation.x"


def test_script_context_is_recognised():
    event = RecorderEvent("script_started", 99.0, {"entity_id": "script.good_night"}, context_id="c1")
    change = _change("light.a", "off", 100.0, context_id="c1")
    causality.classify([change], build_index([change], [event]))
    assert change.cause is Cause.SCRIPT
    assert change.origin_entity_id == "script.good_night"


def test_multi_hop_parent_chain_resolves_to_the_root_automation():
    """A scene inside a script inside an automation still attributes correctly."""
    event = RecorderEvent("automation_triggered", 90.0, {"entity_id": "automation.evening"}, context_id="root")
    chain = [
        _change("script.evening", "on", 91.0, context_id="c1", context_parent_id="root"),
        _change("scene.cosy", "on", 92.0, context_id="c2", context_parent_id="c1"),
        _change("light.a", "on", 93.0, context_id="c3", context_parent_id="c2"),
    ]
    causality.classify(chain, build_index(chain, [event]))
    assert [c.cause for c in chain] == [Cause.AUTOMATION] * 3
    assert chain[-1].origin_entity_id == "automation.evening"
    assert chain[-1].chain_depth == 3


def test_human_parent_chain_resolves_to_human():
    parent = _change("scene.movie", "on", 100.0, context_id="p1", context_user_id="u1")
    child = _change("light.a", "off", 101.0, context_id="c1", context_parent_id="p1")
    causality.classify([parent, child], build_index([parent, child]))
    assert child.cause is Cause.HUMAN


def test_no_context_means_device():
    change = _change("sensor.temp", "21.5", 100.0)
    causality.classify([change], build_index([change]))
    assert change.cause is Cause.DEVICE


def test_cyclic_parent_chain_terminates():
    a = _change("light.a", "on", 100.0, context_id="c1", context_parent_id="c2")
    b = _change("light.b", "on", 101.0, context_id="c2", context_parent_id="c1")
    causality.classify([a, b], build_index([a, b]))
    assert a.cause is Cause.DEVICE  # unresolved, but no infinite loop


def test_excluded_user_is_not_treated_as_human():
    change = _change("light.a", "on", 100.0, context_id="c1", context_user_id="service-account")
    causality.classify([change], build_index([change]), excluded_users=["service-account"])
    assert change.cause is not Cause.HUMAN


def test_index_reports_missing_user_context():
    change = _change("light.a", "on", 100.0, context_id="c1")
    assert build_index([change]).has_user_context is False
    with_user = _change("light.b", "on", 100.0, context_id="c2", context_user_id="u1")
    assert build_index([change, with_user]).has_user_context is True


# --- overrides ---------------------------------------------------------
def _automated(entity_id, state, ts, origin="automation.x"):
    change = _change(entity_id, state, ts)
    change.cause = Cause.AUTOMATION
    change.origin_entity_id = origin
    return change


def _human(entity_id, state, ts):
    change = _change(entity_id, state, ts)
    change.cause = Cause.HUMAN
    return change


def test_override_detected_within_the_window():
    changes = [_automated("light.a", "off", 100.0), _human("light.a", "on", 130.0)]
    overrides = detect_overrides(changes, window_seconds=120)
    assert len(overrides) == 1
    assert overrides[0].automation_entity_id == "automation.x"
    assert overrides[0].automation_state == "off"
    assert overrides[0].human_state == "on"
    assert overrides[0].delay_seconds == 30.0


def test_no_override_outside_the_window():
    changes = [_automated("light.a", "off", 100.0), _human("light.a", "on", 400.0)]
    assert detect_overrides(changes, window_seconds=120) == []


def test_human_agreeing_is_not_an_override():
    changes = [_automated("light.a", "off", 100.0), _human("light.a", "off", 110.0)]
    assert detect_overrides(changes, window_seconds=120) == []


def test_override_is_per_entity():
    changes = [_automated("light.a", "off", 100.0), _human("light.b", "on", 110.0)]
    assert detect_overrides(changes, window_seconds=120) == []


def test_only_the_first_correction_counts_once():
    changes = [
        _automated("light.a", "off", 100.0),
        _human("light.a", "on", 110.0),
        _human("light.a", "off", 115.0),
    ]
    assert len(detect_overrides(changes, window_seconds=120)) == 1


def test_override_summary_aggregates_preferred_state():
    changes = []
    for i in range(3):
        base = 1000.0 * i
        changes += [_automated("light.a", "off", base), _human("light.a", "on", base + 20)]
    overrides = detect_overrides(changes, window_seconds=120)
    summary = causality.override_summary(overrides)
    assert summary["automation.x"]["count"] == 3
    assert summary["automation.x"]["preferred_states"] == {"on": 3}


# --- against the synthetic fixture ------------------------------------
def test_fixture_overrides_are_recovered(classified, fixture_db):
    _path, truth = fixture_db
    changes, index = classified
    assert index.has_user_context is True

    stats = causality.causality_stats(changes)
    assert stats.get("human", 0) > 0
    assert stats.get("automation", 0) > 0

    overrides = detect_overrides(changes, window_seconds=120)
    expected = truth.overrides
    # Every detected override must be one we actually injected.
    injected = {(round(o["ts"], 1), o["entity_id"]) for o in expected}
    for override in overrides:
        assert (round(override.ts, 1), override.entity_id) in injected
    # And we must find essentially all of them (the last day can fall outside).
    assert len(overrides) >= len(expected) - 1
    assert all(o.automation_entity_id == "automation.bedtime_dim" for o in overrides)
