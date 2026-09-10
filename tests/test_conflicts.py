"""Conflict, loop, redundancy and shared-device-race detection."""

from __future__ import annotations

from amminer.automations import normalise_automation
from amminer.conflicts import (
    DeviceGraph,
    annotate_candidates,
    audit_existing,
    check_candidate,
    has_blocking_conflict,
)
from amminer.miners.base import Action, Candidate, Trigger


def automation(alias, trigger, action, **extra):
    raw = {"id": alias, "alias": alias, "trigger": trigger, "action": action}
    raw.update(extra)
    return normalise_automation(raw)


def candidate(service, entity_id, at=None, trigger_entity=None, to_state=None):
    triggers = []
    if at:
        triggers.append(Trigger(kind="time", at=at))
    if trigger_entity:
        triggers.append(Trigger(kind="state", entity_id=trigger_entity, to_state=to_state))
    return Candidate(
        miner="test",
        title="candidate",
        triggers=triggers,
        actions=[Action(service=service, entity_id=entity_id)],
    )


# --- normalisation -----------------------------------------------------
def test_normalise_extracts_targets_from_every_shorthand():
    parsed = automation(
        "a",
        [{"platform": "time", "at": "22:00:00"}],
        [
            {"service": "light.turn_off", "target": {"entity_id": ["light.a", "light.b"]}},
            {"service": "switch.turn_on", "entity_id": "switch.c"},
        ],
    )
    assert parsed.action_entities == ["light.a", "light.b", "switch.c"]
    assert parsed.trigger_times == ["22:00:00"]
    assert ("light.a", "off") in parsed.targets()
    assert ("switch.c", "on") in parsed.targets()


def test_normalise_walks_choose_branches():
    parsed = automation(
        "a",
        [{"platform": "state", "entity_id": "person.alex", "to": "home"}],
        [
            {
                "choose": [
                    {
                        "conditions": [{"condition": "sun", "after": "sunset"}],
                        "sequence": [{"service": "light.turn_on", "entity_id": "light.hall"}],
                    }
                ],
                "default": [{"service": "switch.turn_on", "entity_id": "switch.fan"}],
            }
        ],
    )
    assert set(parsed.action_entities) == {"light.hall", "switch.fan"}
    assert parsed.trigger_entities == ["person.alex"]


def test_normalise_accepts_modern_plural_and_action_keys():
    parsed = normalise_automation(
        {
            "id": "x",
            "alias": "modern",
            "triggers": [{"trigger": "state", "entity_id": "binary_sensor.m", "to": "on"}],
            "actions": [{"action": "light.turn_on", "target": {"entity_id": "light.a"}}],
        }
    )
    assert parsed.trigger_entities == ["binary_sensor.m"]
    assert parsed.action_entities == ["light.a"]


# --- value inconsistency ----------------------------------------------
def test_value_inconsistency_on_overlapping_time_triggers():
    existing = [
        automation(
            "night off",
            [{"platform": "time", "at": "22:00:00"}],
            [{"service": "light.turn_off", "entity_id": "light.bedroom"}],
        )
    ]
    conflicts = check_candidate(candidate("light.turn_on", "light.bedroom", at="22:15:00"), existing)
    kinds = {c.kind for c in conflicts}
    assert "value_inconsistency" in kinds
    assert any(c.severity == "error" for c in conflicts)


def test_no_conflict_when_times_are_far_apart():
    existing = [
        automation(
            "morning off",
            [{"platform": "time", "at": "07:00:00"}],
            [{"service": "light.turn_off", "entity_id": "light.bedroom"}],
        )
    ]
    conflicts = check_candidate(candidate("light.turn_on", "light.bedroom", at="22:15:00"), existing)
    assert not [c for c in conflicts if c.kind == "value_inconsistency"]


def test_value_inconsistency_on_shared_trigger_entity():
    existing = [
        automation(
            "away off",
            [{"platform": "state", "entity_id": "person.alex", "to": "not_home"}],
            [{"service": "light.turn_off", "entity_id": "light.hall"}],
        )
    ]
    conflicts = check_candidate(
        candidate("light.turn_on", "light.hall", trigger_entity="person.alex", to_state="home"),
        existing,
    )
    assert any(c.kind == "value_inconsistency" for c in conflicts)


def test_disabled_automation_does_not_conflict():
    existing = [
        automation(
            "night off",
            [{"platform": "time", "at": "22:00:00"}],
            [{"service": "light.turn_off", "entity_id": "light.bedroom"}],
        )
    ]
    existing[0].enabled = False
    conflicts = check_candidate(candidate("light.turn_on", "light.bedroom", at="22:15:00"), existing)
    assert not [c for c in conflicts if c.kind == "value_inconsistency"]


# --- redundancy --------------------------------------------------------
def test_redundancy_detected_for_near_duplicates():
    existing = [
        automation(
            "already exists",
            [{"platform": "time", "at": "06:30:00"}],
            [{"service": "light.turn_on", "entity_id": "light.kitchen"}],
        )
    ]
    conflicts = check_candidate(candidate("light.turn_on", "light.kitchen", at="06:35:00"), existing)
    assert any(c.kind == "redundancy" for c in conflicts)


# --- loops -------------------------------------------------------------
def test_loop_detected_through_an_existing_automation():
    """Candidate turns on a light; a rule watching that light turns on a switch
    which is the candidate's own trigger."""
    existing = [
        automation(
            "chain",
            [{"platform": "state", "entity_id": "light.hall", "to": "on"}],
            [{"service": "switch.turn_on", "entity_id": "switch.trigger"}],
        )
    ]
    candidate_rule = Candidate(
        miner="test",
        title="loop",
        triggers=[Trigger(kind="state", entity_id="switch.trigger", to_state="on")],
        actions=[Action(service="light.turn_on", entity_id="light.hall")],
    )
    conflicts = check_candidate(candidate_rule, existing)
    loops = [c for c in conflicts if c.kind == "loop"]
    assert loops and loops[0].severity == "error"


def test_no_false_loop_on_unrelated_rules():
    existing = [
        automation(
            "unrelated",
            [{"platform": "state", "entity_id": "binary_sensor.door", "to": "on"}],
            [{"service": "light.turn_on", "entity_id": "light.porch"}],
        )
    ]
    candidate_rule = Candidate(
        miner="test",
        title="fine",
        triggers=[Trigger(kind="state", entity_id="person.alex", to_state="home")],
        actions=[Action(service="light.turn_on", entity_id="light.hall")],
    )
    assert not [c for c in check_candidate(candidate_rule, existing) if c.kind == "loop"]


# --- self conflicts ----------------------------------------------------
def test_self_contradicting_candidate_flagged():
    rule = Candidate(
        miner="test",
        title="both",
        triggers=[Trigger(kind="time", at="06:00:00")],
        actions=[
            Action(service="light.turn_on", entity_id="light.a"),
            Action(service="light.turn_off", entity_id="light.a"),
        ],
    )
    conflicts = check_candidate(rule, [])
    assert any(c.kind == "value_inconsistency" and c.severity == "error" for c in conflicts)


# --- shared device races ----------------------------------------------
def test_shared_device_race_detected():
    class Info:
        def __init__(self, entity_id, device_id):
            self.entity_id = entity_id
            self.device_id = device_id
            self.area_id = None

    class Resolver:
        entities = {
            "light.lamp": Info("light.lamp", "dev1"),
            "switch.lamp_relay": Info("switch.lamp_relay", "dev1"),
        }

    existing = [
        automation(
            "relay rule",
            [{"platform": "time", "at": "22:00:00"}],
            [{"service": "switch.turn_off", "entity_id": "switch.lamp_relay"}],
        )
    ]
    graph = DeviceGraph(Resolver())
    assert graph.share_device("light.lamp", "switch.lamp_relay")
    conflicts = check_candidate(
        candidate("light.turn_on", "light.lamp", at="22:10:00"), existing, graph
    )
    assert any(c.kind == "shared_device_race" for c in conflicts)


# --- annotation & audit ------------------------------------------------
def test_annotate_and_blocking_flag():
    existing = [
        automation(
            "night off",
            [{"platform": "time", "at": "22:00:00"}],
            [{"service": "light.turn_off", "entity_id": "light.bedroom"}],
        )
    ]
    rules = [
        candidate("light.turn_on", "light.bedroom", at="22:15:00"),
        candidate("light.turn_on", "light.porch", at="03:00:00"),
    ]
    annotate_candidates(rules, existing)
    assert has_blocking_conflict(rules[0]) is True
    assert has_blocking_conflict(rules[1]) is False
    assert rules[0].conflicts[0]["severity"] == "error"


def test_audit_finds_conflicting_existing_automations():
    existing = [
        automation(
            "A",
            [{"platform": "time", "at": "22:00:00"}],
            [{"service": "light.turn_off", "entity_id": "light.bedroom"}],
        ),
        automation(
            "B",
            [{"platform": "time", "at": "22:10:00"}],
            [{"service": "light.turn_on", "entity_id": "light.bedroom"}],
        ),
    ]
    findings = audit_existing(existing)
    assert any(f["kind"] == "value_inconsistency" for f in findings)
    assert findings[0]["severity"] == "error"


def test_audit_finds_redundant_existing_automations():
    existing = [
        automation("A", [{"platform": "time", "at": "22:00:00"}],
                   [{"service": "light.turn_off", "entity_id": "light.bedroom"}]),
        automation("B", [{"platform": "time", "at": "22:05:00"}],
                   [{"service": "light.turn_off", "entity_id": "light.bedroom"}]),
    ]
    findings = audit_existing(existing)
    assert any(f["kind"] == "redundancy" for f in findings)


def test_audit_is_clean_for_unrelated_rules():
    existing = [
        automation("A", [{"platform": "time", "at": "07:00:00"}],
                   [{"service": "light.turn_on", "entity_id": "light.a"}]),
        automation("B", [{"platform": "time", "at": "22:00:00"}],
                   [{"service": "switch.turn_off", "entity_id": "switch.b"}]),
    ]
    assert audit_existing(existing) == []


# --- blind spots --------------------------------------------------------
def _candidate_on(entity_id: str = "light.hallway") -> Candidate:
    return Candidate(
        miner="test",
        title="turn it on",
        triggers=[Trigger(kind="state", entity_id="binary_sensor.motion", to_state="off")],
        actions=[Action(service="light.turn_on", entity_id=entity_id)],
    )


class _Info:
    def __init__(self, entity_id, area_id=None, device_id=None):
        self.entity_id, self.area_id, self.device_id = entity_id, area_id, device_id


class _Resolver:
    def __init__(self, *infos):
        self.entities = {info.entity_id: info for info in infos}


def test_a_legacy_target_inside_data_is_not_invisible():
    """service + data: {entity_id: ...} is still valid, still very common."""
    existing = normalise_automation({
        "alias": "Legacy off",
        "trigger": [{"platform": "state", "entity_id": "binary_sensor.motion", "to": "off"}],
        "action": [{"service": "light.turn_off", "data": {"entity_id": "light.hallway"}}],
    })
    assert existing.targets() == {("light.hallway", "off")}
    conflicts = check_candidate(_candidate_on(), [existing], DeviceGraph())
    assert [c.kind for c in conflicts] == ["value_inconsistency"]


def test_an_area_target_is_expanded_through_the_registry():
    existing = normalise_automation({
        "alias": "Area off",
        "trigger": [{"platform": "state", "entity_id": "binary_sensor.motion", "to": "off"}],
        "action": [{"service": "light.turn_off", "target": {"area_id": "hallway"}}],
    })
    graph = DeviceGraph(_Resolver(_Info("light.hallway", area_id="hallway")))
    conflicts = check_candidate(_candidate_on(), [existing], graph)
    assert any(c.kind == "value_inconsistency" for c in conflicts)


def test_an_unexpandable_target_is_reported_rather_than_ignored():
    """Not knowing what an automation touches is not the same as it touching nothing."""
    existing = normalise_automation({
        "alias": "Area off",
        "trigger": [{"platform": "state", "entity_id": "binary_sensor.motion", "to": "off"}],
        "action": [{"service": "light.turn_off", "target": {"area_id": "hallway"}}],
    })
    conflicts = check_candidate(_candidate_on(), [existing], DeviceGraph())
    assert [c.kind for c in conflicts] == ["unenumerable_target"]


def test_a_clock_rule_and_a_state_rule_can_still_fight_over_one_lamp():
    """Never sharing a trigger is not the same as never colliding."""
    existing = normalise_automation({
        "alias": "Night off",
        "trigger": [{"platform": "time", "at": "22:00:00"}],
        "action": [{"service": "light.turn_off", "target": {"entity_id": "light.hallway"}}],
    })
    conflicts = check_candidate(_candidate_on(), [existing], DeviceGraph())
    assert [c.kind for c in conflicts] == ["value_inconsistency"]


def test_redundancy_still_needs_positive_evidence_of_overlap():
    """The looser test is for value conflicts only; near-duplicate stays strict."""
    existing = normalise_automation({
        "alias": "Night on",
        "trigger": [{"platform": "time", "at": "22:00:00"}],
        "action": [{"service": "light.turn_on", "target": {"entity_id": "light.hallway"}}],
    })
    conflicts = check_candidate(_candidate_on(), [existing], DeviceGraph())
    assert [c.kind for c in conflicts] == []
