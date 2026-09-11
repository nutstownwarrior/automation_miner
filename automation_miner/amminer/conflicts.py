"""Conflict, loop and redundancy checks against existing automations.

Follows the trigger-action-programming conflict literature (SHACR, AutoIoT and
the TAP conflict-detection surveys).  Four classes are detected:

``value_inconsistency``
    two rules drive the same entity to different states under conditions that
    can overlap,
``loop``
    the candidate's action triggers an existing rule whose action re-triggers
    the candidate (directly, or through a short chain),
``redundancy``
    a near-duplicate of an automation the user already has,
``shared_device_race``
    two rules act on the *same physical device* at overlapping times, even
    though the entity ids differ.

A conflicting candidate is never auto-enabled; the finding is shown in the UI so
the user decides.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .automations import ExistingAutomation
from .conditions import provably_exclusive
from .miners.base import Candidate
from .util.timeutil import MINUTES_PER_DAY, circular_distance

_LOGGER = logging.getLogger(__name__)

#: Two time triggers this close together are treated as overlapping.
TIME_OVERLAP_MINUTES = 45

SEVERITY_ORDER = {"error": 3, "warning": 2, "info": 1}


@dataclass
class Conflict:
    kind: str
    severity: str
    message: str
    other: str | None = None
    other_entity_id: str | None = None
    entities: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "other": self.other,
            "other_entity_id": self.other_entity_id,
            "entities": self.entities,
        }


# ----------------------------------------------------------------------
class DeviceGraph:
    """Maps entities to the physical device and area they belong to.

    Two entities on the same device (a bulb's ``light.`` and its ``switch.``,
    say) contend for the same hardware even though their ids differ - that is
    what makes a "shared device race" detectable at all.
    """

    def __init__(self, resolver=None) -> None:
        self.entity_to_device: dict[str, str] = {}
        self.entity_to_area: dict[str, str] = {}
        self.device_to_entities: dict[str, list[str]] = defaultdict(list)
        self.area_to_entities: dict[str, list[str]] = defaultdict(list)
        self.known = resolver is not None
        if resolver is not None:
            for info in resolver.entities.values():
                if info.device_id:
                    self.entity_to_device[info.entity_id] = info.device_id
                    self.device_to_entities[info.device_id].append(info.entity_id)
                if info.area_id:
                    self.entity_to_area[info.entity_id] = info.area_id
                    self.area_to_entities[info.area_id].append(info.entity_id)

    def device_of(self, entity_id: str) -> str | None:
        return self.entity_to_device.get(entity_id)

    def siblings(self, entity_id: str) -> list[str]:
        device_id = self.device_of(entity_id)
        if not device_id:
            return []
        return [e for e in self.device_to_entities.get(device_id, []) if e != entity_id]

    def share_device(self, a: str, b: str) -> bool:
        device_a, device_b = self.device_of(a), self.device_of(b)
        return bool(device_a and device_a == device_b)

    def expand(self, area_ids: Sequence[str], device_ids: Sequence[str]) -> tuple[set[str], bool]:
        """Entities named by area/device targets, and whether any could not be.

        "Turn off the kitchen" and "turn off this device" name real entities;
        they just do not spell them.  Reading only the ``entity_id`` field makes
        such an automation look like it touches nothing, so a candidate that
        fights with it is reported as conflict-free.  When the registry cannot
        say what a target covers, that is returned too - an unenumerable target
        must not be silently read as an empty one.
        """
        found: set[str] = set()
        unresolved = False
        for area_id in area_ids:
            entities = self.area_to_entities.get(area_id)
            if entities:
                found.update(entities)
            else:
                unresolved = True
        for device_id in device_ids:
            entities = self.device_to_entities.get(device_id)
            if entities:
                found.update(entities)
            else:
                unresolved = True
        return found, unresolved


def existing_targets(
    automation: ExistingAutomation, graph: DeviceGraph
) -> tuple[set[tuple[str, str | None]], bool]:
    """``(entity, wanted state)`` pairs an automation drives, area targets included."""
    targets = set(automation.targets())
    unresolved = False
    for action in automation.actions:
        if not action.area_ids and not action.device_ids:
            continue
        entities, missing = graph.expand(action.area_ids, action.device_ids)
        unresolved = unresolved or missing
        targets.update((entity_id, action.target_state) for entity_id in entities)
    return targets, unresolved


# ----------------------------------------------------------------------
def _candidate_times(candidate: Candidate) -> list[int]:
    minutes: list[int] = []
    for trigger in candidate.triggers:
        if trigger.kind == "time" and trigger.at:
            parts = str(trigger.at).split(":")
            try:
                minutes.append(int(parts[0]) * 60 + int(parts[1]))
            except (ValueError, IndexError):
                continue
    return minutes


def _existing_times(automation: ExistingAutomation) -> list[int]:
    minutes: list[int] = []
    for at in automation.trigger_times:
        parts = str(at).split(":")
        try:
            minutes.append(int(parts[0]) * 60 + int(parts[1]))
        except (ValueError, IndexError):
            continue
    return minutes


def _triggers_overlap(
    candidate: Candidate, automation: ExistingAutomation, strict: bool = True
) -> bool:
    """Could both rules plausibly fire in the same situation?

    ``strict`` asks the conservative question - is there positive evidence that
    these two fire together?  That is the right test for calling something a
    near-duplicate, where flagging every unrelated pair would be noise.  It is
    the wrong test for two rules driving one entity to opposite states: "at
    22:00, off" and "when motion stops, on" never share a trigger and never
    share a clock time, and they still fight over the same lamp every evening.
    """
    # A mined candidate carries conditions too, and the same rule applies: two
    # rules whose conditions cannot both hold never fire in the same situation.
    if provably_exclusive(
        candidate.conditions,
        automation.raw.get("condition") or automation.raw.get("conditions"),
    ):
        return False
    shared_entities = set(candidate.trigger_entities) & set(automation.trigger_entities)
    if shared_entities:
        return True
    candidate_minutes = _candidate_times(candidate)
    existing_minutes = _existing_times(automation)
    if candidate_minutes and existing_minutes:
        return any(
            circular_distance(a, b, MINUTES_PER_DAY) <= TIME_OVERLAP_MINUTES
            for a in candidate_minutes
            for b in existing_minutes
        )
    # One rule is time-driven and the other state-driven.  Nothing rules out
    # their firing in the same situation; there is simply no evidence either
    # way.
    return not strict


def check_value_inconsistency(
    candidate: Candidate,
    automations: Sequence[ExistingAutomation],
    graph: DeviceGraph | None = None,
) -> list[Conflict]:
    out: list[Conflict] = []
    graph = graph or DeviceGraph()
    candidate_targets = {
        (action.entity_id, action.target_state)
        for action in candidate.actions
        if action.entity_id
    }
    candidate_entities = {entity_id for entity_id, _ in candidate_targets}
    for automation in automations:
        if not automation.enabled:
            continue
        targets, unresolved = existing_targets(automation, graph)
        if unresolved and candidate_entities:
            out.append(
                Conflict(
                    kind="unenumerable_target",
                    severity="warning",
                    message=(
                        f"'{automation.alias}' acts on a whole area or device that this "
                        "instance's registry could not expand, so it may well touch "
                        f"{', '.join(sorted(candidate_entities))} too - the conflict check "
                        "cannot rule it out."
                    ),
                    other=automation.alias,
                    other_entity_id=automation.entity_id,
                    entities=sorted(candidate_entities),
                )
            )
        for entity_id, wanted in candidate_targets:
            for other_entity, other_state in targets:
                if other_entity != entity_id:
                    continue
                if wanted is None or other_state is None or wanted == other_state:
                    continue
                # Two rules pulling one entity opposite ways is worth reporting
                # even without positive evidence that they fire together.
                if not _triggers_overlap(candidate, automation, strict=False):
                    continue
                out.append(
                    Conflict(
                        kind="value_inconsistency",
                        severity="error",
                        message=(
                            f"'{automation.alias}' drives {entity_id} to '{other_state}' under "
                            f"conditions that overlap this rule's '{wanted}'."
                        ),
                        other=automation.alias,
                        other_entity_id=automation.entity_id,
                        entities=[entity_id],
                    )
                )
    return out


def check_redundancy(
    candidate: Candidate,
    automations: Sequence[ExistingAutomation],
    graph: DeviceGraph | None = None,
) -> list[Conflict]:
    out: list[Conflict] = []
    graph = graph or DeviceGraph()
    candidate_targets = {
        (action.entity_id, action.target_state)
        for action in candidate.actions
        if action.entity_id
    }
    for automation in automations:
        overlap = candidate_targets & existing_targets(automation, graph)[0]
        if not overlap:
            continue
        if not _triggers_overlap(candidate, automation):
            continue
        entities = sorted({entity_id for entity_id, _ in overlap if entity_id})
        out.append(
            Conflict(
                kind="redundancy",
                severity="warning",
                message=(
                    f"'{automation.alias}' already does this ({', '.join(entities)}) under "
                    "overlapping conditions - this rule would be a near-duplicate."
                ),
                other=automation.alias,
                other_entity_id=automation.entity_id,
                entities=entities,
            )
        )
    return out


def check_loops(
    candidate: Candidate, automations: Sequence[ExistingAutomation], max_depth: int = 3
) -> list[Conflict]:
    """Does acting on X wake a rule that ends up re-triggering this one?"""
    out: list[Conflict] = []
    candidate_trigger_entities = set(candidate.trigger_entities)
    if not candidate_trigger_entities:
        return out

    # entity -> automations triggered by it
    triggered_by: dict[str, list[ExistingAutomation]] = defaultdict(list)
    for automation in automations:
        if not automation.enabled:
            continue
        for entity_id in automation.trigger_entities:
            triggered_by[entity_id].append(automation)

    frontier: list[tuple[str, list[str]]] = [
        (entity_id, [candidate.title]) for entity_id in candidate.target_entities
    ]
    seen: set[str] = set()
    depth = 0
    while frontier and depth < max_depth:
        next_frontier: list[tuple[str, list[str]]] = []
        for entity_id, path in frontier:
            for automation in triggered_by.get(entity_id, []):
                if automation.alias in path:
                    continue
                new_path = path + [automation.alias]
                for produced in automation.action_entities:
                    if produced in candidate_trigger_entities:
                        out.append(
                            Conflict(
                                kind="loop",
                                severity="error",
                                message=(
                                    "Dependency cycle: this rule changes "
                                    f"{entity_id}, which triggers '{automation.alias}', which "
                                    f"changes {produced} - a trigger of this rule. "
                                    f"Path: {' -> '.join(new_path)}."
                                ),
                                other=automation.alias,
                                other_entity_id=automation.entity_id,
                                entities=[entity_id, produced],
                            )
                        )
                    elif produced not in seen:
                        seen.add(produced)
                        next_frontier.append((produced, new_path))
        frontier = next_frontier
        depth += 1
    return out


def check_shared_device_races(
    candidate: Candidate,
    automations: Sequence[ExistingAutomation],
    graph: DeviceGraph,
) -> list[Conflict]:
    out: list[Conflict] = []
    for action in candidate.actions:
        if not action.entity_id:
            continue
        siblings = set(graph.siblings(action.entity_id))
        if not siblings:
            continue
        for automation in automations:
            if not automation.enabled:
                continue
            contended = siblings & set(automation.action_entities)
            if not contended:
                continue
            if not _triggers_overlap(candidate, automation):
                continue
            out.append(
                Conflict(
                    kind="shared_device_race",
                    severity="warning",
                    message=(
                        f"'{automation.alias}' controls {', '.join(sorted(contended))} on the "
                        f"same physical device as {action.entity_id}, at an overlapping time."
                    ),
                    other=automation.alias,
                    other_entity_id=automation.entity_id,
                    entities=[action.entity_id, *sorted(contended)],
                )
            )
    return out


def check_self_conflicts(candidate: Candidate) -> list[Conflict]:
    """A rule that contradicts itself (two actions on one entity)."""
    out: list[Conflict] = []
    by_entity: dict[str, set[str | None]] = defaultdict(set)
    for action in candidate.actions:
        if action.entity_id:
            by_entity[action.entity_id].add(action.target_state)
    for entity_id, states in by_entity.items():
        concrete = {s for s in states if s is not None}
        if len(concrete) > 1:
            out.append(
                Conflict(
                    kind="value_inconsistency",
                    severity="error",
                    message=(
                        f"This rule sets {entity_id} to both "
                        f"{' and '.join(sorted(concrete))} in one run."
                    ),
                    entities=[entity_id],
                )
            )
    return out


def check_candidate(
    candidate: Candidate,
    automations: Sequence[ExistingAutomation],
    graph: DeviceGraph | None = None,
) -> list[Conflict]:
    """Run every check against one candidate."""
    graph = graph or DeviceGraph()
    conflicts = (
        check_self_conflicts(candidate)
        + check_value_inconsistency(candidate, automations, graph)
        + check_loops(candidate, automations)
        + check_redundancy(candidate, automations, graph)
        + check_shared_device_races(candidate, automations, graph)
    )
    # De-duplicate identical findings and rank the worst first.
    unique: dict[tuple[str, str | None, str], Conflict] = {}
    for conflict in conflicts:
        key = (conflict.kind, conflict.other, conflict.message)
        unique.setdefault(key, conflict)
    return sorted(
        unique.values(), key=lambda c: -SEVERITY_ORDER.get(c.severity, 0)
    )


def annotate_candidates(
    candidates: Sequence[Candidate],
    automations: Sequence[ExistingAutomation],
    resolver=None,
) -> Sequence[Candidate]:
    """Attach conflict findings to every candidate, in place."""
    graph = DeviceGraph(resolver)
    for candidate in candidates:
        candidate.conflicts = [c.as_dict() for c in check_candidate(candidate, automations, graph)]
    blocking = sum(
        1 for c in candidates if any(x["severity"] == "error" for x in c.conflicts)
    )
    _LOGGER.info(
        "conflict check: %d of %d candidates have blocking conflicts", blocking, len(candidates)
    )
    return candidates


def has_blocking_conflict(candidate: Candidate) -> bool:
    return any(conflict.get("severity") == "error" for conflict in candidate.conflicts)


def _condition_signature(conditions: Any) -> str:
    """A stable rendering, so "the same conditions" is decidable."""
    from .conditions import describe

    return describe(conditions)


def audit_existing(
    automations: Sequence[ExistingAutomation], resolver=None
) -> list[dict[str, Any]]:
    """Audit the user's existing automations against each other.

    ``resolver`` is accepted for symmetry with :func:`annotate_candidates` and
    for future device-graph checks between existing rules.
    """
    findings: list[dict[str, Any]] = []
    for index, first in enumerate(automations):
        for second in automations[index + 1 :]:
            if not (first.enabled and second.enabled):
                continue
            shared_triggers = set(first.trigger_entities) & set(second.trigger_entities)
            shared_time = bool(
                _existing_times(first)
                and _existing_times(second)
                and any(
                    circular_distance(a, b, MINUTES_PER_DAY) <= TIME_OVERLAP_MINUTES
                    for a in _existing_times(first)
                    for b in _existing_times(second)
                )
            )
            if not (shared_triggers or shared_time):
                continue
            # Sharing a trigger is not sharing a situation.  Conditions are how
            # people say "this one is for when I am out, that one for when I am
            # in", and two rules whose conditions cannot both hold never collide
            # however much else they have in common.
            first_conditions = first.raw.get("condition") or first.raw.get("conditions")
            second_conditions = second.raw.get("condition") or second.raw.get("conditions")
            if provably_exclusive(first_conditions, second_conditions):
                continue
            differ = _condition_signature(first_conditions) != _condition_signature(
                second_conditions
            )
            qualifier = (
                "their conditions differ, so they may never both apply"
                if differ
                else "under overlapping conditions"
            )
            for entity_a, state_a in first.targets():
                for entity_b, state_b in second.targets():
                    if entity_a != entity_b:
                        continue
                    if state_a is None or state_b is None:
                        continue
                    if state_a != state_b:
                        findings.append(
                            {
                                "kind": "value_inconsistency",
                                # Only an error when they really do coincide;
                                # unproven overlap is a question, not a verdict.
                                "severity": "warning" if differ else "error",
                                "message": (
                                    f"'{first.alias}' sets {entity_a} to '{state_a}' while "
                                    f"'{second.alias}' sets it to '{state_b}' - {qualifier}."
                                ),
                                "automations": [first.alias, second.alias],
                                "entities": [entity_a],
                            }
                        )
                    else:
                        findings.append(
                            {
                                "kind": "redundancy",
                                "severity": "info",
                                "message": (
                                    f"'{first.alias}' and '{second.alias}' both set {entity_a} "
                                    f"to '{state_a}' - {qualifier}."
                                ),
                                "automations": [first.alias, second.alias],
                                "entities": [entity_a],
                            }
                        )
    # Loops between existing automations.
    triggered_by: dict[str, list[ExistingAutomation]] = defaultdict(list)
    for automation in automations:
        for entity_id in automation.trigger_entities:
            triggered_by[entity_id].append(automation)
    for automation in automations:
        for produced in automation.action_entities:
            for other in triggered_by.get(produced, []):
                if other is automation:
                    continue
                if set(other.action_entities) & set(automation.trigger_entities):
                    findings.append(
                        {
                            "kind": "loop",
                            "severity": "error",
                            "message": (
                                f"'{automation.alias}' and '{other.alias}' can trigger each "
                                f"other through {produced}."
                            ),
                            "automations": [automation.alias, other.alias],
                            "entities": [produced],
                        }
                    )
    unique: dict[str, dict[str, Any]] = {}
    for finding in findings:
        unique.setdefault(f"{finding['kind']}|{finding['message']}", finding)
    return sorted(
        unique.values(), key=lambda f: -SEVERITY_ORDER.get(str(f["severity"]), 0)
    )
