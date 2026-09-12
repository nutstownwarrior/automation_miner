"""Optional AI feature 8 - consolidating related rules into one named scene.

Six separate cards that all say "at about 22:40" are six separate automations
the user has to accept, name and maintain, when what their history actually
shows is one habit: going to bed.  The miners cannot see that, because each one
mines a single action and has no vocabulary for "these belong together".

A model does have that vocabulary, so it is asked to propose the groupings - and
then every claim it makes is checked by something that is not the model:

* the members must exist; an invented id is dropped,
* they must share a trigger the backtester would still credit each member's
  action against, which is decided by comparing the triggers, not by the model
  saying so,
* the group must not contain two actions that fight over the same entity,
* the consolidated rule is **backtested as a unit**, against the same gate as
  every mined candidate.  It does not inherit its members' scores, because the
  scene fires all of the actions together and that is a different rule from any
  of its parts.

A scene that fails the gate is not surfaced.  A scene that passes is added
alongside its members and never replaces them: the model is not permitted to
make a suggestion the user was already shown disappear.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..backtest import backtest
from ..miners.base import Candidate, Trigger
from ..util.text import clean_model_text
from ..util.timeutil import time_of_day_minutes
from .provider import BaseProvider, LLMError

_LOGGER = logging.getLogger(__name__)

#: One action is not a scene, and a "scene" of twenty is a house-wide reset
#: nobody asked for.
MIN_MEMBERS = 2
MAX_MEMBERS = 8

#: A handful of named scenes is a proposal; a dozen is a reorganisation.
MAX_SCENES = 5

SYSTEM_PROMPT = """\
You group home-automation suggestions that are really ONE routine.

Each input is a suggestion that was found in a person's real usage history.
Several of them often describe the same moment - going to bed, leaving for work,
settling down to watch something - split into one card per device.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- Only use the "id" values given. Never invent one.
- Only group suggestions that happen at the SAME MOMENT for the SAME reason.
  Two things that both happen in the evening are not one scene.
- Each group needs at least 2 and at most 8 members. Use an id at most once.
- At most 5 groups. Returning none is a perfectly good answer.
- "name" is what a person would call the routine: "Bedtime", "Leaving the
  house", "Movie night". Two or three words. No device names.
- Do not group things that undo each other.

Shape:
{"scenes": [{"name": "Bedtime", "members": ["abc123", "def456"],
             "reason": "All of these happen as the house is shut down for the night."}]}
"""


@dataclass
class Scene:
    """A proposed grouping that survived every check."""

    name: str
    reason: str
    members: list[str] = field(default_factory=list)
    candidate: Candidate | None = None
    accepted: bool = False
    backtest: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reason": self.reason,
            "members": list(self.members),
            "accepted": self.accepted,
            "backtest": self.backtest,
        }


@dataclass
class SceneResult:
    """What was proposed, what survived, and why the rest did not."""

    scenes: list[Scene] = field(default_factory=list)
    proposed: int = 0
    rejected_unknown_members: int = 0
    rejected_incompatible: int = 0
    rejected_contradictory: int = 0
    rejected_by_backtest: int = 0
    rejected_duplicate: int = 0
    error: str | None = None

    @property
    def accepted(self) -> list[Candidate]:
        return [s.candidate for s in self.scenes if s.accepted and s.candidate is not None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposed": self.proposed,
            "accepted": len(self.accepted),
            "rejected_unknown_members": self.rejected_unknown_members,
            "rejected_incompatible": self.rejected_incompatible,
            "rejected_contradictory": self.rejected_contradictory,
            "rejected_by_backtest": self.rejected_by_backtest,
            "rejected_duplicate": self.rejected_duplicate,
            "scenes": [s.as_dict() for s in self.scenes],
            "error": self.error,
        }


def _trigger_seconds(trigger: Trigger) -> float | None:
    minutes = time_of_day_minutes(trigger.at)
    if minutes is None:
        return None
    seconds = 0
    parts = str(trigger.at).strip().split(":")
    if len(parts) == 3:
        try:
            seconds = int(parts[2])
        except ValueError:
            seconds = 0
    return minutes * 60.0 + seconds


def shared_trigger(
    candidates: Sequence[Candidate], tolerance_seconds: float
) -> Trigger | None:
    """The one trigger that can stand in for every candidate's own, or ``None``.

    This is the check that makes a consolidation honest.  *tolerance_seconds* is
    the backtester's own match tolerance on purpose: a group may only be merged
    onto a single trigger if the backtester would still credit each member's
    real action against that trigger.  Anything looser would be a scene whose
    parts are measured against a moment they never happened at.
    """
    firsts = [c.triggers[0] for c in candidates if c.triggers]
    if len(firsts) != len(candidates) or not firsts:
        return None
    kinds = {t.kind for t in firsts}
    if len(kinds) != 1:
        return None
    kind = kinds.pop()

    if kind == "time":
        seconds = [_trigger_seconds(t) for t in firsts]
        if any(s is None for s in seconds):
            return None
        values = [float(s) for s in seconds]  # type: ignore[arg-type]
        anchor = _tight_anchor(values, tolerance_seconds)
        if anchor is None:
            return None
        return Trigger(kind="time", at=_format_seconds(anchor))

    if kind == "state":
        entities = {t.entity_id for t in firsts}
        states = {t.to_state for t in firsts}
        if len(entities) != 1 or len(states) != 1:
            return None
        holds = {t.for_seconds for t in firsts}
        return Trigger(
            kind="state",
            entity_id=entities.pop(),
            to_state=states.pop(),
            # The longest hold any member required, so the scene never fires
            # sooner than the strictest member would have.
            for_seconds=max((h for h in holds if h), default=None),
        )

    # Sun, numeric_state and time_pattern have no usable single stand-in.  Sun
    # is the interesting one: the members really do share a moment, but
    # :mod:`amminer.backtest` cannot simulate a sun trigger, so every sun
    # grouping was built, sent through the gate and rejected there with an
    # internal-sounding reason.  Refusing it here costs nothing and reports it
    # as what it is - a grouping this feature cannot measure yet.
    return None


DAY = 86400.0


def _tight_anchor(values: list[float], tolerance_seconds: float) -> float | None:
    """One time-of-day that stands in for all of *values*, or ``None``.

    Distances are measured around the clock, not along a number line.  22:58 and
    00:02 are four minutes apart; subtracting seconds-since-midnight makes them
    look like nearly a full day, so every routine that straddles midnight - the
    exact "going to bed" case this feature exists for - was refused.
    """
    if not values:
        return None
    best: tuple[float, float] | None = None
    for anchor in values:
        offsets = [((value - anchor + DAY / 2) % DAY) - DAY / 2 for value in values]
        span = max(offsets) - min(offsets)
        if span > tolerance_seconds:
            continue
        # Rank by span so the tightest framing wins, and break ties on the
        # anchor itself so the result does not depend on member order.
        if best is None or (span, anchor) < best:
            best = (span, anchor)
    if best is None:
        return None
    anchor = best[1]
    offsets = sorted(((value - anchor + DAY / 2) % DAY) - DAY / 2 for value in values)
    # The median, not the mean: one outlier inside the tolerance should not drag
    # the shared moment away from where most of them actually happen.
    return (anchor + offsets[len(offsets) // 2]) % DAY


def _format_seconds(total: float) -> str:
    total = int(total) % 86400
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


#: Services whose effect depends on what the entity was already doing, so two
#: members naming one of these on the same entity can always undo each other.
_REVERSIBLE = ("toggle",)


def contradictory_actions(candidates: Sequence[Candidate]) -> bool:
    """True if two members would fight over the same entity.

    Opposite target states are the obvious case and the only one that used to be
    checked, which missed the commoner one: two members calling the *same*
    service on the same entity with *different* data.  ``light.turn_on`` at
    brightness 30 and at brightness 255 map to the same target state, survive
    the action de-duplication because their payloads differ, and end up in one
    scene firing both back to back in whatever order the model happened to list
    them.  Services with no binary state at all - ``climate.set_temperature``,
    ``fan.set_percentage`` - were invisible here for the same reason.
    """
    wanted: dict[str, set[str]] = {}
    payloads: dict[tuple[str, str], set[str]] = {}
    toggled: set[str] = set()
    touched: dict[str, int] = {}
    for candidate in candidates:
        for action in candidate.actions:
            if not action.entity_id:
                continue
            touched[action.entity_id] = touched.get(action.entity_id, 0) + 1
            if action.service.split(".", 1)[-1] in _REVERSIBLE:
                toggled.add(action.entity_id)
            state = action.target_state
            if state is not None:
                wanted.setdefault(action.entity_id, set()).add(state)
            key = (action.entity_id, action.service)
            payloads.setdefault(key, set()).add(
                json.dumps(action.data or {}, sort_keys=True, default=str)
            )
    if any(len(states) > 1 for states in wanted.values()):
        return True
    if any(len(bodies) > 1 for bodies in payloads.values()):
        return True
    # A toggle alongside anything else on the same entity can undo it.
    return any(touched.get(entity_id, 0) > 1 for entity_id in toggled)


def _condition_key(condition) -> str:
    """What makes two conditions the same condition.

    Compared on meaning, not on the serialised dict.  ``source`` is free text
    recording which miner produced the condition, and ``weekday`` is built in
    whatever order it was read in - so two members carrying the genuinely same
    constraint could compare unequal, and the shared constraint would be dropped
    from the scene, which then fires more widely than any member's own evidence
    supports.
    """
    return json.dumps(
        {
            "kind": condition.kind,
            "entity_id": condition.entity_id,
            "state": condition.state,
            "above": condition.above,
            "below": condition.below,
            "weekday": sorted(condition.weekday or []),
            "after": condition.after,
            "before": condition.before,
        },
        sort_keys=True,
        default=str,
    )


def _consolidate(name: str, reason: str, members: Sequence[Candidate],
                 trigger: Trigger) -> Candidate:
    """Build the single candidate that stands for the whole group.

    Only conditions *every* member carries are kept.  A condition one member
    alone required would gate the other members' actions on something their own
    evidence never involved; leaving it out instead means the scene may fire
    where that member would not have, which the backtest measures and charges to
    the scene.
    """
    shared_conditions = []
    first, *rest = members
    for condition in first.conditions:
        key = _condition_key(condition)
        if all(any(_condition_key(c) == key for c in other.conditions) for other in rest):
            shared_conditions.append(condition)

    actions: list[Any] = []
    seen: set[str] = set()
    for member in members:
        for action in member.actions:
            key = json.dumps(action.as_dict(), sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            actions.append(action)

    scene = Candidate(
        miner="scene",
        title=name,
        triggers=[trigger],
        conditions=shared_conditions,
        actions=actions,
        # Deliberately no score and no evidence copied from the members: the
        # scene is a different rule from any of its parts and is measured from
        # scratch.
        description=reason,
    )
    scene.extra["scene"] = {
        "name": name,
        "reason": reason,
        "members": [m.id for m in members],
        "member_titles": [m.title for m in members],
    }
    return scene


def propose_and_verify(
    candidates: Sequence[Candidate],
    changes: Sequence[Any],
    store,
    options,
    window: tuple[float, float],
    provider: BaseProvider,
    resolver=None,
    overrides: Sequence[Any] = (),
) -> SceneResult:
    """Ask for groupings, then measure each consolidated rule.  Never raises."""
    result = SceneResult()
    groupable = [c for c in candidates if c.actions and c.triggers]
    if len(groupable) < MIN_MEMBERS:
        return result
    if not provider.enabled:
        result.error = "no LLM provider configured"
        return result

    by_id = {c.id: c for c in groupable}
    prompt = json.dumps(
        {
            "suggestions": [
                {
                    "id": c.id,
                    "title": c.title,
                    "when": c.triggers[0].describe(resolver),
                    "does": [a.describe(resolver) for a in c.actions],
                }
                for c in groupable
            ]
        },
        indent=2,
        default=str,
    )
    try:
        raw = provider.complete_json(SYSTEM_PROMPT, prompt)
    except LLMError as err:
        result.error = str(err)
        _LOGGER.warning("Scene proposal failed: %s", err)
        return result

    proposals = raw.get("scenes")
    if not isinstance(proposals, list):
        return result

    tolerance = float(options.backtest_match_tolerance_seconds)
    used: set[str] = set()
    built: set[str] = set()
    for proposal in proposals[:MAX_SCENES]:
        if not isinstance(proposal, dict):
            continue
        result.proposed += 1
        name = clean_model_text(proposal.get("name"), limit=60)
        raw_members = proposal.get("members")
        if not name or not isinstance(raw_members, list):
            result.rejected_unknown_members += 1
            continue
        member_ids: list[str] = []
        for member_id in raw_members:
            # An id used twice, or already spent on another scene, would let one
            # action be counted into two rules that both fire.
            if isinstance(member_id, str) and member_id in by_id and member_id not in used:
                if member_id not in member_ids:
                    member_ids.append(member_id)
        if len(member_ids) != len([m for m in raw_members if isinstance(m, str)]):
            result.rejected_unknown_members += 1
        if not MIN_MEMBERS <= len(member_ids) <= MAX_MEMBERS:
            continue

        members = [by_id[m] for m in member_ids]
        trigger = shared_trigger(members, tolerance)
        if trigger is None:
            result.rejected_incompatible += 1
            continue
        if contradictory_actions(members):
            result.rejected_contradictory += 1
            continue

        scene = Scene(
            name=name,
            reason=clean_model_text(proposal.get("reason"), fallback="no reason given"),
            members=member_ids,
        )
        consolidated = _consolidate(scene.name, scene.reason, members, trigger)
        outcome = backtest(
            consolidated, changes, store, options, window, overrides, validate_holdout=True
        )
        consolidated.backtest = outcome.as_dict()
        # The same formula :func:`amminer.backtest.backtest_all` applies, from a
        # mined score of zero.  A scene therefore ranks below the parts it was
        # built from unless its own measured precision earns otherwise, which is
        # the only ordering a proposal with no evidence of its own deserves.
        if outcome.precision is not None:
            consolidated.score = round(outcome.precision / 2.0, 4)
        scene.backtest = consolidated.backtest
        scene.accepted = outcome.passed
        scene.candidate = consolidated
        if consolidated.id in built:
            # Two groupings can consolidate to the same rule - the action list
            # is de-duplicated, so a group and a superset of it whose extra
            # member adds nothing new produce identical triggers, conditions and
            # actions, and `Candidate.id` hashes exactly those.  Both would be
            # persisted under one id, and the second would silently overwrite
            # the first's name and members.
            result.rejected_duplicate += 1
            continue
        result.scenes.append(scene)
        if not outcome.passed:
            result.rejected_by_backtest += 1
            continue
        built.add(consolidated.id)
        used.update(member_ids)
    return result


def apply_scenes(candidates: Sequence[Candidate], result: SceneResult) -> Sequence[Candidate]:
    """Point each member at the scene it belongs to.  Removes nothing."""
    for scene in result.scenes:
        if not scene.accepted or scene.candidate is None:
            continue
        belongs = set(scene.members)
        for candidate in candidates:
            if candidate.id in belongs:
                candidate.extra["part_of_scene"] = {
                    "id": scene.candidate.id,
                    "name": scene.name,
                }
    return candidates
