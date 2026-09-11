"""When can two rules both apply?

The conflict checks compare triggers and targets.  Two automations that share a
trigger and drive the same entity look identical to them - but an automation is
trigger *and* conditions, and conditions are how people say "this one is for
when I am out, that one for when I am in".  Ignoring them made the audit report
complementary pairs as conflicts, while claiming they applied "under overlapping
conditions" - a claim nothing had checked.

This module answers the narrow, decidable part of that question: are two
condition sets *provably* unable to hold at once?  It never guesses.  "I cannot
prove these are exclusive" is a different answer from "these overlap", and the
callers are careful to say which one they have.

The logic rests on conditions within one automation being ANDed: if any single
condition of A contradicts any single condition of B, then A and B can never
both be satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .util.timeutil import MINUTES_PER_DAY, time_of_day_minutes

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True)
class Fact:
    """One condition, in a shape two rules can be compared in."""

    kind: str  # "state" | "numeric" | "time" | "sun" | "unknown"
    entity_id: str | None = None
    states: frozenset[str] = field(default_factory=frozenset)
    above: float | None = None
    below: float | None = None
    weekdays: frozenset[str] = field(default_factory=frozenset)
    after: int | None = None
    before: int | None = None
    sun: str | None = None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _time_fact(after: Any, before: Any, weekday: Any) -> Fact:
    days = {
        str(d).strip().lower()[:3]
        for d in _as_list(weekday)
        if str(d).strip().lower()[:3] in WEEKDAYS
    }
    return Fact(
        kind="time",
        after=time_of_day_minutes(after) if after else None,
        before=time_of_day_minutes(before) if before else None,
        weekdays=frozenset(days),
    )


def fact_from_raw(condition: Any) -> Fact:
    """One Home Assistant condition dict as a :class:`Fact`."""
    if not isinstance(condition, dict):
        return Fact(kind="unknown")
    kind = str(condition.get("condition") or "").strip().lower()
    entities = [e for e in _as_list(condition.get("entity_id")) if isinstance(e, str)]
    entity_id = entities[0] if len(entities) == 1 else None

    if kind == "state":
        states = {str(s).strip().lower() for s in _as_list(condition.get("state"))}
        if entity_id and states:
            return Fact(kind="state", entity_id=entity_id, states=frozenset(states))
    elif kind == "numeric_state":
        above, below = _number(condition.get("above")), _number(condition.get("below"))
        if entity_id and (above is not None or below is not None):
            return Fact(kind="numeric", entity_id=entity_id, above=above, below=below)
    elif kind == "time":
        return _time_fact(
            condition.get("after"), condition.get("before"), condition.get("weekday")
        )
    elif kind == "sun":
        # "after: sunrise" / "before: sunset" describe the daylight half.
        after, before = condition.get("after"), condition.get("before")
        if after == "sunrise" and not before:
            return Fact(kind="sun", sun="above_horizon")
        if before == "sunrise" and not after:
            return Fact(kind="sun", sun="below_horizon")
        if after == "sunset" and not before:
            return Fact(kind="sun", sun="below_horizon")
        if before == "sunset" and not after:
            return Fact(kind="sun", sun="above_horizon")
    return Fact(kind="unknown")


def fact_from_candidate(condition: Any) -> Fact:
    """One mined :class:`amminer.miners.base.Condition` as a :class:`Fact`."""
    kind = getattr(condition, "kind", "")
    entity_id = getattr(condition, "entity_id", None)
    if kind == "state" and entity_id and getattr(condition, "state", None) is not None:
        return Fact(
            kind="state",
            entity_id=entity_id,
            states=frozenset({str(condition.state).strip().lower()}),
        )
    if kind == "numeric_state" and entity_id:
        above = _number(getattr(condition, "above", None))
        below = _number(getattr(condition, "below", None))
        if above is not None or below is not None:
            return Fact(kind="numeric", entity_id=entity_id, above=above, below=below)
    if kind == "time":
        return _time_fact(
            getattr(condition, "after", None),
            getattr(condition, "before", None),
            getattr(condition, "weekday", None),
        )
    return Fact(kind="unknown")


def facts(conditions: Any) -> list[Fact]:
    """Normalise either shape of condition list into facts."""
    out: list[Fact] = []
    for condition in _as_list(conditions):
        if isinstance(condition, dict):
            out.append(fact_from_raw(condition))
        else:
            out.append(fact_from_candidate(condition))
    return out


def _windows(fact: Fact) -> list[tuple[int, int]]:
    """A time condition as non-wrapping [start, end] minute intervals."""
    after = fact.after if fact.after is not None else 0
    before = fact.before if fact.before is not None else MINUTES_PER_DAY
    if after <= before:
        return [(after, before)]
    # Crosses midnight, so it is two intervals on a 24-hour line.
    return [(after, MINUTES_PER_DAY), (0, before)]


def _windows_disjoint(a: Fact, b: Fact) -> bool:
    if a.after is None and a.before is None:
        return False
    if b.after is None and b.before is None:
        return False
    return not any(
        start_a <= end_b and start_b <= end_a
        for start_a, end_a in _windows(a)
        for start_b, end_b in _windows(b)
    )


def contradict(a: Fact, b: Fact) -> bool:
    """Can these two single conditions provably never hold at the same time?"""
    if a.kind == "state" and b.kind == "state" and a.entity_id == b.entity_id:
        # Two required states for one entity, with nothing in common.
        return not (a.states & b.states)

    if a.kind == "numeric" and b.kind == "numeric" and a.entity_id == b.entity_id:
        # Home Assistant is strict on both sides: value > above and < below.
        if a.below is not None and b.above is not None and a.below <= b.above:
            return True
        if b.below is not None and a.above is not None and b.below <= a.above:
            return True
        return False

    if a.kind == "sun" and b.kind == "sun":
        return bool(a.sun and b.sun and a.sun != b.sun)

    if a.kind == "time" and b.kind == "time":
        if a.weekdays and b.weekdays and not (a.weekdays & b.weekdays):
            return True
        return _windows_disjoint(a, b)

    return False


def provably_exclusive(first: Any, second: Any) -> bool:
    """True only when the two condition sets can never both be satisfied.

    False means "not proven", which is not the same as "they overlap" - an
    unknown condition kind, a template, or two unrelated entities all land here.
    Callers must not report a False as evidence of overlap.
    """
    left, right = facts(first), facts(second)
    return any(contradict(a, b) for a in left for b in right)


def describe(conditions: Any) -> str:
    """A short human rendering, for a message that has to name them."""
    parts: list[str] = []
    for fact in facts(conditions):
        if fact.kind == "state" and fact.entity_id:
            parts.append(f"{fact.entity_id} is {'/'.join(sorted(fact.states))}")
        elif fact.kind == "numeric" and fact.entity_id:
            bounds = []
            if fact.above is not None:
                bounds.append(f"above {fact.above:g}")
            if fact.below is not None:
                bounds.append(f"below {fact.below:g}")
            parts.append(f"{fact.entity_id} is {' and '.join(bounds)}")
        elif fact.kind == "time":
            if fact.weekdays:
                parts.append("on " + ", ".join(d for d in WEEKDAYS if d in fact.weekdays))
            if fact.after is not None or fact.before is not None:
                parts.append("within a time window")
        elif fact.kind == "sun":
            parts.append("the sun is " + str(fact.sun).replace("_", " "))
    return "; ".join(parts) if parts else "no conditions"
