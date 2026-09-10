"""Reconstruct *why* a state changed, and detect user overrides.

Home Assistant exposes no native "root cause" for a state change.  It does
record a context on every row:

``context_user_id``
    set when a logged-in user initiated the change (UI, app, voice with a user).
``context_parent_id``
    set when the change is a *consequence* of another context - the chain that
    leads back to an ``automation_triggered`` / ``script_started`` event.

So we build the causal graph ourselves: index every context by id, link children
to parents, then walk each chain to its root.  The root is either a human, an
automation/script, or nothing at all (a device reporting on its own).

An **override** is the money signal: a human contradicting what an automation
just did to the same entity, within ``override_window_seconds``.  It is negative
evidence against that automation and positive evidence for the behaviour the
human actually wanted.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .models import Cause, OverrideEvent, RecorderEvent, StateChange

_LOGGER = logging.getLogger(__name__)

MAX_CHAIN_DEPTH = 12


@dataclass
class ContextOrigin:
    """What sits at the root of one context chain."""

    cause: Cause
    entity_id: str | None = None
    user_id: str | None = None
    depth: int = 0


@dataclass
class CausalityIndex:
    """Context id -> origin, built from events and state rows."""

    origins: dict[str, ContextOrigin] = field(default_factory=dict)
    parents: dict[str, str] = field(default_factory=dict)
    users: dict[str, str] = field(default_factory=dict)
    #: True when the recorder actually carries context user ids for this window.
    has_user_context: bool = False

    def add_event(self, event: RecorderEvent) -> None:
        if event.context_id:
            if event.context_parent_id and event.context_parent_id != event.context_id:
                self.parents.setdefault(event.context_id, event.context_parent_id)
            if event.context_user_id:
                self.users.setdefault(event.context_id, event.context_user_id)
                self.has_user_context = True
            if event.event_type in ("automation_triggered", "script_started"):
                cause = (
                    Cause.AUTOMATION
                    if event.event_type == "automation_triggered"
                    else Cause.SCRIPT
                )
                entity_id = event.entity_id or event.data.get("entity_id")
                existing = self.origins.get(event.context_id)
                # An automation that calls a script shares its context with the
                # script it started.  The automation is the root cause; which of
                # the two events the recorder hands us first is not something to
                # depend on, so state the precedence rather than assume an order.
                if existing is None or (
                    cause is Cause.AUTOMATION and existing.cause is Cause.SCRIPT
                ):
                    self.origins[event.context_id] = ContextOrigin(cause, entity_id=entity_id)

    def add_state(self, change: StateChange) -> None:
        if not change.context_id:
            return
        if change.context_parent_id and change.context_parent_id != change.context_id:
            self.parents.setdefault(change.context_id, change.context_parent_id)
        if change.context_user_id:
            self.users.setdefault(change.context_id, change.context_user_id)
            self.has_user_context = True
        # An automation entity turning "on" is itself the origin marker when the
        # events table has been purged but the automation's own state row lives.
        if change.domain == "automation" and change.context_id not in self.origins:
            if change.state == "on" or change.attributes.get("last_triggered"):
                self.origins.setdefault(
                    change.context_id, ContextOrigin(Cause.AUTOMATION, entity_id=change.entity_id)
                )

    # ------------------------------------------------------------------
    def resolve(self, context_id: str | None, user_id: str | None = None) -> ContextOrigin:
        """Walk the parent chain to the root cause of *context_id*.

        A user id does *not* win over an automation or script origin on the same
        context.  Home Assistant reuses one context for a whole automation or
        script run and carries the starting user's id down it, so a light turned
        off by a script that someone pressed "run" on arrives carrying that
        person's user id.  Reading that as "a human turned this light off" makes
        every step of every hand-started script look like manual behaviour worth
        automating - behaviour that is, by definition, already automated.  What
        the *context* says happened comes first; the user id answers a different
        question, which is who set it going.
        """
        if not context_id:
            # No context to reason about.  A user id alone still identifies a
            # person, and its absence is genuinely unknown, not a device.
            if user_id:
                return ContextOrigin(Cause.HUMAN, user_id=user_id)
            return ContextOrigin(Cause.UNKNOWN)

        seen: set[str] = set()
        current = context_id
        depth = 0
        while current and current not in seen and depth <= MAX_CHAIN_DEPTH:
            seen.add(current)
            origin = self.origins.get(current)
            if origin is not None:
                return ContextOrigin(origin.cause, entity_id=origin.entity_id, depth=depth)
            direct_user = self.users.get(current)
            if direct_user:
                return ContextOrigin(Cause.HUMAN, user_id=direct_user, depth=depth)
            current = self.parents.get(current)
            depth += 1
        if user_id:
            return ContextOrigin(Cause.HUMAN, user_id=user_id, depth=depth)
        # A context that named nobody and no automation: something in the house
        # acted on its own.
        return ContextOrigin(Cause.DEVICE, depth=depth)


def build_index(
    changes: Iterable[StateChange], events: Iterable[RecorderEvent] = ()
) -> CausalityIndex:
    """Build the causality index from a window of rows."""
    index = CausalityIndex()
    for event in events:
        index.add_event(event)
    for change in changes:
        index.add_state(change)
    return index


def classify(
    changes: Sequence[StateChange],
    index: CausalityIndex,
    excluded_users: Iterable[str] = (),
) -> Sequence[StateChange]:
    """Annotate every change in place with its cause and origin entity."""
    blocked = {u for u in excluded_users if u}
    for change in changes:
        origin = index.resolve(change.context_id, change.context_user_id)
        if origin.cause is Cause.HUMAN and origin.user_id in blocked:
            # Service accounts (e.g. a long-lived token used by another add-on)
            # look like humans but are not.
            change.cause = Cause.AUTOMATION
            change.origin_entity_id = None
        elif origin.cause is Cause.UNKNOWN:
            # No context at all.  That is not the same as "a device did it" -
            # it is a row from before Home Assistant recorded contexts, or one
            # restored after a restart.  Recording the guess as a fact would
            # put it into device-driven pattern counts as though we knew.
            change.cause = Cause.UNKNOWN
            change.origin_entity_id = None
        else:
            change.cause = origin.cause
            change.origin_entity_id = origin.entity_id
        change.chain_depth = origin.depth
    return changes


def annotate(
    changes: Sequence[StateChange],
    events: Iterable[RecorderEvent] = (),
    excluded_users: Iterable[str] = (),
) -> tuple[Sequence[StateChange], CausalityIndex]:
    """Convenience: build the index and classify in one pass."""
    index = build_index(changes, events)
    classify(changes, index, excluded_users)
    return changes, index


# ----------------------------------------------------------------------
def detect_overrides(
    changes: Sequence[StateChange],
    window_seconds: float = 120.0,
    include_device_causes: bool = False,
) -> list[OverrideEvent]:
    """Find human corrections of automation actions.

    For each entity we scan chronologically.  When an automation-caused change
    sets entity X to state ``S`` and a human sets X to something *different*
    within ``window_seconds``, that is an override.  A human confirming the same
    state is not an override.
    """
    by_entity: dict[str, list[StateChange]] = defaultdict(list)
    for change in changes:
        by_entity[change.entity_id].append(change)

    overrides: list[OverrideEvent] = []
    for entity_id, rows in by_entity.items():
        rows = sorted(rows, key=lambda c: c.ts)
        pending: StateChange | None = None
        for change in rows:
            if change.cause.is_automated:
                pending = change
                continue
            is_human = change.cause is Cause.HUMAN or (
                include_device_causes and change.cause is Cause.DEVICE
            )
            if not is_human or pending is None:
                continue
            delay = change.ts - pending.ts
            if delay < 0 or delay > window_seconds:
                pending = None
                continue
            if change.state == pending.state:
                # The human agreed with the automation - not an override.
                continue
            overrides.append(
                OverrideEvent(
                    entity_id=entity_id,
                    ts=change.ts,
                    automation_entity_id=pending.origin_entity_id,
                    automation_state=pending.state,
                    human_state=change.state,
                    delay_seconds=delay,
                    context_id=change.context_id,
                )
            )
            pending = None
    overrides.sort(key=lambda o: o.ts)
    return overrides


def override_summary(overrides: Sequence[OverrideEvent]) -> dict[str, dict[str, object]]:
    """Aggregate overrides per automation, for the UI and for negative feedback."""
    summary: dict[str, dict[str, object]] = {}
    for override in overrides:
        key = override.automation_entity_id or "unknown"
        entry = summary.setdefault(
            key, {"count": 0, "entities": set(), "preferred_states": defaultdict(int)}
        )
        entry["count"] = int(entry["count"]) + 1  # type: ignore[arg-type]
        entry["entities"].add(override.entity_id)  # type: ignore[union-attr]
        entry["preferred_states"][override.human_state] += 1  # type: ignore[index]
    for entry in summary.values():
        entry["entities"] = sorted(entry["entities"])  # type: ignore[arg-type]
        entry["preferred_states"] = dict(entry["preferred_states"])  # type: ignore[arg-type]
    return summary


def causality_stats(changes: Sequence[StateChange]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for change in changes:
        counts[change.cause.value] += 1
    return dict(counts)
