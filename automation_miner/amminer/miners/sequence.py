"""Miner D - sequential pattern mining (PrefixSpan-style) over action sequences.

Sequences are built two ways and merged:

1. **Context sequences** - every state change sharing a root context id, in
   order.  These are exactly the multi-step routines Home Assistant itself
   produced or a user produced in one interaction.
2. **Session sequences** - human actions grouped into activity sessions
   separated by more than ``session_gap_seconds`` of silence.  This is what
   catches "arrive home -> lights on -> thermostat up", where each step has its
   own context.

We then run a small PrefixSpan: grow frequent prefixes one item at a time,
keeping only those meeting ``sequence_min_occurrences``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

from ..config import ACTIONABLE_DOMAINS, Options
from ..recorderdb.models import Cause, StateChange
from .base import Action, Candidate, Evidence, Trigger
from .time_of_day import service_for

_LOGGER = logging.getLogger(__name__)

DEFAULT_SESSION_GAP = 300.0
MAX_PATTERN_LENGTH = 4


def _item(change: StateChange) -> str:
    return f"{change.entity_id}={change.state.lower()}"


def build_sequences(
    changes: Sequence[StateChange],
    options: Options,
    session_gap: float = DEFAULT_SESSION_GAP,
) -> list[list[str]]:
    """Build ordered, de-duplicated item sequences."""
    usable = [
        change
        for change in changes
        if change.is_transition
        and not options.is_excluded(change.entity_id)
        and change.state.lower() not in ("unknown", "unavailable", "")
        and change.cause in (Cause.HUMAN, Cause.DEVICE, Cause.AUTOMATION, Cause.SCRIPT)
        and change.numeric is None
    ]
    usable.sort(key=lambda c: c.ts)

    sequences: list[list[str]] = []

    # 1. context-rooted sequences
    by_context: dict[str, list[StateChange]] = defaultdict(list)
    for change in usable:
        if change.context_id and (change.cause.is_automated or change.chain_depth > 0):
            by_context[change.context_id].append(change)
    for rows in by_context.values():
        if len(rows) >= 2:
            sequences.append(_dedupe([_item(r) for r in rows]))

    # 2. activity sessions (human-anchored)
    session: list[StateChange] = []
    last_ts: float | None = None
    for change in usable:
        if last_ts is not None and change.ts - last_ts > session_gap:
            if len(session) >= 2 and any(c.cause is Cause.HUMAN for c in session):
                sequences.append(_dedupe([_item(r) for r in session]))
            session = []
        session.append(change)
        last_ts = change.ts
    if len(session) >= 2 and any(c.cause is Cause.HUMAN for c in session):
        sequences.append(_dedupe([_item(r) for r in session]))

    return [s for s in sequences if len(s) >= 2]


def _dedupe(items: list[str]) -> list[str]:
    """Keep first occurrence order, drop immediate and later repeats."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _contains_subsequence(sequence: Sequence[str], pattern: Sequence[str]) -> bool:
    it = iter(sequence)
    return all(item in it for item in pattern)


def prefixspan(
    sequences: Sequence[Sequence[str]],
    min_support: int,
    max_length: int = MAX_PATTERN_LENGTH,
) -> list[tuple[tuple[str, ...], int]]:
    """Minimal PrefixSpan: frequent ordered subsequences with their support."""
    if not sequences or min_support < 1:
        return []

    results: list[tuple[tuple[str, ...], int]] = []

    def grow(prefix: tuple[str, ...], projected: list[Sequence[str]]) -> None:
        if len(prefix) >= max_length:
            return
        counts: dict[str, int] = defaultdict(int)
        for suffix in projected:
            for item in set(suffix):
                counts[item] += 1
        for item, support in sorted(counts.items(), key=lambda kv: -kv[1]):
            if support < min_support:
                continue
            new_prefix = prefix + (item,)
            new_projected: list[Sequence[str]] = []
            for suffix in projected:
                if item in suffix:
                    new_projected.append(suffix[list(suffix).index(item) + 1 :])
            if len(new_prefix) >= 2:
                results.append((new_prefix, support))
            grow(new_prefix, new_projected)

    grow((), [list(s) for s in sequences])
    return results


def mine(
    changes: Sequence[StateChange],
    options: Options,
    window: tuple[float, float] | None = None,
    resolver=None,
) -> list[Candidate]:
    """Mine multi-step routines."""
    sequences = build_sequences(changes, options)
    if len(sequences) < options.sequence_min_occurrences:
        _LOGGER.info("sequence miner: only %d sequences, skipping", len(sequences))
        return []

    patterns = prefixspan(sequences, options.sequence_min_occurrences)
    if not patterns:
        _LOGGER.info("sequence miner produced 0 patterns from %d sequences", len(sequences))
        return []

    # Prefer longer patterns, and drop any pattern fully contained in a longer
    # one with (nearly) the same support - it adds no information.
    patterns.sort(key=lambda kv: (-len(kv[0]), -kv[1]))
    kept: list[tuple[tuple[str, ...], int]] = []
    for pattern, support in patterns:
        redundant = any(
            len(pattern) < len(other)
            and _contains_subsequence(other, pattern)
            and support <= other_support * 1.1
            for other, other_support in kept
        )
        if not redundant:
            kept.append((pattern, support))

    human_items: dict[str, int] = defaultdict(int)
    for change in changes:
        if change.cause is Cause.HUMAN and change.is_transition:
            human_items[_item(change)] += 1

    start_ts = window[0] if window else min((c.ts for c in changes), default=0.0)
    end_ts = window[1] if window else max((c.ts for c in changes), default=0.0)

    candidates: list[Candidate] = []
    for pattern, support in kept:
        trigger_item = pattern[0]
        step_items = list(pattern[1:])
        actions: list[Action] = []
        for item in step_items:
            entity_id, _, state = item.partition("=")
            if entity_id.split(".", 1)[0] not in ACTIONABLE_DOMAINS:
                continue
            if human_items.get(item, 0) < 2:
                continue
            service = service_for(entity_id, state)
            if service is None:
                continue
            actions.append(Action(service=service[0], entity_id=entity_id, data=dict(service[1])))
        if len(actions) < 2:
            # A single-step "routine" is really an association rule; miner C
            # already covers those with proper support/confidence/lift.
            continue

        t_entity, _, t_state = trigger_item.partition("=")
        confidence = support / max(len(sequences), 1)
        t_name = resolver.name_of(t_entity) if resolver else t_entity
        action_names = [a.describe(resolver) for a in actions]

        candidates.append(
            Candidate(
                miner="sequence",
                title=f"Routine: {t_name} '{t_state}' then {len(actions)} steps",
                description=(
                    f"After {t_name} becomes '{t_state}' you usually: "
                    + ", then ".join(action_names)
                ),
                triggers=[Trigger(kind="state", entity_id=t_entity, to_state=t_state)],
                actions=actions,
                evidence=Evidence(
                    occurrences=support,
                    opportunities=len(sequences),
                    confidence=confidence,
                    support=confidence,
                    window_start_ts=start_ts,
                    window_end_ts=end_ts,
                    window_days=(end_ts - start_ts) / 86400.0,
                    notes=[
                        f"This exact ordered sequence appeared {support} times in "
                        f"{len(sequences)} activity sessions.",
                        "Sequences were reconstructed from context chains and activity sessions.",
                    ],
                    extra={"pattern": list(pattern)},
                ),
                score=round(min(confidence * 2.0, 1.0) * min(support / 10.0, 1.0), 4),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    _LOGGER.info("sequence miner produced %d candidates", len(candidates))
    return candidates
