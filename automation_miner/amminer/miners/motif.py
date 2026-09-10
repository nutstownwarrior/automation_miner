"""Miner E - matrix-profile motif discovery on numeric sensor series.

``stumpy`` gives us the matrix profile, whose lowest points are *motifs*:
sub-sequences that repeat.  We use it to find the recurring numeric shape that
precedes a human action - "the blinds get closed when illuminance falls through
~600 lx around sunset".

``stumpy`` pulls in ``numba``, which has no musl wheel, so it is imported
lazily.  When it is missing this miner falls back to a deterministic
threshold-crossing search that finds the same *kind* of rule (a level crossing
that reliably precedes the action) without the matrix profile.  Either way the
add-on keeps working; the UI reports which path ran.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Options
from ..enrich.signals import SignalSeries, SignalStore
from ..recorderdb.models import StateChange
from .base import Action, Candidate, Evidence, Trigger
from .time_of_day import human_action_events, service_for

_LOGGER = logging.getLogger(__name__)

#: How long before an action a crossing still counts as its cause.
LEAD_WINDOW_SECONDS = 900.0
MIN_ACTIONS = 5


def stumpy_available() -> bool:
    """Report (without importing numba at startup) whether stumpy can be used."""
    from importlib.util import find_spec

    try:
        return find_spec("stumpy") is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


@dataclass
class Crossing:
    entity_id: str
    threshold: float
    direction: str  # "falling" | "rising"
    hits: int
    actions: int
    false_crossings: int

    @property
    def recall(self) -> float:
        return self.hits / self.actions if self.actions else 0.0

    @property
    def precision(self) -> float:
        total = self.hits + self.false_crossings
        return self.hits / total if total else 0.0

    @property
    def strength(self) -> float:
        return self.precision * self.recall


def _crossings(series: SignalSeries, threshold: float, direction: str) -> list[float]:
    """Timestamps at which the series crosses *threshold* in *direction*."""
    out: list[float] = []
    previous: float | None = None
    for ts, value in zip(series.times, series.values, strict=True):
        try:
            current = float(value)
        except (TypeError, ValueError):
            continue
        if previous is not None:
            if direction == "falling" and previous >= threshold > current:
                out.append(ts)
            elif direction == "rising" and previous <= threshold < current:
                out.append(ts)
        previous = current
    return out


def _best_crossing(
    series: SignalSeries, action_ts: Sequence[float], lead: float = LEAD_WINDOW_SECONDS
) -> Crossing | None:
    """Threshold whose crossings best predict the action times."""
    values = [float(v) for v in series.values if isinstance(v, (int, float))]
    if len(values) < 20 or len(action_ts) < MIN_ACTIONS:
        return None
    ordered = sorted(values)
    # Test deciles rather than every observed value: enough resolution for a
    # human-readable rule, and O(10) instead of O(n) full passes.
    thresholds = sorted({round(ordered[int(len(ordered) * q / 10)], 1) for q in range(1, 10)})

    best: Crossing | None = None
    for threshold in thresholds:
        for direction in ("falling", "rising"):
            crossing_ts = _crossings(series, threshold, direction)
            if not crossing_ts:
                continue
            matched_actions = 0
            used: set[int] = set()
            for action in action_ts:
                for index, ts in enumerate(crossing_ts):
                    if 0 <= action - ts <= lead and index not in used:
                        used.add(index)
                        matched_actions += 1
                        break
            false_crossings = len(crossing_ts) - len(used)
            candidate = Crossing(
                entity_id=series.entity_id,
                threshold=threshold,
                direction=direction,
                hits=matched_actions,
                actions=len(action_ts),
                false_crossings=false_crossings,
            )
            if candidate.recall < 0.5:
                continue
            if best is None or candidate.strength > best.strength:
                best = candidate
    return best


def find_motifs(series: SignalSeries, window_size: int = 12, max_motifs: int = 3) -> list[dict[str, Any]]:
    """Matrix-profile motifs for *series*; ``[]`` when stumpy is unavailable."""
    if not stumpy_available() or len(series) < window_size * 4:
        return []
    try:
        import numpy as np
        import stumpy  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - guarded by stumpy_available()
        return []
    try:
        values = np.array([float(v) for v in series.values], dtype=float)
        profile = stumpy.stump(values, m=window_size)
        distances = profile[:, 0].astype(float)
        order = np.argsort(distances)[:max_motifs]
        return [
            {
                "index": int(i),
                "neighbour": int(profile[i, 1]),
                "distance": float(distances[i]),
                "start_ts": series.times[int(i)],
                "window_size": window_size,
            }
            for i in order
            if np.isfinite(distances[i])
        ]
    except Exception as err:  # noqa: BLE001 - stumpy is optional, never fatal
        # At debug level a genuine bug in this path is invisible in production;
        # a warning costs nothing, because it can only fire once per series and
        # the caller degrades cleanly either way.
        _LOGGER.warning(
            "Matrix-profile search failed for %s (%s: %s); falling back to "
            "threshold-crossing search",
            series.entity_id,
            type(err).__name__,
            err,
        )
        return []


def mine(
    changes: Sequence[StateChange],
    options: Options,
    store: SignalStore,
    window: tuple[float, float],
    resolver=None,
) -> list[Candidate]:
    """Mine numeric-driven conditional rules, motif-assisted when possible."""
    numeric = store.numeric_entities()
    if not numeric:
        return []
    grouped = human_action_events(changes, options)
    used_stumpy = stumpy_available()
    candidates: list[Candidate] = []

    for (entity_id, state), rows in grouped.items():
        if len(rows) < MIN_ACTIONS:
            continue
        service = service_for(entity_id, state)
        if service is None:
            continue
        action_ts = sorted(row.ts for row in rows)

        best: Crossing | None = None
        for signal_entity in numeric:
            if signal_entity.split("#", 1)[0] == entity_id:
                continue
            series = store.get(signal_entity)
            if series is None:
                continue
            crossing = _best_crossing(series, action_ts)
            if crossing is not None and (best is None or crossing.strength > best.strength):
                best = crossing
        if best is None or best.strength < 0.4:
            continue

        series = store.get(best.entity_id)
        motifs = find_motifs(series) if series is not None else []

        service_name, service_data = service
        name = resolver.name_of(entity_id) if resolver else entity_id
        signal_name = resolver.name_of(best.entity_id) if resolver else best.entity_id
        verb = service_name.split(".", 1)[-1].replace("_", " ")
        direction_word = "drops below" if best.direction == "falling" else "rises above"

        trigger = Trigger(
            kind="numeric_state",
            entity_id=best.entity_id,
            below=best.threshold if best.direction == "falling" else None,
            above=best.threshold if best.direction == "rising" else None,
        )

        candidates.append(
            Candidate(
                miner="motif",
                title=f"{verb.capitalize()} {name} when {signal_name} {direction_word} {best.threshold:g}",
                description=(
                    f"{best.hits} of your {best.actions} manual '{state}' actions on {name} "
                    f"followed within {LEAD_WINDOW_SECONDS / 60:.0f} min of {signal_name} "
                    f"{direction_word} {best.threshold:g}."
                ),
                triggers=[trigger],
                actions=[Action(service=service_name, entity_id=entity_id, data=dict(service_data))],
                evidence=Evidence(
                    occurrences=best.hits,
                    opportunities=best.actions,
                    consistency=best.recall,
                    confidence=best.precision,
                    window_start_ts=window[0],
                    window_end_ts=window[1],
                    window_days=(window[1] - window[0]) / 86400.0,
                    samples=action_ts[:50],
                    notes=[
                        f"Recall {best.recall:.0%} (actions explained), precision "
                        f"{best.precision:.0%} (crossings that led to an action).",
                        f"{best.false_crossings} crossings did not lead to an action.",
                        (
                            # find_motifs runs over the whole raw series and
                            # knows nothing about the threshold or the actions
                            # this rule is built from, so "confirmed" was a
                            # claim about a computation that never looked at
                            # the rule.  The recall and precision above are
                            # what the rule actually rests on.
                            f"Matrix profile also found {len(motifs)} repeating shape(s) "
                            "elsewhere in this signal; the rule above does not depend on them."
                            if motifs
                            else "stumpy is not installed; used deterministic threshold-crossing "
                            "search instead (same rule shape, no matrix profile)."
                        ),
                    ],
                    extra={
                        "crossing": best.__dict__,
                        "motifs": motifs,
                        "stumpy": used_stumpy,
                    },
                ),
                score=round(best.strength, 4),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    _LOGGER.info(
        "motif miner produced %d candidates (stumpy=%s)", len(candidates), used_stumpy
    )
    return candidates
