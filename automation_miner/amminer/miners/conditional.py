"""Miner F - conditional patterns against enriched external signals.

Given the human actions the time-of-day miner already grouped, ask: *what was
different about the world when the user did this?*

For each candidate signal we compare the values observed **at action time**
against a **background** sample of the same signal.  A signal is only accepted
as a condition when it is genuinely discriminative:

* numeric signals - a decision stump maximising the information gain of a
  single threshold ("outdoor temp < 8 °C"),
* categorical signals - a dominant value with a real lift over its base rate
  ("you are home", "price level is expensive").

This is what turns "you turn the heater on in the evening" into "you turn the
heater on when it is below 8 °C outside" - a rule that will not misfire in July.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ..config import Options
from ..enrich.detect import SignalSet
from ..enrich.signals import SignalStore
from ..recorderdb.models import StateChange
from .base import Action, Candidate, Condition, Evidence, Trigger
from .time_of_day import human_action_events, service_for

_LOGGER = logging.getLogger(__name__)

#: A signal read from more than this long ago is treated as unknown.
MAX_SIGNAL_STALENESS = 3 * 3600.0

MIN_POSITIVE_SAMPLES = 4
MIN_LIFT = 1.4
MIN_PURITY = 0.75


@dataclass
class ConditionFinding:
    """One discriminative signal, ready to become a :class:`Condition`."""

    entity_id: str
    kind: str  # "numeric" | "categorical"
    purity: float
    lift: float
    coverage: float
    above: float | None = None
    below: float | None = None
    value: str | None = None
    baseline: float = 0.0

    @property
    def strength(self) -> float:
        return self.purity * min(self.lift / 2.0, 1.0)

    def to_condition(self, source: str) -> Condition:
        if self.kind == "numeric":
            return Condition(
                kind="numeric_state",
                entity_id=self.entity_id,
                above=self.above,
                below=self.below,
                source=source,
            )
        return Condition(
            kind="state", entity_id=self.entity_id, state=self.value, source=source
        )


def _background_timestamps(
    action_timestamps: Sequence[float], window: tuple[float, float], per_day: int = 6
) -> list[float]:
    """Sample the analysis window uniformly, as a "what usually happens" baseline.

    Sampling at the *same times of day* as the actions would bake the
    time-of-day pattern into the baseline and hide any genuine external driver,
    so the background is a plain uniform grid instead.
    """
    start_ts, end_ts = window
    if end_ts <= start_ts:
        return []
    step = 86400.0 / max(per_day, 1)
    samples: list[float] = []
    ts = start_ts
    while ts < end_ts:
        samples.append(ts)
        ts += step
    return samples


def _numeric_stump(
    positives: Sequence[float], background: Sequence[float]
) -> ConditionFinding | None:
    """Best single threshold separating action-time values from the background."""
    if len(positives) < MIN_POSITIVE_SAMPLES or len(background) < 10:
        return None
    thresholds = sorted({round(v, 2) for v in list(positives) + list(background)})
    if len(thresholds) < 3:
        return None

    best: ConditionFinding | None = None
    for threshold in thresholds:
        for direction in ("below", "above"):
            if direction == "below":
                pos_hits = sum(1 for v in positives if v < threshold)
                bg_hits = sum(1 for v in background if v < threshold)
            else:
                pos_hits = sum(1 for v in positives if v > threshold)
                bg_hits = sum(1 for v in background if v > threshold)
            coverage = pos_hits / len(positives)
            # Laplace-smoothed baseline: an unobserved background case must not
            # produce an infinite (and meaningless) lift.
            baseline = max(bg_hits, 0.5) / len(background)
            if coverage < MIN_PURITY:
                continue
            lift = coverage / baseline
            if lift < MIN_LIFT:
                continue
            finding = ConditionFinding(
                entity_id="",
                kind="numeric",
                purity=coverage,
                lift=lift,
                coverage=coverage,
                baseline=baseline,
                below=threshold if direction == "below" else None,
                above=threshold if direction == "above" else None,
            )
            if best is None or finding.strength > best.strength:
                best = finding
    return best


def _categorical_split(
    positives: Sequence[str], background: Sequence[str]
) -> ConditionFinding | None:
    """Dominant categorical value with a real lift over its background rate."""
    if len(positives) < MIN_POSITIVE_SAMPLES or len(background) < 10:
        return None
    pos_counts = Counter(positives)
    bg_counts = Counter(background)
    value, count = pos_counts.most_common(1)[0]
    coverage = count / len(positives)
    baseline = max(bg_counts.get(value, 0), 0.5) / len(background)
    if coverage < MIN_PURITY:
        return None
    lift = coverage / baseline
    if lift < MIN_LIFT:
        return None
    return ConditionFinding(
        entity_id="",
        kind="categorical",
        purity=coverage,
        lift=lift,
        coverage=coverage,
        baseline=baseline,
        value=str(value),
    )


def find_conditions(
    action_timestamps: Sequence[float],
    store: SignalStore,
    candidate_entities: Sequence[str],
    window: tuple[float, float],
    max_conditions: int = 2,
) -> list[ConditionFinding]:
    """Rank the signals that best explain *when* the action happens."""
    background_ts = _background_timestamps(action_timestamps, window)
    findings: list[ConditionFinding] = []

    for entity_id in candidate_entities:
        series = store.get(entity_id)
        if series is None or series.empty:
            continue
        positives_raw = [
            series.value_at(ts, MAX_SIGNAL_STALENESS) for ts in action_timestamps
        ]
        background_raw = [
            series.value_at(ts, MAX_SIGNAL_STALENESS) for ts in background_ts
        ]
        positives = [v for v in positives_raw if v is not None]
        background = [v for v in background_raw if v is not None]
        if len(positives) < MIN_POSITIVE_SAMPLES:
            continue

        if series.numeric:
            finding = _numeric_stump(
                [float(v) for v in positives], [float(v) for v in background]
            )
        else:
            finding = _categorical_split(
                [str(v) for v in positives], [str(v) for v in background]
            )
        if finding is None:
            continue
        finding.entity_id = entity_id
        findings.append(finding)

    findings.sort(key=lambda f: f.strength, reverse=True)
    # Keep at most one condition per entity and cap the total: a rule with five
    # conditions is unreadable and almost always overfitted.
    chosen: list[ConditionFinding] = []
    used: set[str] = set()
    for finding in findings:
        if finding.entity_id in used:
            continue
        chosen.append(finding)
        used.add(finding.entity_id)
        if len(chosen) >= max_conditions:
            break
    return chosen


def condition_entities(signals: SignalSet) -> list[str]:
    """Signals worth testing as conditions, in rough order of usefulness."""
    ordered: list[str] = []
    for group in (
        signals.outdoor_temperature,
        signals.illuminance,
        signals.person,
        signals.workday,
        signals.holiday,
        signals.occupancy,
        signals.price_level,
        signals.energy_price,
        signals.carbon_intensity,
        signals.solar_forecast,
        signals.weather,
        signals.weather_warning,
        signals.room_presence,
    ):
        for entity_id in group:
            if entity_id not in ordered:
                ordered.append(entity_id)
    return ordered


def mine(
    changes: Sequence[StateChange],
    options: Options,
    signals: SignalSet,
    store: SignalStore,
    window: tuple[float, float],
    resolver=None,
) -> list[Candidate]:
    """Mine "you do X when the world looks like Y" candidates."""
    testable = condition_entities(signals)
    if not testable:
        _LOGGER.info("conditional miner: no external signals available, skipping")
        return []

    grouped = human_action_events(changes, options)
    candidates: list[Candidate] = []

    for (entity_id, state), rows in grouped.items():
        if len(rows) < max(options.min_occurrences, MIN_POSITIVE_SAMPLES):
            continue
        service = service_for(entity_id, state)
        if service is None:
            continue
        # Never explain an entity with itself, or with a signal on the same device.
        testable_here = [e for e in testable if e.split("#", 1)[0] != entity_id]
        timestamps = [row.ts for row in rows]
        findings = find_conditions(timestamps, store, testable_here, window)
        if not findings:
            continue

        service_name, service_data = service
        primary = findings[0]
        conditions = [f.to_condition("conditional miner") for f in findings]

        trigger_entity = primary.entity_id.split("#", 1)[0]
        if primary.kind == "numeric":
            triggers = [
                Trigger(
                    kind="numeric_state",
                    entity_id=trigger_entity,
                    above=primary.above,
                    below=primary.below,
                )
            ]
            # The trigger already encodes the primary finding.
            conditions = conditions[1:]
        else:
            triggers = [
                Trigger(kind="state", entity_id=trigger_entity, to_state=primary.value)
            ]
            conditions = conditions[1:]

        name = resolver.name_of(entity_id) if resolver else entity_id
        signal_name = resolver.name_of(trigger_entity) if resolver else trigger_entity
        verb = service_name.split(".", 1)[-1].replace("_", " ")
        if primary.kind == "numeric":
            bound = (
                f"below {primary.below:g}" if primary.below is not None else f"above {primary.above:g}"
            )
            explanation = f"{signal_name} is {bound}"
        else:
            explanation = f"{signal_name} is '{primary.value}'"

        candidates.append(
            Candidate(
                miner="conditional",
                title=f"{verb.capitalize()} {name} when {explanation}",
                description=(
                    f"You manually {verb} {name} in {primary.coverage:.0%} of cases while "
                    f"{explanation} - {primary.lift:.1f}x more often than the "
                    f"{primary.baseline:.0%} background rate."
                ),
                triggers=triggers,
                conditions=conditions,
                actions=[Action(service=service_name, entity_id=entity_id, data=dict(service_data))],
                evidence=Evidence(
                    occurrences=len(rows),
                    consistency=primary.purity,
                    confidence=primary.purity,
                    lift=primary.lift,
                    window_start_ts=window[0],
                    window_end_ts=window[1],
                    window_days=(window[1] - window[0]) / 86400.0,
                    samples=timestamps[:50],
                    notes=[
                        f"{primary.coverage:.0%} of your {len(rows)} manual '{state}' actions "
                        f"happened while {explanation}.",
                        f"Background rate for that condition is only {primary.baseline:.0%} "
                        f"(lift {primary.lift:.2f}).",
                    ]
                    + [
                        f"Secondary condition: {f.entity_id} "
                        f"(purity {f.purity:.0%}, lift {f.lift:.2f})"
                        for f in findings[1:]
                    ],
                    extra={
                        "target_state": state,
                        "findings": [f.__dict__ for f in findings],
                    },
                ),
                score=round(min(primary.strength, 1.0), 4),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    _LOGGER.info("conditional miner produced %d candidates", len(candidates))
    return candidates
