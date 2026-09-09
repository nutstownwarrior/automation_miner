"""Miner A - time-of-day / day-of-week habit mining on **human** actions.

The naive version of this ("you turned the light on at 06:31, 06:29, 06:33 -
here's an automation") is what most suggestion add-ons ship.  Two things make
this one materially better:

1. Only **human-caused** changes are mined.  Actions an automation already
   performs are excluded via the context chain, so we never suggest automating
   something that is already automated.
2. Consistency is measured against *opportunities*, not against occurrences.
   Firing on 6 of 7 weekdays scores much higher than 6 scattered hits across
   60 days, and the number of eligible days is reported in the evidence.

Clustering is circular (23:55 and 00:05 are 10 minutes apart, not 23 hours).
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..config import ACTIONABLE_DOMAINS, Options
from ..recorderdb.models import Cause, StateChange
from ..util.timeutil import (
    MINUTES_PER_DAY,
    circular_distance,
    circular_mean_minutes,
    circular_std_minutes,
    day_key,
    hhmm,
    local_tz,
    minute_of_day,
)
from .base import Action, Candidate, Condition, Evidence, Trigger

_LOGGER = logging.getLogger(__name__)

WEEKDAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

#: Service to call for a given (domain, target state).
_SERVICE_MAP: dict[tuple[str, str], str] = {
    ("light", "on"): "light.turn_on",
    ("light", "off"): "light.turn_off",
    ("switch", "on"): "switch.turn_on",
    ("switch", "off"): "switch.turn_off",
    ("fan", "on"): "fan.turn_on",
    ("fan", "off"): "fan.turn_off",
    ("input_boolean", "on"): "input_boolean.turn_on",
    ("input_boolean", "off"): "input_boolean.turn_off",
    ("siren", "on"): "siren.turn_on",
    ("siren", "off"): "siren.turn_off",
    ("humidifier", "on"): "humidifier.turn_on",
    ("humidifier", "off"): "humidifier.turn_off",
    ("cover", "open"): "cover.open_cover",
    ("cover", "closed"): "cover.close_cover",
    ("lock", "locked"): "lock.lock",
    ("lock", "unlocked"): "lock.unlock",
    ("media_player", "off"): "media_player.turn_off",
    ("media_player", "playing"): "media_player.media_play",
    ("vacuum", "cleaning"): "vacuum.start",
    ("vacuum", "docked"): "vacuum.return_to_base",
    ("water_heater", "on"): "water_heater.turn_on",
    ("water_heater", "off"): "water_heater.turn_off",
    ("valve", "open"): "valve.open_valve",
    ("valve", "closed"): "valve.close_valve",
}


def service_for(entity_id: str, state: str) -> tuple[str, dict] | None:
    """Map ``(entity, target state)`` onto a concrete service call."""
    domain = entity_id.split(".", 1)[0]
    state = (state or "").lower()
    direct = _SERVICE_MAP.get((domain, state))
    if direct:
        return direct, {}
    if domain == "climate" and state not in ("off", "unavailable", "unknown"):
        return "climate.set_hvac_mode", {"hvac_mode": state}
    if domain == "climate" and state == "off":
        return "climate.turn_off", {}
    if domain in ("input_select", "select") and state:
        return f"{domain}.select_option", {"option": state}
    if domain == "scene":
        return "scene.turn_on", {}
    if domain == "script":
        return "script.turn_on", {}
    if state in ("on", "off") and domain in ACTIONABLE_DOMAINS:
        return f"homeassistant.turn_{state}", {}
    return None


@dataclass
class DayFilter:
    """Which days a habit applies to."""

    key: str  # "all" | "weekdays" | "weekends" | "mon" ...
    weekdays: tuple[int, ...]

    @property
    def label(self) -> str:
        return {"all": "every day", "weekdays": "weekdays", "weekends": "weekends"}.get(
            self.key, self.key
        )

    @property
    def condition_weekdays(self) -> list[str]:
        if self.key == "all":
            return []
        return [WEEKDAY_NAMES[d] for d in self.weekdays]


DAY_FILTERS = (
    DayFilter("all", (0, 1, 2, 3, 4, 5, 6)),
    DayFilter("weekdays", (0, 1, 2, 3, 4)),
    DayFilter("weekends", (5, 6)),
)


def _eligible_days(start_ts: float, end_ts: float, weekdays: Sequence[int], tz) -> int:
    """Count calendar days in the window matching *weekdays*."""
    if end_ts <= start_ts:
        return 0
    first = dt.datetime.fromtimestamp(start_ts, tz).date()
    last = dt.datetime.fromtimestamp(end_ts, tz).date()
    allowed = set(weekdays)
    count = 0
    day = first
    while day <= last:
        if day.weekday() in allowed:
            count += 1
        day += dt.timedelta(days=1)
    return count


def _best_cluster(minutes: Sequence[int], bandwidth: int) -> tuple[float, list[int]] | None:
    """Find the densest circular cluster of minute-of-day values.

    Each observed minute is treated as a candidate centre; the window that
    captures the most *distinct* observations wins.  This is O(n^2) but n is the
    number of times a person touched one entity, which is small.
    """
    if not minutes:
        return None
    best_centre: float | None = None
    best_members: list[int] = []
    for centre in sorted(set(minutes)):
        members = [m for m in minutes if circular_distance(m, centre) <= bandwidth]
        if len(members) > len(best_members):
            best_members = members
            best_centre = float(centre)
    if best_centre is None or not best_members:
        return None
    # Re-centre on the circular mean of the members for a tighter fit.
    refined = circular_mean_minutes(best_members)
    members = [m for m in minutes if circular_distance(m, refined) <= bandwidth]
    if len(members) >= len(best_members):
        return refined % MINUTES_PER_DAY, members
    return best_centre % MINUTES_PER_DAY, best_members


def human_action_events(
    changes: Iterable[StateChange], options: Options
) -> dict[tuple[str, str], list[StateChange]]:
    """Group genuinely human state transitions by ``(entity_id, target state)``."""
    grouped: dict[tuple[str, str], list[StateChange]] = defaultdict(list)
    for change in changes:
        if change.cause is not Cause.HUMAN:
            continue
        if change.domain not in ACTIONABLE_DOMAINS:
            continue
        if options.is_excluded(change.entity_id):
            continue
        state = (change.state or "").lower()
        if state in ("unknown", "unavailable", ""):
            continue
        if not change.is_transition:
            continue
        grouped[(change.entity_id, state)].append(change)
    return grouped


def mine(
    changes: Sequence[StateChange],
    options: Options,
    window: tuple[float, float] | None = None,
    resolver=None,
) -> list[Candidate]:
    """Mine time-of-day habits from classified state changes."""
    if not changes:
        return []
    tz = local_tz()
    start_ts = window[0] if window else min(c.ts for c in changes)
    end_ts = window[1] if window else max(c.ts for c in changes)
    window_days = max((end_ts - start_ts) / 86400.0, 1.0)

    grouped = human_action_events(changes, options)
    candidates: list[Candidate] = []

    for (entity_id, state), rows in grouped.items():
        if len(rows) < options.min_occurrences:
            continue
        service = service_for(entity_id, state)
        if service is None:
            continue
        service_name, service_data = service

        best: tuple[float, DayFilter, float, list[StateChange], int] | None = None
        for day_filter in DAY_FILTERS:
            subset = [r for r in rows if dt.datetime.fromtimestamp(r.ts, tz).weekday() in day_filter.weekdays]
            if len(subset) < options.min_occurrences:
                continue
            minutes = [minute_of_day(r.ts, tz) for r in subset]
            cluster = _best_cluster(minutes, options.time_cluster_minutes)
            if cluster is None:
                continue
            centre, _members = cluster
            # Keep at most one occurrence per calendar day: a habit is
            # "happens on this day", not "happened five times that evening".
            per_day: dict[dt.date, StateChange] = {}
            for row in subset:
                if circular_distance(minute_of_day(row.ts, tz), centre) > options.time_cluster_minutes:
                    continue
                key = day_key(row.ts, tz)
                if key not in per_day or row.ts < per_day[key].ts:
                    per_day[key] = row
            hits = sorted(per_day.values(), key=lambda r: r.ts)
            if len(hits) < options.min_occurrences:
                continue
            opportunities = _eligible_days(start_ts, end_ts, day_filter.weekdays, tz)
            if opportunities <= 0:
                continue
            consistency = min(len(hits) / opportunities, 1.0)
            if best is None or consistency > best[0]:
                best = (consistency, day_filter, centre, hits, opportunities)

        if best is None:
            continue
        consistency, day_filter, centre, hits, opportunities = best
        if consistency < options.min_consistency:
            continue

        minutes = [minute_of_day(r.ts, tz) for r in hits]
        spread = circular_std_minutes(minutes)
        trigger_time = hhmm(circular_mean_minutes(minutes))

        conditions: list[Condition] = []
        if day_filter.condition_weekdays:
            conditions.append(
                Condition(
                    kind="time",
                    weekday=day_filter.condition_weekdays,
                    source="day-of-week clustering",
                )
            )

        name = resolver.name_of(entity_id) if resolver else entity_id
        area = resolver.resolve(entity_id).area_name if resolver else None
        where = f" in {area}" if area else ""
        verb = service_name.split(".", 1)[-1].replace("_", " ")

        candidate = Candidate(
            miner="time_of_day",
            title=f"{verb.capitalize()} {name}{where} at {trigger_time[:5]} ({day_filter.label})",
            triggers=[Trigger(kind="time", at=trigger_time)],
            conditions=conditions,
            actions=[Action(service=service_name, entity_id=entity_id, data=dict(service_data))],
            evidence=Evidence(
                occurrences=len(hits),
                opportunities=opportunities,
                consistency=consistency,
                spread_minutes=spread,
                window_start_ts=start_ts,
                window_end_ts=end_ts,
                window_days=window_days,
                samples=[r.ts for r in hits],
                notes=[
                    f"{len(hits)} of {opportunities} {day_filter.label} had a manual "
                    f"'{state}' within ±{options.time_cluster_minutes} min of {trigger_time[:5]}",
                    "Only human-caused changes were counted (context user id present).",
                ],
                extra={"target_state": state, "day_filter": day_filter.key},
            ),
            score=round(consistency * min(len(hits) / max(options.min_occurrences, 1), 2.0) / 2.0, 4),
        )
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.score, reverse=True)
    _LOGGER.info("time_of_day miner produced %d candidates", len(candidates))
    return candidates
