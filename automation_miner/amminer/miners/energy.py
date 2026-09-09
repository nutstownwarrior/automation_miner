"""Energy load-shifting suggestions.

Only runs when the instance actually has a dynamic-price signal (Tibber,
Nordpool, ENTSO-e, aWATTar/EPEX, Energi Data Service) *and* a deferrable load
(dishwasher, washing machine, EV charger, water heater, pool pump).

The finding is concrete and costed: "you start the dishwasher manually at 19:40,
which is in the most expensive third of the day 62 % of the time; the cheapest
2-hour block is 02:00-04:00, worth about X per run".
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from collections.abc import Sequence

from ..config import Options
from ..enrich.detect import SignalSet
from ..enrich.signals import SignalStore
from ..recorderdb.models import Cause, StateChange
from ..util.timeutil import hhmm, local_tz, minute_of_day
from .base import Action, Candidate, Condition, Evidence, Trigger
from .time_of_day import service_for

_LOGGER = logging.getLogger(__name__)

MIN_RUNS = 4
BLOCK_HOURS = 2

EXPENSIVE_LEVELS = {"expensive", "very_expensive", "very expensive", "high"}
CHEAP_LEVELS = {"cheap", "very_cheap", "very cheap", "low"}


def _price_series(signals: SignalSet, store: SignalStore):
    for entity_id in list(signals.energy_price) + list(signals.price_level):
        series = store.get(entity_id)
        if series is not None and not series.empty:
            return entity_id, series
    return None, None


def _hourly_price_profile(series) -> dict[int, float]:
    """Average numeric price per hour-of-day."""
    tz = local_tz()
    buckets: dict[int, list[float]] = defaultdict(list)
    for ts, value in zip(series.times, series.values, strict=True):
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        buckets[minute_of_day(ts, tz) // 60].append(price)
    return {hour: statistics.fmean(values) for hour, values in buckets.items() if values}


def _cheapest_block(profile: dict[int, float], hours: int = BLOCK_HOURS) -> tuple[int, float] | None:
    if len(profile) < 12:
        return None
    best: tuple[int, float] | None = None
    for start in range(24):
        window = [profile.get((start + offset) % 24) for offset in range(hours)]
        if any(value is None for value in window):
            continue
        mean = statistics.fmean([v for v in window if v is not None])
        if best is None or mean < best[1]:
            best = (start, mean)
    return best


def mine(
    changes: Sequence[StateChange],
    options: Options,
    signals: SignalSet,
    store: SignalStore,
    window: tuple[float, float],
    resolver=None,
) -> list[Candidate]:
    """Suggest shifting manual runs of deferrable loads into cheap price windows."""
    price_entity, price_series = _price_series(signals, store)
    if price_series is None:
        _LOGGER.info("energy miner: no dynamic price signal, skipping")
        return []
    if not signals.deferrable_loads and not signals.ev_charger:
        _LOGGER.info("energy miner: no deferrable loads detected, skipping")
        return []

    tz = local_tz()
    numeric_prices = price_series.numeric
    profile = _hourly_price_profile(price_series) if numeric_prices else {}
    cheapest = _cheapest_block(profile) if profile else None
    if numeric_prices and profile:
        prices = sorted(profile.values())
        expensive_cut = prices[int(len(prices) * 2 / 3)]
    else:
        expensive_cut = None

    loads = [
        entity_id
        for entity_id in list(signals.deferrable_loads) + list(signals.ev_charger)
        if entity_id.split(".", 1)[0] in ("switch", "vacuum", "water_heater")
        and not options.is_excluded(entity_id)
    ]
    if not loads:
        return []

    runs: dict[str, list[StateChange]] = defaultdict(list)
    for change in changes:
        if change.entity_id in loads and change.cause is Cause.HUMAN and change.is_transition:
            if change.state.lower() in ("on", "cleaning", "heat"):
                runs[change.entity_id].append(change)

    candidates: list[Candidate] = []
    for entity_id, rows in runs.items():
        if len(rows) < MIN_RUNS:
            continue
        expensive_runs = 0
        run_prices: list[float] = []
        for row in rows:
            value = price_series.value_at(row.ts, max_staleness=7200)
            if value is None:
                continue
            if numeric_prices:
                run_prices.append(float(value))
                if expensive_cut is not None and float(value) >= expensive_cut:
                    expensive_runs += 1
            elif str(value).lower() in EXPENSIVE_LEVELS:
                expensive_runs += 1
        rated = len(run_prices) if numeric_prices else len(rows)
        if rated == 0:
            continue
        expensive_share = expensive_runs / rated
        if expensive_share < 0.35:
            continue  # already well-timed - nothing to suggest

        service = service_for(entity_id, "on")
        if service is None:
            continue
        service_name, service_data = service
        name = resolver.name_of(entity_id) if resolver else entity_id
        price_name = resolver.name_of(price_entity) if resolver else price_entity
        typical_minute = int(statistics.fmean([minute_of_day(r.ts, tz) for r in rows]))

        notes = [
            f"{expensive_runs} of {rated} manual runs started in an expensive price window "
            f"({expensive_share:.0%}).",
            f"You typically start it around {hhmm(typical_minute)[:5]}.",
        ]
        conditions: list[Condition] = []
        if cheapest is not None:
            start_hour, mean_price = cheapest
            trigger = Trigger(kind="time", at=f"{start_hour:02d}:00:00")
            saving = None
            if run_prices:
                saving = statistics.fmean(run_prices) - mean_price
                notes.append(
                    f"Cheapest {BLOCK_HOURS} h block is {start_hour:02d}:00-"
                    f"{(start_hour + BLOCK_HOURS) % 24:02d}:00 at {mean_price:.3f} vs your "
                    f"average {statistics.fmean(run_prices):.3f} - about "
                    f"{saving:.3f} per unit saved per run."
                )
            title = (
                f"Shift {name} to the cheap {start_hour:02d}:00-"
                f"{(start_hour + BLOCK_HOURS) % 24:02d}:00 block"
            )
        else:
            trigger = Trigger(kind="state", entity_id=price_entity, to_state="cheap")
            conditions.append(
                Condition(kind="state", entity_id=price_entity, state="cheap", source="price level")
            )
            title = f"Run {name} when {price_name} is cheap"
            notes.append(f"Trigger on {price_name} reaching a cheap level instead of a fixed time.")

        candidates.append(
            Candidate(
                miner="energy_shift",
                title=title,
                description=(
                    f"You start {name} by hand during expensive electricity "
                    f"{expensive_share:.0%} of the time. Shifting it into the cheap window "
                    "costs you nothing in convenience for a deferrable load."
                ),
                triggers=[trigger],
                conditions=conditions,
                actions=[Action(service=service_name, entity_id=entity_id, data=dict(service_data))],
                evidence=Evidence(
                    occurrences=len(rows),
                    opportunities=rated,
                    confidence=expensive_share,
                    window_start_ts=window[0],
                    window_end_ts=window[1],
                    window_days=(window[1] - window[0]) / 86400.0,
                    samples=[r.ts for r in rows][:50],
                    notes=notes,
                    extra={
                        "price_entity": price_entity,
                        "expensive_share": expensive_share,
                        "cheapest_block": cheapest,
                        "hourly_profile": profile,
                    },
                ),
                score=round(min(expensive_share, 1.0) * min(len(rows) / 10.0, 1.0), 4),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    _LOGGER.info("energy miner produced %d candidates", len(candidates))
    return candidates
