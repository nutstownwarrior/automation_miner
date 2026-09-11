"""Time-indexed signal series, used to evaluate conditions at any past moment.

A :class:`SignalSeries` is a sorted list of ``(ts, value)`` samples with a
``value_at(ts)`` that returns the last value at or before ``ts`` - exactly the
semantics Home Assistant conditions have.  This is what lets the conditional
miner and the backtester ask "what was the outdoor temperature when the user
turned the heater on?" without re-querying the recorder.

Long-term statistics are used as a *fallback* source for numeric sensors: they
survive ``purge_keep_days``, so seasonality is still minable on a default
10-day-retention SQLite install.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..recorderdb.models import StateChange

_LOGGER = logging.getLogger(__name__)


@dataclass
class SignalSeries:
    """One entity's history as a step function over time."""

    entity_id: str
    times: list[float] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)
    numeric: bool = False
    source: str = "states"

    def __len__(self) -> int:
        return len(self.times)

    @property
    def empty(self) -> bool:
        return not self.times

    def add(self, ts: float, value: Any) -> None:
        self.times.append(ts)
        self.values.append(value)

    def finalise(self) -> SignalSeries:
        if self.times:
            order = sorted(range(len(self.times)), key=lambda i: self.times[i])
            self.times = [self.times[i] for i in order]
            self.values = [self.values[i] for i in order]
        return self

    def value_at(self, ts: float, max_staleness: float | None = None) -> Any:
        """Last value at or before *ts* (``None`` when unknown or too stale)."""
        if not self.times:
            return None
        index = bisect.bisect_right(self.times, ts) - 1
        if index < 0:
            return None
        if max_staleness is not None and ts - self.times[index] > max_staleness:
            return None
        return self.values[index]

    def numeric_at(self, ts: float, max_staleness: float | None = None) -> float | None:
        value = self.value_at(ts, max_staleness)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def sample_values(self, timestamps: Iterable[float], max_staleness: float | None = None) -> list[Any]:
        return [self.value_at(ts, max_staleness) for ts in timestamps]

    def distinct_values(self, limit: int = 20) -> list[Any]:
        seen: list[Any] = []
        for value in self.values:
            if value not in seen:
                seen.append(value)
                if len(seen) >= limit:
                    break
        return seen


class SignalStore:
    """All signal series for one analysis run."""

    def __init__(self) -> None:
        self.series: dict[str, SignalSeries] = {}

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self.series

    def __len__(self) -> int:
        return len(self.series)

    def get(self, entity_id: str) -> SignalSeries | None:
        return self.series.get(entity_id)

    def add(self, series: SignalSeries) -> None:
        if not series.empty:
            self.series[series.entity_id] = series.finalise()

    def value_at(self, entity_id: str, ts: float, max_staleness: float | None = None) -> Any:
        series = self.series.get(entity_id)
        return series.value_at(ts, max_staleness) if series else None

    def numeric_at(self, entity_id: str, ts: float, max_staleness: float | None = None) -> float | None:
        series = self.series.get(entity_id)
        return series.numeric_at(ts, max_staleness) if series else None

    def numeric_entities(self) -> list[str]:
        return sorted(e for e, s in self.series.items() if s.numeric)

    def categorical_entities(self) -> list[str]:
        return sorted(e for e, s in self.series.items() if not s.numeric)

    def stats(self) -> dict[str, Any]:
        return {
            "series": len(self.series),
            "numeric": len(self.numeric_entities()),
            "categorical": len(self.categorical_entities()),
            "from_statistics": sorted(
                e for e, s in self.series.items() if s.source == "statistics"
            ),
        }


#: Long-term statistics rows are hourly buckets stamped at the hour they open.
STATISTICS_INTERVAL_SECONDS = 3600.0


def _merge_statistics(
    raw: SignalSeries | None, stats: SignalSeries
) -> SignalSeries | None:
    """Fill the gaps around *raw* with hourly means, without replacing it.

    Statistics survive ``purge_keep_days`` and raw history does not, so for an
    entity with ten days of readings and a year of statistics the statistics
    series is far longer - and swapping one for the other traded precise values
    for hourly averages over the very period we had the real thing for.  Raw
    points win wherever they exist; statistics fill only outside their span.
    """
    if stats.empty:
        return None
    if raw is None or raw.empty:
        return stats
    if not raw.numeric:
        # An hourly mean says nothing useful about a categorical entity.
        return None
    first, last = raw.times[0], raw.times[-1]
    merged = SignalSeries(raw.entity_id, numeric=True, source=f"{raw.source}+statistics")
    for ts, value in zip(stats.times, stats.values, strict=True):
        if ts < first or ts > last:
            merged.add(ts, value)
    if merged.empty:
        return None
    for ts, value in zip(raw.times, raw.values, strict=True):
        merged.add(ts, value)
    return merged.finalise()


def build_signal_store(
    changes: Sequence[StateChange],
    wanted: Iterable[str],
    queries=None,
    window: tuple[float, float] | None = None,
) -> SignalStore:
    """Build series for *wanted* entities from raw states, then top up from LTS.

    ``wanted`` entries may carry an ``entity#attribute`` suffix (used for
    ``weather.*#temperature``), which is read from the state attributes instead
    of the state value.
    """
    store = SignalStore()
    wanted_list = list(dict.fromkeys(wanted))
    plain = {w.split("#", 1)[0] for w in wanted_list}
    attribute_wants: dict[str, list[str]] = {}
    for want in wanted_list:
        if "#" in want:
            entity_id, _, attribute = want.partition("#")
            attribute_wants.setdefault(entity_id, []).append(attribute)

    builders: dict[str, SignalSeries] = {}
    for change in changes:
        if change.entity_id not in plain:
            continue
        state = (change.state or "").lower()
        if state in ("unknown", "unavailable", ""):
            continue
        series = builders.setdefault(change.entity_id, SignalSeries(change.entity_id))
        numeric = change.numeric
        if numeric is not None:
            series.numeric = True
            series.add(change.ts, numeric)
        else:
            series.add(change.ts, change.state)
        for attribute in attribute_wants.get(change.entity_id, []):
            value = change.attributes.get(attribute)
            if value is None:
                continue
            key = f"{change.entity_id}#{attribute}"
            attr_series = builders.setdefault(key, SignalSeries(key))
            try:
                attr_series.add(change.ts, float(value))
                attr_series.numeric = True
            except (TypeError, ValueError):
                attr_series.add(change.ts, str(value))

    for series in builders.values():
        store.add(series)

    # --- fall back to long-term statistics where raw history is thin ---
    if queries is not None:
        missing = [
            entity_id
            for entity_id in plain
            if entity_id not in store or len(store.series[entity_id]) < 24
        ]
        if missing:
            try:
                available = set(queries.statistic_ids())
            except Exception as err:  # noqa: BLE001 - statistics are optional
                _LOGGER.debug("Could not list statistics: %s", err)
                available = set()
            requested = [entity_id for entity_id in missing if entity_id in available]
            if requested:
                start_ts = window[0] if window else None
                end_ts = window[1] if window else None
                try:
                    rows = queries.statistics(requested, start_ts, end_ts)
                except Exception as err:  # noqa: BLE001
                    _LOGGER.debug("Could not read statistics: %s", err)
                    rows = []
                by_entity: dict[str, SignalSeries] = {}
                for row in rows:
                    value = row.get("mean")
                    if value is None:
                        value = row.get("max")
                    if value is None:
                        continue
                    entity_id = str(row["statistic_id"])
                    series = by_entity.setdefault(
                        entity_id,
                        SignalSeries(entity_id, numeric=True, source="statistics"),
                    )
                    # start_ts opens the hour this mean covers.  Stamping the
                    # point there makes "what was the temperature at 14:05?"
                    # answerable with an average of 14:00-15:00 - readings that
                    # had not happened yet.  A backtest that can see the future
                    # is not a backtest, so the point is stamped when its
                    # interval closes and is fully in the past.
                    series.add(
                        float(row["start_ts"]) + STATISTICS_INTERVAL_SECONDS, float(value)
                    )
                for entity_id, series in by_entity.items():
                    merged = _merge_statistics(store.get(entity_id), series)
                    if merged is not None:
                        _LOGGER.info(
                            "Filled %s from long-term statistics (%d hourly points)",
                            entity_id,
                            len(series),
                        )
                        store.add(merged)

    _LOGGER.info("Signal store: %s", store.stats())
    return store
