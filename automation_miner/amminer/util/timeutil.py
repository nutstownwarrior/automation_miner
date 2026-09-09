"""Timestamp helpers.

The recorder stores unix timestamps (float seconds, UTC).  All internal maths is
done on those; only presentation converts to local time.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SECONDS_PER_DAY = 86400
MINUTES_PER_DAY = 1440


def local_tz() -> dt.tzinfo:
    """Best-effort local timezone (Supervisor sets ``TZ`` in the container)."""
    name = os.environ.get("TZ")
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return dt.datetime.now().astimezone().tzinfo or dt.UTC


def now_ts() -> float:
    return time.time()


def to_local(ts: float, tz: dt.tzinfo | None = None) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, tz or local_tz())


def to_utc(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, dt.UTC)


def minute_of_day(ts: float, tz: dt.tzinfo | None = None) -> int:
    local = to_local(ts, tz)
    return local.hour * 60 + local.minute


def day_key(ts: float, tz: dt.tzinfo | None = None) -> dt.date:
    return to_local(ts, tz).date()


def is_weekend(ts: float, tz: dt.tzinfo | None = None) -> bool:
    return to_local(ts, tz).weekday() >= 5


def hhmm(minutes: int) -> str:
    minutes = int(round(minutes)) % MINUTES_PER_DAY
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def circular_mean_minutes(values: list[int] | list[float]) -> float:
    """Mean of minute-of-day values on a 24 h circle."""
    import math

    if not values:
        return 0.0
    angles = [2 * math.pi * (v % MINUTES_PER_DAY) / MINUTES_PER_DAY for v in values]
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return float(values[0] % MINUTES_PER_DAY)
    mean_angle = math.atan2(sin_sum / len(angles), cos_sum / len(angles))
    return (mean_angle % (2 * math.pi)) * MINUTES_PER_DAY / (2 * math.pi)


def circular_distance(a: float, b: float, period: int = MINUTES_PER_DAY) -> float:
    """Shortest distance between two points on a circle of length *period*."""
    diff = abs(a - b) % period
    return min(diff, period - diff)


def circular_std_minutes(values: list[int] | list[float]) -> float:
    """Circular standard deviation, in minutes."""
    import math

    if len(values) < 2:
        return 0.0
    angles = [2 * math.pi * (v % MINUTES_PER_DAY) / MINUTES_PER_DAY for v in values]
    sin_mean = sum(math.sin(a) for a in angles) / len(angles)
    cos_mean = sum(math.cos(a) for a in angles) / len(angles)
    r = math.sqrt(sin_mean**2 + cos_mean**2)
    r = min(max(r, 1e-9), 1.0)
    return math.sqrt(-2.0 * math.log(r)) * MINUTES_PER_DAY / (2 * math.pi)
