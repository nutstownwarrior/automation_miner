"""Replay a candidate against real history before showing it to anyone.

The idea is HAWatcher's shadow execution (USENIX Security 2021): run the rule
against the recorded past without acting, and compare what it *would* have done
with what the user actually did.

We compute, over the analysis window:

``true_fires``
    the rule fired and the user really did that action around then,
``false_fires``
    the rule fired and the user did not - the nuisance metric that matters most,
``missed``
    the user did the action and the rule would not have fired,
``precision`` / ``recall``
    the usual ratios, plus ``false_fires_per_week`` which is what a person
    actually feels.

A candidate is only surfaced when it clears every one of
``backtest_min_true_fires``, ``backtest_min_precision``, ``backtest_min_recall``,
``backtest_max_false_fires_per_week`` and ``backtest_max_nuisance_fires``.

The evidence floor comes first and is the one that is easy to leave out.  A rule
that fired once, correctly, has 100% precision and zero false fires per week: it
clears every ratio in this module while resting on a single observation.  Ratios
only mean something once there is enough underneath them to divide, so a
candidate must have been right ``backtest_min_true_fires`` times before its
percentages are allowed to speak for it.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import Options
from .enrich.signals import SignalStore
from .miners.base import Candidate, Condition, Trigger
from .recorderdb.models import Cause, OverrideEvent, StateChange
from .util.timeutil import local_tz, minute_of_day, time_of_day_minutes

_LOGGER = logging.getLogger(__name__)

WEEKDAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

#: A state trigger must line up much more tightly than a daily time trigger.
STATE_TRIGGER_TOLERANCE = 300.0

#: Trigger kinds that are events rather than clock times, and so are held to
#: :data:`STATE_TRIGGER_TOLERANCE` regardless of what else the rule triggers on.
EVENT_TRIGGER_KINDS = ("state", "numeric_state")


@dataclass(frozen=True)
class Fire:
    """One moment the rule would have fired, and what set it off.

    The trigger kind travels with the fire because the two kinds are matched
    against reality at different tolerances: "some time in the next quarter
    hour" is a reasonable reading of a daily habit, and a dishonest one for a
    rule that claims the door opening caused the light.
    """

    ts: float
    trigger_kind: str

    @property
    def tolerance_is_tight(self) -> bool:
        return self.trigger_kind in EVENT_TRIGGER_KINDS


@dataclass
class BacktestResult:
    """Outcome of replaying one candidate."""

    true_fires: int = 0
    false_fires: int = 0
    missed: int = 0
    total_fires: int = 0
    window_days: float = 0.0
    nuisance_fires: int = 0
    passed: bool = False
    reason: str = ""
    simulated: bool = True
    fire_samples: list[float] = field(default_factory=list)
    false_fire_samples: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        total = self.true_fires + self.false_fires
        return self.true_fires / total if total else None

    @property
    def recall(self) -> float | None:
        total = self.true_fires + self.missed
        return self.true_fires / total if total else None

    @property
    def false_fires_per_week(self) -> float:
        if self.window_days <= 0:
            return 0.0
        return self.false_fires * 7.0 / self.window_days

    def as_dict(self) -> dict[str, Any]:
        return {
            "true_fires": self.true_fires,
            "false_fires": self.false_fires,
            "missed": self.missed,
            "total_fires": self.total_fires,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "false_fires_per_week": round(self.false_fires_per_week, 2),
            "nuisance_fires": self.nuisance_fires,
            "window_days": round(self.window_days, 1),
            "passed": self.passed,
            "reason": self.reason,
            "simulated": self.simulated,
            "fire_samples": self.fire_samples[:20],
            "false_fire_samples": self.false_fire_samples[:20],
            "summary": self.summary(),
        }

    def summary(self) -> str:
        if not self.simulated:
            return self.reason or "not simulated"
        precision = f"{self.precision:.0%}" if self.precision is not None else "n/a"
        recall = f"{self.recall:.0%}" if self.recall is not None else "n/a"
        return (
            f"{self.true_fires} correct, {self.false_fires} unwanted "
            f"({self.false_fires_per_week:.1f}/week), {self.missed} missed - "
            f"precision {precision}, recall {recall}"
        )


# ----------------------------------------------------------------------
def _condition_holds(
    condition: Condition, ts: float, store: SignalStore, tz
) -> bool:
    if condition.kind == "time":
        if condition.weekday:
            allowed = {WEEKDAY_INDEX[d] for d in condition.weekday if d in WEEKDAY_INDEX}
            if allowed and dt.datetime.fromtimestamp(ts, tz).weekday() not in allowed:
                return False
        minute = minute_of_day(ts, tz)
        for bound, fails in ((condition.after, lambda m, b: m < b),
                             (condition.before, lambda m, b: m > b)):
            if not bound:
                continue
            limit = time_of_day_minutes(bound)
            if limit is None:
                # An unevaluable bound is "cannot verify", which for a gate
                # means not satisfied - never silently no constraint at all.
                return False
            if fails(minute, limit):
                return False
        return True
    if condition.kind == "state":
        value = store.value_at(condition.entity_id or "", ts, max_staleness=6 * 3600)
        if value is None:
            return False
        return str(value).lower() == str(condition.state).lower()
    if condition.kind == "numeric_state":
        value = store.numeric_at(condition.entity_id or "", ts, max_staleness=6 * 3600)
        if value is None:
            return False
        if condition.above is not None and value <= condition.above:
            return False
        if condition.below is not None and value >= condition.below:
            return False
        return True
    # Unknown condition kinds are treated as "cannot verify" -> not satisfied,
    # which is the conservative choice for a gate.
    return False


def _time_trigger_fires(trigger: Trigger, window: tuple[float, float], tz) -> list[float]:
    """Every daily occurrence of a ``time`` trigger inside the window."""
    minutes = time_of_day_minutes(trigger.at or "")
    if minutes is None:
        return []
    start_ts, end_ts = window
    day = dt.datetime.fromtimestamp(start_ts, tz).date()
    last = dt.datetime.fromtimestamp(end_ts, tz).date()
    fires: list[float] = []
    while day <= last:
        moment = dt.datetime.combine(
            day, dt.time(minutes // 60, minutes % 60), tzinfo=tz
        ).timestamp()
        if start_ts <= moment <= end_ts:
            fires.append(moment)
        day += dt.timedelta(days=1)
    return fires


def _state_trigger_fires(trigger: Trigger, store: SignalStore) -> list[float]:
    series = store.get(trigger.entity_id or "")
    if series is None:
        return []
    fires: list[float] = []
    previous: Any = None
    for ts, value in zip(series.times, series.values, strict=True):
        text = str(value).lower()
        if trigger.to_state is None:
            if previous is not None and text != str(previous).lower():
                fires.append(ts)
        elif text == trigger.to_state.lower() and (
            previous is None or str(previous).lower() != text
        ):
            fires.append(ts)
        previous = value
    return fires


def _numeric_trigger_fires(trigger: Trigger, store: SignalStore) -> list[float]:
    series = store.get(trigger.entity_id or "")
    if series is None:
        return []
    fires: list[float] = []
    previous: float | None = None
    for ts, value in zip(series.times, series.values, strict=True):
        try:
            current = float(value)
        except (TypeError, ValueError):
            continue
        if previous is not None:
            crossed_below = (
                trigger.below is not None and previous >= trigger.below > current
            )
            crossed_above = (
                trigger.above is not None and previous <= trigger.above < current
            )
            if crossed_below or crossed_above:
                fires.append(ts)
        previous = current
    return fires


def simulate_fires(
    candidate: Candidate, store: SignalStore, window: tuple[float, float], tz=None
) -> tuple[list[Fire], str | None]:
    """Every moment the candidate would have fired.  Returns ``(fires, error)``."""
    tz = tz or local_tz()
    fires: list[Fire] = []
    for trigger in candidate.triggers:
        if trigger.kind == "time":
            times = _time_trigger_fires(trigger, window, tz)
        elif trigger.kind == "state":
            times = _state_trigger_fires(trigger, store)
        elif trigger.kind == "numeric_state":
            times = _numeric_trigger_fires(trigger, store)
        else:
            return [], f"trigger kind '{trigger.kind}' cannot be simulated"
        fires.extend(Fire(ts, trigger.kind) for ts in times)
    if not fires:
        return [], None

    # Two triggers landing on the same instant is one fire, held to the tighter
    # of the two tolerances - a rule does not get the benefit of its loosest
    # trigger just for having one.
    by_ts: dict[float, Fire] = {}
    for fire in fires:
        existing = by_ts.get(fire.ts)
        if existing is None or (fire.tolerance_is_tight and not existing.tolerance_is_tight):
            by_ts[fire.ts] = fire

    kept = [
        by_ts[ts]
        for ts in sorted(by_ts)
        if window[0] <= ts <= window[1]
        and all(_condition_holds(c, ts, store, tz) for c in candidate.conditions)
    ]
    # Collapse bursts: a rule that fires five times in a minute is one event to
    # a human, and Home Assistant's own trigger would also be re-entrant-guarded.
    collapsed: list[Fire] = []
    for fire in kept:
        if not collapsed or fire.ts - collapsed[-1].ts > 60.0:
            collapsed.append(fire)
        elif fire.tolerance_is_tight and not collapsed[-1].tolerance_is_tight:
            # Same burst, but this one must be matched more strictly.
            collapsed[-1] = Fire(collapsed[-1].ts, fire.trigger_kind)
    return collapsed, None


def ground_truth_actions(
    candidate: Candidate, changes: Sequence[StateChange]
) -> list[float]:
    """When the user really performed the candidate's action, by hand."""
    wanted: set[tuple[str, str]] = set()
    for action in candidate.actions:
        if not action.entity_id:
            continue
        target = action.target_state
        if target is None:
            target = str(
                action.data.get("hvac_mode") or action.data.get("option") or ""
            ).lower()
        if target:
            wanted.add((action.entity_id, target))
    if not wanted:
        return []
    return sorted(
        change.ts
        for change in changes
        if change.cause is Cause.HUMAN
        and change.is_transition
        and (change.entity_id, change.state.lower()) in wanted
    )


def backtest(
    candidate: Candidate,
    changes: Sequence[StateChange],
    store: SignalStore,
    options: Options,
    window: tuple[float, float],
    overrides: Sequence[OverrideEvent] = (),
) -> BacktestResult:
    """Replay one candidate and decide whether it may be surfaced."""
    tz = local_tz()
    window_days = max((window[1] - window[0]) / 86400.0, 0.01)
    result = BacktestResult(window_days=window_days)

    if not candidate.actions:
        result.simulated = False
        result.reason = "candidate has no action to simulate"
        result.passed = True  # audit-only findings (stale/unused) are not gated
        return result

    fires, error = simulate_fires(candidate, store, window, tz)
    if error:
        result.simulated = False
        result.reason = error
        result.passed = False
        return result

    truth = ground_truth_actions(candidate, changes)
    if not truth:
        result.simulated = False
        result.reason = "no manual occurrences of this action found to compare against"
        result.passed = False
        return result

    unmatched_truth = list(truth)
    for fire in fires:
        # Per fire, not per candidate: a rule that triggers on both a clock time
        # and a door opening must not judge the door by the clock's tolerance.
        tolerance = (
            STATE_TRIGGER_TOLERANCE
            if fire.tolerance_is_tight
            else float(options.backtest_match_tolerance_seconds)
        )
        best_index: int | None = None
        best_delta = tolerance + 1
        for index, action_ts in enumerate(unmatched_truth):
            delta = abs(action_ts - fire.ts)
            if delta <= tolerance and delta < best_delta:
                best_index, best_delta = index, delta
        if best_index is None:
            result.false_fires += 1
            result.false_fire_samples.append(fire.ts)
        else:
            result.true_fires += 1
            result.fire_samples.append(fire.ts)
            unmatched_truth.pop(best_index)

    result.missed = len(unmatched_truth)
    result.total_fires = len(fires)

    # A false fire on an entity the user has historically overridden is worse
    # than a merely unnecessary one - it is an active annoyance.
    override_ts = [o.ts for o in overrides if o.entity_id in candidate.target_entities]
    result.nuisance_fires = sum(
        1
        for fire_ts in result.false_fire_samples
        if any(0 <= o - fire_ts <= options.override_window_seconds * 4 for o in override_ts)
    )

    precision = result.precision
    recall = result.recall
    reasons: list[str] = []

    # The floor comes before the ratios.  One correct fire and no wrong ones is
    # 100% precision, 0 nuisance fires per week, and one observation - it clears
    # every threshold below without having shown anything.
    if precision is None:
        reasons.append("the rule never fired in the analysed window")
    else:
        if result.true_fires < options.backtest_min_true_fires:
            reasons.append(
                f"the rule was only right {result.true_fires} "
                f"time{'' if result.true_fires == 1 else 's'} in "
                f"{result.window_days:.0f} days, which is too little to judge it on "
                f"(at least {options.backtest_min_true_fires} needed)"
            )
        if precision < options.backtest_min_precision:
            reasons.append(
                f"precision {precision:.0%} is below the "
                f"{options.backtest_min_precision:.0%} threshold"
            )
        if recall is not None and recall < options.backtest_min_recall:
            reasons.append(
                f"the rule would only have covered {recall:.0%} of the "
                f"{result.true_fires + result.missed} times you actually did this, "
                f"below the {options.backtest_min_recall:.0%} threshold"
            )

    if result.false_fires_per_week > options.backtest_max_false_fires_per_week:
        reasons.append(
            f"{result.false_fires_per_week:.1f} unwanted fires per week exceeds the "
            f"limit of {options.backtest_max_false_fires_per_week:g}"
        )
    # Computing this and then not acting on it was the worst of both: a rule
    # that fires exactly where the user has already reached over and undone an
    # automation is the one most likely to be resented.
    if result.nuisance_fires > options.backtest_max_nuisance_fires:
        reasons.append(
            f"{result.nuisance_fires} of its unwanted fires land where you have "
            "previously overridden an automation"
        )

    result.passed = not reasons
    result.reason = (
        "; ".join(reasons)
        if reasons
        else f"right {result.true_fires} times, meets every threshold"
    )
    return result


def backtest_all(
    candidates: Sequence[Candidate],
    changes: Sequence[StateChange],
    store: SignalStore,
    options: Options,
    window: tuple[float, float],
    overrides: Sequence[OverrideEvent] = (),
) -> tuple[list[Candidate], list[Candidate]]:
    """Backtest every candidate; return ``(passed, rejected)``."""
    passed: list[Candidate] = []
    rejected: list[Candidate] = []
    for candidate in candidates:
        result = backtest(candidate, changes, store, options, window, overrides)
        candidate.backtest = result.as_dict()
        if result.passed:
            # A well-backtested rule deserves to outrank a merely frequent one.
            if result.precision is not None:
                candidate.score = round((candidate.score + result.precision) / 2.0, 4)
            passed.append(candidate)
        else:
            rejected.append(candidate)
    _LOGGER.info(
        "backtest: %d passed, %d rejected of %d candidates",
        len(passed),
        len(rejected),
        len(candidates),
    )
    return passed, rejected
