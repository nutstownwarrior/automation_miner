"""Does what we already applied still perform?

Backtesting (:mod:`amminer.backtest`) judges a rule once, before it ships. This
module is the other half: for every automation this add-on has ever written to
Home Assistant (recorded by :meth:`amminer.store.db.Store.record_applied_automation`
at apply time), it asks the same question again, on a schedule, using history
gathered *after* the rule went live - a habit that was right in March can stop
being right in July, and nothing before this module ever looked again to find
out.

Three pieces of machinery this project already has are reused rather than
rebuilt:

``amminer.backtest.simulate_fires_detailed``
    replays the shipped rule's trigger/condition logic over real recorder
    history - exactly the same simulation the pre-apply backtest ran, now
    pointed at the period since it was applied.
``amminer.recorderdb.causality``
    every :class:`~amminer.recorderdb.models.StateChange` handed to this
    module has already been classified by cause; the automation's *own*
    recorded state rows (Home Assistant writes one every time it actually
    fires, because ``last_triggered`` changes) are how many times it really
    ran is counted.
``amminer.recorderdb.causality.detect_overrides``
    already run once per analysis, over the same window; this module only
    filters that list down to the one automation being judged.

Nothing here writes to Home Assistant, changes a suggestion, or disables
anything. The worst an unhealthy verdict does is put a sentence of
recommended action in front of the user - the retiring or retuning stays
theirs to do.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import backtest as backtest_module
from .automations import ExistingAutomation
from .config import Options
from .enrich.signals import SignalStore
from .llm.equivalence import matches
from .recorderdb.models import OverrideEvent, StateChange

STATUS_ACTIVE = "active"
STATUS_DELETED = "deleted"
STATUS_EDITED = "edited"
STATUS_DISABLED = "disabled"
STATUS_UNRESOLVED = "unresolved"

VERDICT_HEALTHY = "healthy"
VERDICT_NOISY = "noisy"
VERDICT_DORMANT = "dormant"
VERDICT_OVERRIDDEN = "overridden"
VERDICT_INSUFFICIENT = "insufficient_data"
#: Not a judgement on the rule's performance at all - shown instead of one of
#: the verdicts above whenever there is nothing to measure yet (the structural
#: statuses: deleted, edited, disabled, unresolved).
VERDICT_NA = "n/a"

#: Below this many days of history since it was applied, a verdict would rest
#: on almost nothing. Mirrors amminer.backtest's backtest_min_holdout_days
#: reasoning: "not enough evidence yet" has to be said outright rather than
#: folded into a confident-looking number.
MIN_DAYS_FOR_VERDICT = 7

#: Floor on predicted fires before "the pattern is still happening but the
#: real automation never ran" is trusted as dormancy rather than as a quiet
#: fortnight. The same reasoning as amminer.backtest's backtest_min_true_fires:
#: one predicted fire is a single observation with a threshold's name on it.
MIN_PREDICTED_FOR_DORMANT = 3

#: Share of the automation's real fires a human reversed within the override
#: window (amminer.recorderdb.causality.detect_overrides) before this counts
#: as an active annoyance rather than an occasional, ordinary correction.
NOISY_OVERRIDE_RATE = 0.3

#: Share above which the automation is not "sometimes wrong" but "wrong most
#: of the time it runs" - worth naming separately because the recommended
#: action is stronger (retire, not retune).
OVERRIDDEN_OVERRIDE_RATE = 0.6


@dataclass
class AutomationHealth:
    """One automation's post-deployment verdict, and the evidence behind it."""

    automation_id: str
    title: str
    applied_ts: float
    days_since_applied: float
    status: str = STATUS_UNRESOLVED
    verdict: str = VERDICT_NA
    entity_id: str | None = None
    days_evaluated: float = 0.0
    predicted_fires: int | None = None
    actual_fires: int | None = None
    overrides: int | None = None
    override_rate: float | None = None
    predicted_fires_per_week: float | None = None
    actual_fires_per_week: float | None = None
    #: A plain sentence explaining the verdict. Never blank - every return
    #: path below sets it, so the UI never has a verdict with no reasoning
    #: under it.
    evidence: str = ""
    #: What the user might do about it. Always advisory wording ("retire it",
    #: "consider retuning") - this module recommends, it never acts, and
    #: nothing it returns is a thing the add-on can execute on its own.
    recommendation: str = ""
    #: Caveats that do not change the verdict but should not be hidden either
    #: (e.g. an automation whose YAML could not be read to confirm it is
    #: unchanged).
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "automation_id": self.automation_id,
            "title": self.title,
            "applied_ts": self.applied_ts,
            "days_since_applied": round(self.days_since_applied, 1),
            "status": self.status,
            "verdict": self.verdict,
            "entity_id": self.entity_id,
            "days_evaluated": round(self.days_evaluated, 1),
            "predicted_fires": self.predicted_fires,
            "actual_fires": self.actual_fires,
            "overrides": self.overrides,
            "override_rate": (
                round(self.override_rate, 4) if self.override_rate is not None else None
            ),
            "predicted_fires_per_week": self.predicted_fires_per_week,
            "actual_fires_per_week": self.actual_fires_per_week,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "notes": self.notes,
        }


def _insufficient(health: AutomationHealth, evidence: str, recommendation: str) -> AutomationHealth:
    health.status = STATUS_ACTIVE
    health.verdict = VERDICT_INSUFFICIENT
    health.evidence = evidence
    health.recommendation = recommendation
    return health


def evaluate(
    applied_row: dict[str, Any],
    existing_by_id: dict[str, ExistingAutomation],
    changes: Sequence[StateChange],
    signal_store: SignalStore,
    overrides: Sequence[OverrideEvent],
    options: Options,
    window: tuple[float, float],
) -> AutomationHealth:
    """Judge one applied automation against history gathered since it shipped.

    ``changes`` and ``overrides`` are exactly what one analysis run already
    computed for backtesting and the audit page - this asks nothing new of
    the recorder, it only looks at what was already read.
    """
    automation_id = str(applied_row["automation_id"])
    applied_ts = float(applied_row["applied_ts"])
    days_since_applied = max((window[1] - applied_ts) / 86400.0, 0.0)
    health = AutomationHealth(
        automation_id=automation_id,
        title=str(applied_row.get("title") or automation_id),
        applied_ts=applied_ts,
        days_since_applied=days_since_applied,
    )

    existing = existing_by_id.get(automation_id)
    if existing is None:
        health.status = STATUS_DELETED
        health.verdict = VERDICT_NA
        health.evidence = (
            "No automation with this id exists in Home Assistant any more - it was "
            "deleted, or renamed in a way that changed its id. There is nothing left "
            "to measure."
        )
        health.recommendation = "Nothing to do: it is already gone."
        return health

    health.entity_id = existing.entity_id

    # A "states-only" entry means the YAML this automation came from could not
    # be read (a config-entry-managed or packaged automation) - there is no
    # config to compare against what was shipped, so editing cannot be
    # detected. That is a caveat on everything below, not a reason to treat
    # every such automation as edited.
    verifiable_config = existing.source != "states-only"
    if verifiable_config and not matches(existing.raw, applied_row["shipped_config"]):
        health.status = STATUS_EDITED
        health.verdict = VERDICT_NA
        health.evidence = (
            "This automation's trigger, condition or action has changed since it was "
            "applied, so the rule that was backtested is not the rule running now. "
            "Its performance cannot be judged against evidence gathered for a "
            "different rule."
        )
        health.recommendation = (
            "No recommendation while it differs from what was applied. Apply it again "
            "from a fresh suggestion to start tracking the new version."
        )
        return health
    if not verifiable_config:
        health.notes.append(
            "Its configuration could not be read to confirm it is unchanged; only its "
            "live on/off state is checked."
        )

    if existing.entity_id is None:
        return _insufficient(
            health,
            "Found in your configuration, but Home Assistant reports no live entity "
            "for it, so whether it is firing cannot be measured.",
            "Check back after Home Assistant restarts or the entity registry updates.",
        )

    if not existing.enabled:
        health.status = STATUS_DISABLED
        health.verdict = VERDICT_NA
        health.evidence = "This automation is currently switched off in Home Assistant."
        health.recommendation = "Nothing to do: it cannot fire while disabled."
        return health

    health.status = STATUS_ACTIVE
    start_ts = max(applied_ts, window[0])
    end_ts = max(window[1], start_ts)
    eval_window = (start_ts, end_ts)
    days_evaluated = max((end_ts - start_ts) / 86400.0, 0.0)
    health.days_evaluated = days_evaluated

    if days_evaluated < MIN_DAYS_FOR_VERDICT:
        return _insufficient(
            health,
            f"Only {days_evaluated:.1f} days of history are available since it was "
            f"applied ({days_since_applied:.1f} days ago) - not enough to judge it yet "
            f"(need at least {MIN_DAYS_FOR_VERDICT}).",
            "Check back once it has had more time to run.",
        )

    from .runner import candidate_from_payload  # local: avoids a pipeline/runner cycle

    candidate = candidate_from_payload(applied_row["candidate_payload"])
    fires, _burst, error = backtest_module.simulate_fires_detailed(
        candidate, signal_store, eval_window
    )
    if error:
        return _insufficient(
            health,
            f"Could not replay this rule against your history: {error}.",
            "Check back after the next run.",
        )

    predicted = len(fires)
    entity_id = existing.entity_id
    # Home Assistant writes a new row for the automation entity itself every
    # time it actually runs - last_triggered changes even when the state
    # ("on") does not, so this is not restricted to StateChange.is_transition.
    # This is the automation's own record of firing, independent of whether
    # any downstream entity changed as a result.
    actual_rows = [
        c for c in changes if c.entity_id == entity_id and start_ts <= c.ts <= end_ts
    ]
    actual = len(actual_rows)
    matching_overrides = [
        o for o in overrides if o.automation_entity_id == entity_id and start_ts <= o.ts <= end_ts
    ]
    override_count = len(matching_overrides)
    override_rate = (override_count / actual) if actual else None

    health.predicted_fires = predicted
    health.actual_fires = actual
    health.overrides = override_count
    health.override_rate = override_rate
    health.predicted_fires_per_week = round(predicted * 7.0 / max(days_evaluated, 0.01), 2)
    health.actual_fires_per_week = round(actual * 7.0 / max(days_evaluated, 0.01), 2)

    if actual == 0 and predicted < MIN_PREDICTED_FOR_DORMANT:
        return _insufficient(
            health,
            f"Neither a real firing nor much sign of the pattern it was built from "
            f"showed up in {days_evaluated:.0f} days - too quiet a period to say "
            "anything yet.",
            "Check back after more history has built up.",
        )

    if actual == 0:
        health.verdict = VERDICT_DORMANT
        health.evidence = (
            f"The pattern it was built from would have applied {predicted} times in "
            f"{days_evaluated:.0f} days, but Home Assistant never actually ran it."
        )
        health.recommendation = (
            "Retire it, or check why it is not firing (a changed entity, a failing "
            "condition, or mode: single blocking re-entry) - this add-on will not "
            "change it for you."
        )
        return health

    if override_rate is not None and override_rate >= OVERRIDDEN_OVERRIDE_RATE:
        health.verdict = VERDICT_OVERRIDDEN
        health.evidence = (
            f"Fired {actual} time{'' if actual == 1 else 's'} in {days_evaluated:.0f} days "
            f"and you undid it {override_count} of those times ({override_rate:.0%})."
        )
        health.recommendation = "Retire it - most of what it does is being reversed."
        return health

    if override_rate is not None and override_rate >= NOISY_OVERRIDE_RATE:
        health.verdict = VERDICT_NOISY
        health.evidence = (
            f"Fired {actual} time{'' if actual == 1 else 's'} in {days_evaluated:.0f} days "
            f"and you undid it {override_count} of those times ({override_rate:.0%})."
        )
        health.recommendation = (
            "Consider retuning its conditions - it is right often enough to keep, "
            "wrong enough to notice."
        )
        return health

    health.verdict = VERDICT_HEALTHY
    health.evidence = (
        f"Fired {actual} time{'' if actual == 1 else 's'} in {days_evaluated:.0f} days, "
        + (
            "never overridden."
            if override_count == 0
            else f"overridden only {override_count} time{'' if override_count == 1 else 's'}."
        )
    )
    health.recommendation = "No action needed."
    return health


def evaluate_all(
    store,
    existing: Sequence[ExistingAutomation],
    changes: Sequence[StateChange],
    signal_store: SignalStore,
    overrides: Sequence[OverrideEvent],
    options: Options,
    window: tuple[float, float],
) -> list[AutomationHealth]:
    """Judge every automation this add-on has applied, and persist the result.

    Called once per analysis run, from :mod:`amminer.pipeline`, with exactly
    the ``changes``/``signal_store``/``overrides`` that run already built for
    backtesting - see this module's docstring for why nothing here re-queries
    the recorder.
    """
    # Keyed by id, not entity_id: that id is this add-on's stable marker (see
    # applied_automations' own comment in amminer.store.db), and it is what
    # amminer.automations.load_existing_automations already keys its own
    # entity-id lookup by.
    existing_by_id = {a.id: a for a in existing if a.id}
    results: list[AutomationHealth] = []
    for row in store.list_applied_automations():
        result = evaluate(row, existing_by_id, changes, signal_store, overrides, options, window)
        store.save_automation_health(
            result.automation_id, result.status, result.verdict, result.as_dict()
        )
        results.append(result)
    return results
