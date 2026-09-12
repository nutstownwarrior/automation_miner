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
``amminer.recorderdb.queries`` (the ``automation_triggered`` events it already
    loads for :mod:`amminer.recorderdb.causality`)
    how many times the automation really ran is counted directly from these -
    not from the automation entity's own recorded state. Home Assistant
    re-asserts every enabled automation's state as "on" on every restart,
    with a fresh context and no trigger behind it; counting that as a firing
    would report a burst of activity every time Home Assistant restarts,
    which is the opposite of what this module exists to be honest about.
``amminer.recorderdb.causality.detect_overrides``
    already run once per analysis, over the same window; this module only
    filters that list down to the one automation being judged.

Nothing here writes to Home Assistant, changes a suggestion, or disables
anything. The worst an unhealthy verdict does is put a sentence of
recommended action in front of the user - the retiring or retuning stays
theirs to do.

Two honesty rules this module holds itself to, beyond the evidence floors
documented on the Options fields below:

* An ``automation_triggered`` event fires whether the automation's own
  trigger matched or a person (or another automation) called
  ``automation.trigger`` by hand. Both count as "it ran" here - a manual
  trigger is still real evidence about whether the rule does something
  useful when it runs - but it means ``actual_fires`` answers "how many
  times did this run", not "how many times did its configured trigger
  condition genuinely match".
* A single automation's health check failing (a corrupted or pre-migration
  snapshot, most plausibly) must cost only that automation's verdict, never
  every other one's - see :func:`evaluate_all`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import backtest as backtest_module
from .automations import ExistingAutomation
from .config import Options
from .enrich.signals import SignalStore
from .llm.equivalence import matches
from .recorderdb.models import OverrideEvent, RecorderEvent

_LOGGER = logging.getLogger(__name__)

STATUS_ACTIVE = "active"
STATUS_DELETED = "deleted"
STATUS_EDITED = "edited"
STATUS_DISABLED = "disabled"
STATUS_UNRESOLVED = "unresolved"

VERDICT_HEALTHY = "healthy"
VERDICT_NOISY = "noisy"
VERDICT_DORMANT = "dormant"
VERDICT_OVERRIDDEN = "overridden"
#: Still running, but firing far less often than the pattern it was built
#: from says it should - distinct from dormant (zero real fires): this one
#: fires occasionally, just not nearly as often as it should be able to.
VERDICT_UNDERPERFORMING = "underperforming"
VERDICT_INSUFFICIENT = "insufficient_data"
#: Not a judgement on the rule's performance at all - shown instead of one of
#: the verdicts above whenever there is nothing to measure yet (the structural
#: statuses: deleted, edited, disabled, unresolved) or the check itself failed
#: (status error).
VERDICT_NA = "n/a"

STATUS_ERROR = "error"

# The actual thresholds (how many days, how few predicted or actual fires,
# which override rates and shortfall ratio) are configurable Options - see
# config.py's health_min_days / health_min_predicted_for_dormant /
# health_min_fires_for_verdict / health_noisy_override_rate /
# health_overridden_override_rate / health_shortfall_ratio and their own
# docstrings for the reasoning. They live there, not as module constants
# here, for the same reason backtest_min_true_fires lives in Options rather
# than in backtest.py: a threshold a user might reasonably want to loosen or
# tighten belongs with the rest of the add-on's configuration.


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
    events: Sequence[RecorderEvent],
    signal_store: SignalStore,
    overrides: Sequence[OverrideEvent],
    options: Options,
    window: tuple[float, float],
) -> AutomationHealth:
    """Judge one applied automation against history gathered since it shipped.

    ``events`` and ``overrides`` are exactly what one analysis run already
    loaded for causality classification and the audit page - this asks
    nothing new of the recorder, it only looks at what was already read.
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

    if days_evaluated < options.health_min_days:
        return _insufficient(
            health,
            f"Only {days_evaluated:.1f} days of history are available since it was "
            f"applied ({days_since_applied:.1f} days ago) - not enough to judge it yet "
            f"(need at least {options.health_min_days}).",
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
    # "Did it really fire" is answered from Home Assistant's own
    # automation_triggered events, not from the automation entity's recorded
    # state: every enabled automation gets restated as "on" (a fresh context,
    # no trigger behind it) on every Home Assistant restart, so counting that
    # as a firing would report a burst of activity every time the box
    # reboots - see this module's docstring.
    actual = sum(
        1
        for event in events
        if event.event_type == "automation_triggered"
        and event.entity_id == entity_id
        and start_ts <= event.ts <= end_ts
    )
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

    if actual == 0 and predicted < options.health_min_predicted_for_dormant:
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

    # A handful of real fires is not enough to trust a RATIO over them: one
    # fire and one revert is override_rate=1.0 and a single observation with
    # a percentage's name on it. Below this floor, every verdict that reads
    # override_rate - healthy included, not only the alarming ones - would be
    # confidence this module has no right to. Dormancy above needed no such
    # floor because it asks a different question (did it fire *at all*), and
    # underperformance and override verdicts below both need it because both
    # are read straight off actual.
    if actual < options.health_min_fires_for_verdict:
        return _insufficient(
            health,
            f"Only fired {actual} time{'' if actual == 1 else 's'} in "
            f"{days_evaluated:.0f} days - too few real fires yet to judge reliably "
            f"(need at least {options.health_min_fires_for_verdict}).",
            "Check back once it has fired more.",
        )

    if override_rate is not None and override_rate >= options.health_overridden_override_rate:
        health.verdict = VERDICT_OVERRIDDEN
        health.evidence = (
            f"Fired {actual} time{'' if actual == 1 else 's'} in {days_evaluated:.0f} days "
            f"and you undid it {override_count} of those times ({override_rate:.0%})."
        )
        health.recommendation = "Retire it - most of what it does is being reversed."
        return health

    if override_rate is not None and override_rate >= options.health_noisy_override_rate:
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

    # override_rate alone cannot see this: an automation nobody is undoing is
    # not thereby healthy if it has almost stopped happening. Dormant is the
    # actual==0 case; this is the case where it still fires sometimes but the
    # pattern it was built from says it should be firing far more - the
    # predicted-vs-actual divergence this module computes but, without this
    # check, never actually used for anything.
    #
    # No separate "enough predicted fires to trust the ratio" floor is needed
    # here the way dormancy has one: actual has already cleared
    # health_min_fires_for_verdict above (>= 1, and by default 5), so a
    # shortfall ratio below health_shortfall_ratio (by default 0.3) already
    # implies predicted is at least actual / health_shortfall_ratio - several
    # times the floor dormancy itself needs to trust a prediction. A small
    # predicted count simply cannot produce a false shortfall here.
    if predicted > 0:
        fire_ratio = actual / predicted
        if fire_ratio < options.health_shortfall_ratio:
            health.verdict = VERDICT_UNDERPERFORMING
            health.evidence = (
                f"The pattern it was built from would have applied {predicted} times "
                f"in {days_evaluated:.0f} days; it actually ran only {actual} of those "
                f"({fire_ratio:.0%})."
            )
            health.recommendation = (
                "Consider retiring or retuning it - it has largely stopped doing what "
                "it was built for, even though it still fires occasionally."
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


def _error_result(applied_row: dict[str, Any], err: Exception) -> AutomationHealth:
    """A row this run could not judge - a bug or a corrupted/pre-migration
    snapshot, most plausibly - reported as exactly that, not folded into
    "not enough data" (a different claim, wrong here) or silently dropped.

    ``applied_row``'s own columns (automation_id, title, applied_ts) come
    straight from NOT NULL database columns, so reading them cannot be what
    raised - it is always the arbitrary, unversioned JSON in
    candidate_payload/shipped_config that can. Falling back to the id itself
    for anything that still cannot be read keeps this from raising a second
    time while building the very report meant to explain the first failure.
    """
    automation_id = str(applied_row.get("automation_id") or "unknown")
    return AutomationHealth(
        automation_id=automation_id,
        title=str(applied_row.get("title") or automation_id),
        applied_ts=float(applied_row.get("applied_ts") or 0.0),
        days_since_applied=0.0,
        status=STATUS_ERROR,
        verdict=VERDICT_NA,
        evidence=f"Could not check this automation's health this run: {type(err).__name__}: {err}",
        recommendation=(
            "No recommendation - this looks like a bug in the health check itself, "
            "not a judgement about the automation. Check back after the next run."
        ),
    )


def evaluate_all(
    store,
    existing: Sequence[ExistingAutomation],
    events: Sequence[RecorderEvent],
    signal_store: SignalStore,
    overrides: Sequence[OverrideEvent],
    options: Options,
    window: tuple[float, float],
) -> list[AutomationHealth]:
    """Judge every automation this add-on has applied, and persist the result.

    Called once per analysis run, from :mod:`amminer.pipeline`, with exactly
    the ``events``/``signal_store``/``overrides`` that run already loaded for
    causality classification and backtesting - see this module's docstring
    for why nothing here re-queries the recorder.
    """
    # Keyed by id, not entity_id: that id is this add-on's stable marker (see
    # applied_automations' own comment in amminer.store.db), and it is what
    # amminer.automations.load_existing_automations already keys its own
    # entity-id lookup by.
    existing_by_id = {a.id: a for a in existing if a.id}
    results: list[AutomationHealth] = []
    for row in store.list_applied_automations():
        try:
            result = evaluate(row, existing_by_id, events, signal_store, overrides, options, window)
        except Exception as err:  # noqa: BLE001 - one bad row must not blank every other verdict
            _LOGGER.exception(
                "Health check failed for %s: %s", row.get("automation_id"), err
            )
            result = _error_result(row, err)
        store.save_automation_health(
            result.automation_id, result.status, result.verdict, result.as_dict()
        )
        results.append(result)
    return results
