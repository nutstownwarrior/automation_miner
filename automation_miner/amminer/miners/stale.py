"""Miner B - stale automations and unused entities.

These are the "clean up your instance" findings.  They are not automation
candidates, so they carry no triggers/actions; the pipeline routes them to the
audit view rather than to the accept/apply flow.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from ..config import ACTIONABLE_DOMAINS, Options
from ..recorderdb.models import Cause, StateChange
from .base import Candidate, Evidence

_LOGGER = logging.getLogger(__name__)


def _parse_last_triggered(value: Any) -> float | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    import datetime as dt

    text = str(value).replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def mine_stale_automations(
    resolver,
    options: Options,
    now: float | None = None,
    override_counts: dict[str, int] | None = None,
) -> list[Candidate]:
    """Automations that have not fired for ``stale_automation_days``."""
    now = now or time.time()
    cutoff = now - options.stale_automation_days * 86400
    override_counts = override_counts or {}
    out: list[Candidate] = []

    for info in resolver.by_domain("automation"):
        if options.is_excluded(info.entity_id):
            continue
        last = _parse_last_triggered(info.attributes.get("last_triggered"))
        state = (info.state or "").lower()
        if state == "off":
            reason = "It is turned off"
            age_days = None
        elif last is None:
            reason = "It has never been triggered"
            age_days = None
        elif last < cutoff:
            age_days = (now - last) / 86400.0
            reason = f"Last triggered {age_days:.0f} days ago"
        else:
            continue

        notes = [reason]
        overrides = override_counts.get(info.entity_id, 0)
        if overrides:
            notes.append(f"You overrode this automation {overrides} times - it may be unwanted.")

        candidate = Candidate(
            miner="stale_automation",
            title=f"Stale automation: {info.name}",
            description=(
                f"{info.name} ({info.entity_id}) looks unused. {reason}. "
                "Review whether it should be removed or repaired."
            ),
            entities=[info.entity_id],
            evidence=Evidence(
                occurrences=0,
                notes=notes,
                extra={
                    "last_triggered": info.attributes.get("last_triggered"),
                    "age_days": round(age_days, 1) if age_days else None,
                    "state": info.state,
                    "override_count": overrides,
                },
            ),
            score=round(min(0.4 + (age_days or 0) / 365.0, 0.9), 4),
        )
        out.append(candidate)

    _LOGGER.info("stale_automation miner produced %d findings", len(out))
    return out


def mine_unused_entities(
    changes: Sequence[StateChange],
    resolver,
    options: Options,
    window: tuple[float, float] | None = None,
) -> list[Candidate]:
    """Controllable entities that no human ever touched in the window."""
    touched: dict[str, int] = defaultdict(int)
    any_change: dict[str, int] = defaultdict(int)
    for change in changes:
        any_change[change.entity_id] += 1
        if change.cause is Cause.HUMAN:
            touched[change.entity_id] += 1

    window_days = ((window[1] - window[0]) / 86400.0) if window else 0.0
    out: list[Candidate] = []
    for info in resolver.entities.values():
        if info.domain not in ACTIONABLE_DOMAINS or info.domain in ("scene", "script"):
            continue
        if options.is_excluded(info.entity_id) or info.disabled or info.hidden:
            continue
        if info.entity_category is not None:  # config/diagnostic entities
            continue
        if touched.get(info.entity_id):
            continue
        out.append(
            Candidate(
                miner="unused_entity",
                title=f"Never used: {info.describe()}",
                description=(
                    f"{info.name} ({info.entity_id}) was never manually controlled in the last "
                    f"{window_days:.0f} days"
                    + (
                        f" (it changed state {any_change[info.entity_id]} times on its own)."
                        if any_change.get(info.entity_id)
                        else " and never changed state at all."
                    )
                ),
                entities=[info.entity_id],
                evidence=Evidence(
                    occurrences=0,
                    window_days=window_days,
                    notes=[
                        "No human-caused state change found in the analysis window.",
                        "Consider removing the device, or automating it so it earns its place.",
                    ],
                    extra={"passive_changes": any_change.get(info.entity_id, 0)},
                ),
                score=0.25,
            )
        )
    _LOGGER.info("unused_entity miner produced %d findings", len(out))
    return out


def mine(
    changes: Sequence[StateChange],
    resolver,
    options: Options,
    window: tuple[float, float] | None = None,
    override_counts: dict[str, int] | None = None,
) -> list[Candidate]:
    now = window[1] if window else None
    return mine_stale_automations(resolver, options, now, override_counts) + mine_unused_entities(
        changes, resolver, options, window
    )
