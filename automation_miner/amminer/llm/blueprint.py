"""Deterministic candidate -> Home Assistant automation rendering.

This is the ``llm_provider: none`` path, and it is *not* a lesser fallback: the
internal schema maps cleanly onto HA triggers/conditions/actions, so a
deterministic renderer produces correct YAML every time.  The LLM path exists to
produce nicer aliases and to handle shapes the renderer cannot express - and its
output goes through exactly the same validation gate as this one.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from ..miners.base import Action, Candidate, Condition, Trigger

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:60] or "automation_miner_rule"


def automation_id(candidate: Candidate) -> str:
    """Stable id so re-applying a suggestion updates rather than duplicates."""
    return f"amminer_{candidate.id}"


def trigger_to_ha(trigger: Trigger) -> dict[str, Any]:
    if trigger.kind == "time":
        return {"platform": "time", "at": trigger.at}
    if trigger.kind == "sun":
        payload: dict[str, Any] = {"platform": "sun", "event": trigger.event or "sunset"}
        if trigger.offset:
            payload["offset"] = trigger.offset
        return payload
    if trigger.kind == "state":
        payload = {"platform": "state", "entity_id": trigger.entity_id}
        if trigger.to_state is not None:
            payload["to"] = trigger.to_state
        if trigger.from_state is not None:
            payload["from"] = trigger.from_state
        if trigger.for_seconds:
            payload["for"] = {"seconds": int(trigger.for_seconds)}
        return payload
    if trigger.kind == "numeric_state":
        payload = {"platform": "numeric_state", "entity_id": trigger.entity_id}
        if trigger.above is not None:
            payload["above"] = trigger.above
        if trigger.below is not None:
            payload["below"] = trigger.below
        return payload
    if trigger.kind == "time_pattern":
        return {"platform": "time_pattern", "minutes": trigger.at or "/30"}
    raise ValueError(f"Cannot render trigger kind '{trigger.kind}'")


def condition_to_ha(condition: Condition) -> dict[str, Any]:
    if condition.kind == "time":
        payload: dict[str, Any] = {"condition": "time"}
        if condition.after:
            payload["after"] = condition.after
        if condition.before:
            payload["before"] = condition.before
        if condition.weekday:
            payload["weekday"] = [d for d in condition.weekday if d in WEEKDAYS]
        return payload
    if condition.kind == "state":
        return {
            "condition": "state",
            "entity_id": condition.entity_id,
            "state": condition.state,
        }
    if condition.kind == "numeric_state":
        payload = {"condition": "numeric_state", "entity_id": condition.entity_id}
        if condition.above is not None:
            payload["above"] = condition.above
        if condition.below is not None:
            payload["below"] = condition.below
        return payload
    if condition.kind == "sun":
        payload = {"condition": "sun"}
        if condition.after:
            payload["after"] = condition.after
        if condition.before:
            payload["before"] = condition.before
        return payload
    raise ValueError(f"Cannot render condition kind '{condition.kind}'")


def action_to_ha(action: Action) -> dict[str, Any]:
    payload: dict[str, Any] = {"service": action.service}
    if action.entity_id:
        payload["target"] = {"entity_id": action.entity_id}
    if action.data:
        payload["data"] = dict(action.data)
    return payload


def candidate_to_automation(
    candidate: Candidate, resolver=None, alias: str | None = None
) -> dict[str, Any]:
    """Render a candidate as a Home Assistant automation config dict."""
    if not candidate.triggers:
        raise ValueError("candidate has no trigger")
    if not candidate.actions:
        raise ValueError("candidate has no action")

    description_lines = [candidate.describe(resolver)]
    if candidate.evidence.summary():
        description_lines.append(f"Evidence: {candidate.evidence.summary()}")
    if candidate.backtest and candidate.backtest.get("summary"):
        description_lines.append(f"Backtest: {candidate.backtest['summary']}")
    description_lines.append("Suggested by Automation Miner.")

    return {
        "id": automation_id(candidate),
        "alias": alias or candidate.title,
        "description": " ".join(description_lines),
        "mode": "single",
        "trigger": [trigger_to_ha(t) for t in candidate.triggers],
        "condition": [condition_to_ha(c) for c in candidate.conditions],
        "action": [action_to_ha(a) for a in candidate.actions],
    }


def render_yaml(config: dict[str, Any]) -> str:
    """Serialise an automation config the way HA's own editor would."""
    return yaml.safe_dump(
        [config], sort_keys=False, default_flow_style=False, allow_unicode=True
    )


def render_candidate_yaml(candidate: Candidate, resolver=None) -> str:
    return render_yaml(candidate_to_automation(candidate, resolver))
