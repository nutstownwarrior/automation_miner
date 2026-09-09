"""Applying a suggestion - always an explicit user action, never automatic."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .llm.generate import GenerationResult

_LOGGER = logging.getLogger(__name__)


@dataclass
class ApplyResult:
    ok: bool = False
    written: bool = False
    reloaded: bool = False
    automation_id: str | None = None
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "written": self.written,
            "reloaded": self.reloaded,
            "automation_id": self.automation_id,
            "errors": self.errors,
            "notes": self.notes,
        }


def apply_generation(generation: GenerationResult, client) -> ApplyResult:
    """Write a validated automation to HA and reload.

    Refuses outright when the validation gate did not pass - a rule that failed
    validation must never reach the user's config, regardless of how it was
    generated.
    """
    result = ApplyResult()
    if not generation.ok or not generation.config:
        result.errors.append(
            "refusing to apply: the automation did not pass validation "
            + ("; ".join(generation.report.errors if generation.report else [])[:300])
        )
        return result
    if client is None or not client.configured:
        result.errors.append(
            "no Home Assistant API access (SUPERVISOR_TOKEN missing); copy the YAML manually"
        )
        return result

    config = dict(generation.config)
    automation_id = str(config.get("id") or "")
    if not automation_id:
        result.errors.append("generated automation has no id")
        return result
    result.automation_id = automation_id
    # HA's config API takes the automation body; the id travels in the URL.
    body = {k: v for k, v in config.items() if k != "id"}

    if not client.upsert_automation(automation_id, body):
        result.errors.append(f"writing the automation failed: {client.last_error}")
        return result
    result.written = True

    if client.reload_automations():
        result.reloaded = True
    else:
        result.notes.append(
            f"automation written but automation.reload failed ({client.last_error}); "
            "reload manually from Developer Tools"
        )

    result.ok = result.written
    _LOGGER.info("Applied automation %s (reloaded=%s)", automation_id, result.reloaded)
    return result
