"""Internal event schema shared by every miner.

Public datasets (CASAS, ARAS, Kasteren) and synthetic fixtures are mapped onto
exactly these types, so the miners never see a recorder-specific shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Cause(str, Enum):  # noqa: UP042 - str mixin keeps JSON round-tripping simple
    """Who caused a state change."""

    HUMAN = "human"
    AUTOMATION = "automation"
    SCRIPT = "script"
    DEVICE = "device"
    UNKNOWN = "unknown"

    @property
    def is_human(self) -> bool:
        return self is Cause.HUMAN

    @property
    def is_automated(self) -> bool:
        return self in (Cause.AUTOMATION, Cause.SCRIPT)


@dataclass(slots=True)
class StateChange:
    """One row of ``states``, already joined and classified."""

    entity_id: str
    state: str
    ts: float
    old_state: str | None = None
    last_changed_ts: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    context_id: str | None = None
    context_user_id: str | None = None
    context_parent_id: str | None = None
    cause: Cause = Cause.UNKNOWN
    #: entity_id of the automation/script at the root of the context chain
    origin_entity_id: str | None = None
    #: how many parent hops we walked to find the origin
    chain_depth: int = 0

    @property
    def domain(self) -> str:
        return self.entity_id.split(".", 1)[0] if "." in self.entity_id else ""

    @property
    def is_transition(self) -> bool:
        """True when the state value actually changed (not just attributes)."""
        return self.old_state is not None and self.old_state != self.state

    @property
    def numeric(self) -> float | None:
        try:
            return float(self.state)
        except (TypeError, ValueError):
            return None


@dataclass(slots=True)
class RecorderEvent:
    """One row of ``events`` (``automation_triggered``, ``script_started`` ...)."""

    event_type: str
    ts: float
    data: dict[str, Any] = field(default_factory=dict)
    context_id: str | None = None
    context_user_id: str | None = None
    context_parent_id: str | None = None

    @property
    def entity_id(self) -> str | None:
        value = self.data.get("entity_id")
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
        return None


@dataclass(slots=True)
class OverrideEvent:
    """A human contradicting an automation shortly after it acted."""

    entity_id: str
    ts: float
    automation_entity_id: str | None
    automation_state: str
    human_state: str
    delay_seconds: float
    context_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "ts": self.ts,
            "automation_entity_id": self.automation_entity_id,
            "automation_state": self.automation_state,
            "human_state": self.human_state,
            "delay_seconds": round(self.delay_seconds, 1),
        }
