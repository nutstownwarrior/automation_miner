"""The internal, neutral candidate schema.

Every miner emits the *same* shape: triggers, conditions, actions plus the
evidence that justified them.  Nothing downstream (backtester, conflict checker,
LLM prompt, blueprint renderer, UI) knows which miner produced a candidate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Trigger:
    """A neutral trigger.

    ``kind`` is one of ``time``, ``state``, ``numeric_state``, ``sun``,
    ``time_pattern`` - a deliberately small subset that maps 1:1 onto Home
    Assistant triggers and is fully simulatable by the backtester.
    """

    kind: str
    entity_id: str | None = None
    to_state: str | None = None
    from_state: str | None = None
    at: str | None = None  # "HH:MM:SS" for kind == "time"
    above: float | None = None
    below: float | None = None
    offset: str | None = None  # for kind == "sun"
    event: str | None = None  # "sunset" / "sunrise"
    for_seconds: int | None = None

    def describe(self, resolver=None) -> str:
        name = self.entity_id or ""
        if resolver is not None and self.entity_id:
            name = resolver.name_of(self.entity_id)
        if self.kind == "time":
            return f"at {str(self.at)[:5]}"
        if self.kind == "sun":
            offset = f" {self.offset}" if self.offset else ""
            return f"at {self.event}{offset}"
        if self.kind == "state":
            target = f" becomes '{self.to_state}'" if self.to_state else " changes"
            hold = f" for {self.for_seconds}s" if self.for_seconds else ""
            return f"{name}{target}{hold}"
        if self.kind == "numeric_state":
            bounds = []
            if self.above is not None:
                bounds.append(f"above {self.above:g}")
            if self.below is not None:
                bounds.append(f"below {self.below:g}")
            return f"{name} goes {' and '.join(bounds)}"
        return f"{self.kind} {name}".strip()

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Condition:
    """A neutral condition (``time``, ``state``, ``numeric_state``, ``sun``)."""

    kind: str
    entity_id: str | None = None
    state: str | None = None
    above: float | None = None
    below: float | None = None
    weekday: list[str] = field(default_factory=list)
    after: str | None = None
    before: str | None = None
    #: Free-text note explaining where the condition came from.
    source: str | None = None

    def describe(self, resolver=None) -> str:
        name = self.entity_id or ""
        if resolver is not None and self.entity_id:
            name = resolver.name_of(self.entity_id)
        if self.kind == "time":
            bits = []
            if self.after:
                bits.append(f"after {self.after}")
            if self.before:
                bits.append(f"before {self.before}")
            if self.weekday:
                bits.append("on " + ", ".join(self.weekday))
            return " and ".join(bits) or "any time"
        if self.kind == "state":
            return f"{name} is '{self.state}'"
        if self.kind == "numeric_state":
            bounds = []
            if self.above is not None:
                bounds.append(f"above {self.above:g}")
            if self.below is not None:
                bounds.append(f"below {self.below:g}")
            return f"{name} is {' and '.join(bounds)}"
        if self.kind == "sun":
            return f"sun {self.after or self.before or ''}".strip()
        return self.kind

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")}


@dataclass
class Action:
    """A neutral action - always a service call against an entity."""

    service: str  # "light.turn_on"
    entity_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return self.service.split(".", 1)[0]

    @property
    def target_state(self) -> str | None:
        """The state this action is expected to produce, when knowable."""
        service = self.service.split(".", 1)[-1]
        mapping = {
            "turn_on": "on",
            "turn_off": "off",
            "open_cover": "open",
            "close_cover": "closed",
            "lock": "locked",
            "unlock": "unlocked",
        }
        return mapping.get(service)

    def describe(self, resolver=None) -> str:
        name = self.entity_id or ""
        if resolver is not None and self.entity_id:
            name = resolver.name_of(self.entity_id)
        verb = self.service.split(".", 1)[-1].replace("_", " ")
        extra = ""
        if self.data:
            pairs = ", ".join(f"{k}={v}" for k, v in sorted(self.data.items()))
            extra = f" ({pairs})"
        return f"{verb} {name}{extra}".strip()

    def as_dict(self) -> dict[str, Any]:
        data = {"service": self.service}
        if self.entity_id:
            data["entity_id"] = self.entity_id
        if self.data:
            data["data"] = self.data
        return data


@dataclass
class Evidence:
    """Why we believe a candidate is real."""

    occurrences: int = 0
    opportunities: int = 0
    consistency: float | None = None
    support: float | None = None
    confidence: float | None = None
    lift: float | None = None
    spread_minutes: float | None = None
    window_start_ts: float | None = None
    window_end_ts: float | None = None
    window_days: float | None = None
    #: Sample of the raw timestamps that produced the pattern (for the UI).
    samples: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        bits = []
        if self.occurrences and self.opportunities:
            bits.append(f"{self.occurrences} of {self.opportunities}")
        elif self.occurrences:
            bits.append(f"{self.occurrences} occurrences")
        if self.consistency is not None:
            bits.append(f"consistency {self.consistency:.0%}")
        if self.confidence is not None:
            bits.append(f"confidence {self.confidence:.0%}")
        if self.lift is not None:
            bits.append(f"lift {self.lift:.2f}")
        if self.spread_minutes is not None:
            bits.append(f"±{self.spread_minutes:.0f} min")
        return ", ".join(bits)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["samples"] = self.samples[:50]
        data["summary"] = self.summary()
        return data


@dataclass
class Candidate:
    """A mined automation candidate, before backtesting and conflict checks."""

    miner: str
    title: str
    triggers: list[Trigger] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    evidence: Evidence = field(default_factory=Evidence)
    score: float = 0.0
    description: str = ""
    #: Every entity the candidate touches - used for conflict analysis.
    entities: list[str] = field(default_factory=list)
    #: Optional per-candidate extras (association rule sets, motif indices, ...)
    extra: dict[str, Any] = field(default_factory=dict)
    #: Filled in later by the pipeline.
    backtest: dict[str, Any] | None = None
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.entities:
            touched: list[str] = []
            for trigger in self.triggers:
                if trigger.entity_id:
                    touched.append(trigger.entity_id)
            for condition in self.conditions:
                if condition.entity_id:
                    touched.append(condition.entity_id)
            for action in self.actions:
                if action.entity_id:
                    touched.append(action.entity_id)
            self.entities = sorted(set(touched))

    # ------------------------------------------------------------------
    @property
    def signature(self) -> str:
        """Stable identity of *what the rule does*, ignoring evidence.

        Two runs that rediscover the same habit produce the same signature, so
        dismissals stick across runs and re-mined candidates are not duplicated.
        """
        return _stable_id(
            self.miner,
            [t.as_dict() for t in self.triggers],
            [c.as_dict() for c in self.conditions],
            [a.as_dict() for a in self.actions],
        )

    @property
    def id(self) -> str:
        return self.signature

    @property
    def target_entities(self) -> list[str]:
        return sorted({a.entity_id for a in self.actions if a.entity_id})

    @property
    def trigger_entities(self) -> list[str]:
        return sorted({t.entity_id for t in self.triggers if t.entity_id})

    def describe(self, resolver=None) -> str:
        trigger_text = " or ".join(t.describe(resolver) for t in self.triggers) or "always"
        action_text = " and ".join(a.describe(resolver) for a in self.actions)
        # "at 06:30" already reads as a clause; "person.alex becomes home" needs
        # a "When" in front of it.
        lead = "" if trigger_text.startswith("at ") else "When "
        text = f"{lead}{trigger_text}, {action_text}".strip()
        text = text[0].upper() + text[1:] if text else text
        if self.conditions:
            condition_text = " and ".join(c.describe(resolver) for c in self.conditions)
            text += f" - but only if {condition_text}"
        return text

    def as_dict(self, resolver=None) -> dict[str, Any]:
        return {
            "id": self.id,
            "miner": self.miner,
            "title": self.title,
            "description": self.description or self.describe(resolver),
            "sentence": self.describe(resolver),
            "score": round(self.score, 4),
            "triggers": [t.as_dict() for t in self.triggers],
            "conditions": [c.as_dict() for c in self.conditions],
            "actions": [a.as_dict() for a in self.actions],
            "trigger_text": [t.describe(resolver) for t in self.triggers],
            "condition_text": [c.describe(resolver) for c in self.conditions],
            "action_text": [a.describe(resolver) for a in self.actions],
            "evidence": self.evidence.as_dict(),
            "entities": self.entities,
            "entity_names": (
                {e: resolver.describe(e) for e in self.entities} if resolver else {}
            ),
            "backtest": self.backtest,
            "conflicts": self.conflicts,
            "extra": self.extra,
        }
