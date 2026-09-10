"""Read and normalise the user's *existing* automations.

Two sources are merged:

* ``automations.yaml`` (and ``automations/**.yaml``) from the HA config dir -
  the full trigger/condition/action definition,
* the live ``automation.*`` states - which gives ``id``, friendly name,
  ``last_triggered`` and whether it is currently enabled.

The result is a normalised :class:`ExistingAutomation` the conflict checker can
reason about without caring about HA's many YAML shorthands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_LOGGER = logging.getLogger(__name__)

#: Services that drive an entity to a known state.
SERVICE_STATE = {
    "turn_on": "on",
    "turn_off": "off",
    "toggle": None,
    "open_cover": "open",
    "close_cover": "closed",
    "lock": "locked",
    "unlock": "unlocked",
    "media_play": "playing",
    "media_pause": "paused",
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _entity_ids(value: Any) -> list[str]:
    """Pull entity ids out of any of HA's target shorthands."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            out.extend(_entity_ids(item))
    elif isinstance(value, dict):
        for key in ("entity_id", "entity"):
            if key in value:
                out.extend(_entity_ids(value[key]))
        if "target" in value:
            out.extend(_entity_ids(value["target"]))
    return [e for e in out if isinstance(e, str) and "." in e]


@dataclass
class NormalisedAction:
    service: str | None
    entity_ids: list[str] = field(default_factory=list)
    area_ids: list[str] = field(default_factory=list)
    device_ids: list[str] = field(default_factory=list)
    target_state: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExistingAutomation:
    """One automation from the user's config, in neutral form."""

    id: str | None
    alias: str
    entity_id: str | None = None
    enabled: bool = True
    mode: str = "single"
    raw: dict[str, Any] = field(default_factory=dict)
    trigger_entities: list[str] = field(default_factory=list)
    trigger_times: list[str] = field(default_factory=list)
    trigger_kinds: list[str] = field(default_factory=list)
    condition_entities: list[str] = field(default_factory=list)
    actions: list[NormalisedAction] = field(default_factory=list)
    source: str = "yaml"

    @property
    def action_entities(self) -> list[str]:
        out: list[str] = []
        for action in self.actions:
            out.extend(action.entity_ids)
        return sorted(set(out))

    @property
    def called_automations(self) -> list[str]:
        """Automations/scripts this one invokes directly."""
        out: list[str] = []
        for action in self.actions:
            if action.service in ("automation.trigger", "script.turn_on"):
                out.extend(action.entity_ids)
            elif action.service and action.service.startswith("script."):
                out.append(action.service)
        return sorted(set(out))

    def targets(self) -> set[tuple[str, str | None]]:
        return {
            (entity_id, action.target_state)
            for action in self.actions
            for entity_id in action.entity_ids
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "alias": self.alias,
            "entity_id": self.entity_id,
            "enabled": self.enabled,
            "mode": self.mode,
            "trigger_entities": self.trigger_entities,
            "trigger_times": self.trigger_times,
            "trigger_kinds": self.trigger_kinds,
            "condition_entities": self.condition_entities,
            "action_entities": self.action_entities,
            "source": self.source,
        }


def _normalise_action(step: Any) -> list[NormalisedAction]:
    """Flatten one action step (including ``choose`` / ``repeat`` branches)."""
    if not isinstance(step, dict):
        return []
    out: list[NormalisedAction] = []

    service = step.get("service") or step.get("action")
    if isinstance(service, str):
        entity_ids = _entity_ids(step.get("entity_id"))
        entity_ids.extend(_entity_ids(step.get("target")))
        # Home Assistant's original spelling put the target inside data:, and
        # plenty of long-lived configurations still do.  Missing it means the
        # conflict checks see an automation that touches nothing.
        if isinstance(step.get("data"), dict):
            entity_ids.extend(_entity_ids(step["data"].get("entity_id")))
        target = step.get("target") if isinstance(step.get("target"), dict) else {}
        area_ids = [a for a in _as_list(target.get("area_id")) if isinstance(a, str)]
        device_ids = [d for d in _as_list(target.get("device_id")) if isinstance(d, str)]
        if isinstance(step.get("data"), dict):
            area_ids.extend(
                a for a in _as_list(step["data"].get("area_id")) if isinstance(a, str)
            )
            device_ids.extend(
                d for d in _as_list(step["data"].get("device_id")) if isinstance(d, str)
            )
        data = step.get("data") if isinstance(step.get("data"), dict) else {}
        state = SERVICE_STATE.get(service.split(".", 1)[-1])
        if state is None and "hvac_mode" in data:
            state = str(data["hvac_mode"])
        if state is None and "option" in data:
            state = str(data["option"])
        out.append(
            NormalisedAction(
                service=service,
                entity_ids=sorted(set(entity_ids)),
                area_ids=area_ids,
                device_ids=device_ids,
                target_state=state,
                data=dict(data),
            )
        )

    if "scene" in step:
        out.append(NormalisedAction(service="scene.turn_on", entity_ids=_entity_ids(step["scene"])))
    for branch_key in ("choose", "then", "else", "default", "sequence", "actions"):
        branch = step.get(branch_key)
        if isinstance(branch, list):
            for item in branch:
                if isinstance(item, dict) and "sequence" in item:
                    for sub in _as_list(item["sequence"]):
                        out.extend(_normalise_action(sub))
                else:
                    out.extend(_normalise_action(item))
    if isinstance(step.get("repeat"), dict):
        for sub in _as_list(step["repeat"].get("sequence")):
            out.extend(_normalise_action(sub))
    return out


def normalise_automation(raw: dict[str, Any], source: str = "yaml") -> ExistingAutomation:
    """Turn one raw automation dict into the neutral shape."""
    triggers = _as_list(raw.get("trigger") or raw.get("triggers"))
    conditions = _as_list(raw.get("condition") or raw.get("conditions"))
    actions = _as_list(raw.get("action") or raw.get("actions"))

    trigger_entities: list[str] = []
    trigger_times: list[str] = []
    trigger_kinds: list[str] = []
    for trigger in triggers:
        if not isinstance(trigger, dict):
            continue
        kind = str(trigger.get("platform") or trigger.get("trigger") or "")
        trigger_kinds.append(kind)
        trigger_entities.extend(_entity_ids(trigger.get("entity_id")))
        if kind == "time":
            trigger_times.extend(str(v) for v in _as_list(trigger.get("at")))

    condition_entities: list[str] = []
    for condition in conditions:
        if isinstance(condition, dict):
            condition_entities.extend(_entity_ids(condition.get("entity_id")))

    normalised_actions: list[NormalisedAction] = []
    for action in actions:
        normalised_actions.extend(_normalise_action(action))

    return ExistingAutomation(
        id=str(raw["id"]) if raw.get("id") is not None else None,
        alias=str(raw.get("alias") or raw.get("id") or "unnamed automation"),
        mode=str(raw.get("mode") or "single"),
        raw=raw,
        trigger_entities=sorted(set(trigger_entities)),
        trigger_times=trigger_times,
        trigger_kinds=trigger_kinds,
        condition_entities=sorted(set(condition_entities)),
        actions=normalised_actions,
        source=source,
    )


def load_existing_automations(ha_config, resolver=None) -> list[ExistingAutomation]:
    """Read every automation we can find, and attach live entity ids."""
    from .discovery.ha_config import load_yaml_file

    found: list[ExistingAutomation] = []
    if ha_config is not None:
        for path in ha_config.automation_files():
            data = load_yaml_file(path, ha_config.config_dir, ha_config.secrets)
            for raw in _as_list(data):
                if isinstance(raw, dict) and (raw.get("trigger") or raw.get("triggers")):
                    found.append(normalise_automation(raw, source=path.name))
        inline = ha_config.config.get("automation")
        for raw in _as_list(inline):
            if isinstance(raw, dict) and (raw.get("trigger") or raw.get("triggers")):
                found.append(normalise_automation(raw, source="configuration.yaml"))

    # Attach the live entity_id / enabled flag by matching the automation's
    # unique id, which HA exposes as the entity's ``id`` attribute.
    if resolver is not None:
        by_id: dict[str, Any] = {}
        for info in resolver.by_domain("automation"):
            unique = info.attributes.get("id") or info.unique_id
            if unique:
                by_id[str(unique)] = info
        matched: set[str] = set()
        for automation in found:
            info = by_id.get(automation.id or "")
            if info is not None:
                automation.entity_id = info.entity_id
                automation.enabled = (info.state or "on").lower() != "off"
                matched.add(info.entity_id)
        # Automations that exist only as entities (UI-created but unreadable
        # config, or from a package) still matter for conflict checks.
        for info in resolver.by_domain("automation"):
            if info.entity_id in matched:
                continue
            found.append(
                ExistingAutomation(
                    id=str(info.attributes.get("id") or ""),
                    alias=info.name,
                    entity_id=info.entity_id,
                    enabled=(info.state or "on").lower() != "off",
                    source="states-only",
                )
            )

    _LOGGER.info("Loaded %d existing automations", len(found))
    return found
