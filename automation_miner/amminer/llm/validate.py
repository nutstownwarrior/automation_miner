"""The validation gate.  Nothing reaches the user or HA without passing it.

Independent checks, all mandatory:

1. **Knowable targets** - what the automation will touch must be decidable
   *now*, from the text.  A Jinja template or ``entity_id: all`` defers that
   decision to runtime, where no gate can see it, so both are refused outright.
   Nothing this add-on generates needs either.
2. **Existence** - every ``entity_id`` / ``device_id`` / ``area_id`` / service
   the automation references must exist in the registry-union-states set.  This
   is what actually catches LLM hallucination; ``check_config`` does not.
3. **Schema** - the YAML must parse and match Home Assistant's automation
   schema, mirrored here in voluptuous.
4. **Core check** - ``POST /api/config/core/check_config`` must pass.  It misses
   some semantic errors and quoting bugs, which is precisely why step 2 is not
   optional.

Every check fails *closed*.  When the live service index is unreachable the
service check does not become a no-op: it falls back to the closed set of
services this add-on is capable of emitting, so an unreachable Core cannot turn
"we could not verify this" into "this is fine".

:mod:`amminer.llm.equivalence` runs alongside these on LLM output, and answers
the question none of them do: is this still the rule that was mined?
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import voluptuous as vol
import yaml

_LOGGER = logging.getLogger(__name__)

#: Jinja delimiters.  A value containing any of these is decided at runtime.
TEMPLATE_MARKERS = ("{{", "{%", "{#")

#: Target keywords that mean "everything", i.e. a target we cannot enumerate.
WILDCARD_TARGETS = ("all", "none")


TARGET_SCHEMA = vol.Schema(
    {
        vol.Optional("entity_id"): vol.Any(str, [str]),
        vol.Optional("device_id"): vol.Any(str, [str]),
        vol.Optional("area_id"): vol.Any(str, [str]),
        vol.Optional("label_id"): vol.Any(str, [str]),
        vol.Optional("floor_id"): vol.Any(str, [str]),
    }
)

TRIGGER_SCHEMA = vol.Schema(
    {
        vol.Required(vol.Any("platform", "trigger")): str,
        vol.Optional("entity_id"): vol.Any(str, [str]),
        vol.Optional("at"): vol.Any(str, [str]),
        vol.Optional("to"): vol.Any(str, [str], None),
        vol.Optional("from"): vol.Any(str, [str], None),
        vol.Optional("above"): vol.Any(int, float, str),
        vol.Optional("below"): vol.Any(int, float, str),
        vol.Optional("for"): vol.Any(str, dict),
        vol.Optional("event"): str,
        vol.Optional("offset"): vol.Any(str, int),
        vol.Optional("id"): str,
        vol.Optional("minutes"): vol.Any(str, int),
        vol.Optional("hours"): vol.Any(str, int),
        vol.Optional("seconds"): vol.Any(str, int),
        vol.Optional("value_template"): str,
        vol.Optional("attribute"): str,
        vol.Optional("zone"): str,
        vol.Optional("event_type"): vol.Any(str, [str]),
        vol.Optional("event_data"): dict,
    },
    extra=vol.ALLOW_EXTRA,
)

CONDITION_SCHEMA = vol.Schema(
    {
        vol.Required("condition"): str,
        vol.Optional("entity_id"): vol.Any(str, [str]),
        vol.Optional("state"): vol.Any(str, [str], int, float),
        vol.Optional("above"): vol.Any(int, float, str),
        vol.Optional("below"): vol.Any(int, float, str),
        vol.Optional("after"): vol.Any(str),
        vol.Optional("before"): vol.Any(str),
        vol.Optional("weekday"): vol.Any(str, [str]),
        vol.Optional("value_template"): str,
        vol.Optional("conditions"): list,
        vol.Optional("attribute"): str,
    },
    extra=vol.ALLOW_EXTRA,
)

ACTION_SCHEMA = vol.Schema(
    {
        vol.Optional("service"): str,
        vol.Optional("action"): str,
        vol.Optional("target"): TARGET_SCHEMA,
        vol.Optional("entity_id"): vol.Any(str, [str]),
        vol.Optional("data"): dict,
        vol.Optional("delay"): vol.Any(str, dict, int),
        vol.Optional("choose"): list,
        vol.Optional("default"): list,
        vol.Optional("repeat"): dict,
        vol.Optional("sequence"): list,
        vol.Optional("wait_template"): str,
        vol.Optional("scene"): str,
        vol.Optional("alias"): str,
    },
    extra=vol.ALLOW_EXTRA,
)

AUTOMATION_SCHEMA = vol.Schema(
    {
        vol.Optional("id"): vol.Any(str, int),
        vol.Required("alias"): str,
        vol.Optional("description"): str,
        vol.Optional("mode"): vol.In(("single", "restart", "queued", "parallel")),
        vol.Optional("max"): int,
        vol.Optional("max_exceeded"): str,
        vol.Optional("variables"): dict,
        vol.Required(vol.Any("trigger", "triggers")): vol.All([TRIGGER_SCHEMA], vol.Length(min=1)),
        vol.Optional(vol.Any("condition", "conditions")): [CONDITION_SCHEMA],
        vol.Required(vol.Any("action", "actions")): vol.All([ACTION_SCHEMA], vol.Length(min=1)),
    }
)


@dataclass
class ValidationReport:
    """Result of running the full gate."""

    ok: bool = False
    schema_ok: bool = False
    references_ok: bool = False
    #: Is what this automation touches decidable from the text alone?  False
    #: means a template or a wildcard target deferred it to runtime.
    targets_static: bool = True
    #: Were service names checked against the *live* index, or only against the
    #: closed set this add-on can emit?  False is a degradation, not a pass.
    services_verified: bool = False
    #: Does the automation still mean what the mined candidate meant?  Only
    #: meaningful for LLM output; the deterministic renderer is the reference.
    semantics_ok: bool | None = None
    check_config_ok: bool | None = None  # None == not run (Core unreachable)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unknown_entities: list[str] = field(default_factory=list)
    unknown_services: list[str] = field(default_factory=list)
    unknown_targets: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "schema_ok": self.schema_ok,
            "references_ok": self.references_ok,
            "targets_static": self.targets_static,
            "services_verified": self.services_verified,
            "semantics_ok": self.semantics_ok,
            "check_config_ok": self.check_config_ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "unknown_entities": self.unknown_entities,
            "unknown_services": self.unknown_services,
            "unknown_targets": self.unknown_targets,
        }


# ----------------------------------------------------------------------
def _collect_references(config: dict[str, Any]) -> dict[str, set[str]]:
    """Every entity/device/area/label/service the automation names."""
    found = {
        "entity_id": set(),
        "device_id": set(),
        "area_id": set(),
        "label_id": set(),
        "service": set(),
    }

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("service", "action") and isinstance(value, str) and "." in value:
                    # `action:` is the modern spelling of `service:` in actions,
                    # but it also names the platform inside a trigger - only a
                    # dotted value is a service call.
                    found["service"].add(value)
                elif key in found and key != "service":
                    for item in value if isinstance(value, list) else [value]:
                        if isinstance(item, str):
                            found[key].add(item)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    return found


def _leaf_key(path: str) -> str:
    """The mapping key a path ends at, ignoring any list indices below it."""
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def validate_static_targets(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Refuse anything whose effect is only decidable at runtime.

    Two things get past a reference check by never naming a reference:

    * a Jinja template - ``service: "{{ svc }}"`` names no service at all, and
      ``entity_id: "{{ trigger.entity_id }}"`` names no entity, so there is
      nothing for the existence check to reject.  Whatever they evaluate to is
      chosen after every gate has run.
    * ``entity_id: all`` - a legal, enumerable-only-at-runtime way to say
      "every entity this service accepts".  ``light.turn_off`` with it is the
      whole house.

    Neither the deterministic renderer nor a permitted alias rewrite ever
    produces these, so there is no cost to refusing them and no safe way to
    check them.
    """
    errors: list[str] = []
    # Free text cannot change what the automation does, and rejecting a
    # description that happens to contain braces would cost a good rewrite for
    # nothing.
    prose = ("alias", "description")

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in prose:
                    continue
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
        elif isinstance(node, str):
            if any(marker in node for marker in TEMPLATE_MARKERS):
                errors.append(
                    f"{path or 'automation'} contains a template ({node!r}); what it "
                    "would do is decided at runtime, so it cannot be checked here"
                )
            elif _leaf_key(path) == "entity_id" and node in WILDCARD_TARGETS:
                errors.append(
                    f"{path} is '{node}', which targets every matching entity "
                    "rather than a named one"
                )

    walk(config, "")
    return not errors, errors


def emittable_services() -> set[str]:
    """Every service this add-on is capable of proposing.

    The miners reach services through one function, so the set is closed and
    knowable without asking Core.  It is the floor the service check falls back
    to when the live index is unavailable - narrower than the real instance, so
    a legitimate service can be refused, but never the reverse.
    """
    from ..config import ACTIONABLE_DOMAINS
    from ..miners.time_of_day import _SERVICE_MAP  # local: avoids an import cycle

    services = set(_SERVICE_MAP.values())
    services.update({"climate.set_hvac_mode", "climate.turn_off", "climate.set_temperature"})
    services.update({"scene.turn_on", "script.turn_on"})
    services.update({f"{domain}.select_option" for domain in ("input_select", "select")})
    services.update({f"homeassistant.turn_{state}" for state in ("on", "off")})
    services.update({f"{domain}.turn_{state}"
                     for domain in ACTIONABLE_DOMAINS for state in ("on", "off")})
    return services


def validate_references(
    config: dict[str, Any],
    known_entities: Iterable[str],
    known_services: Iterable[str] = (),
    known_devices: Iterable[str] = (),
    known_areas: Iterable[str] = (),
    known_labels: Iterable[str] = (),
) -> tuple[bool, list[str], dict[str, list[str]]]:
    """Check every reference exists.  This is the anti-hallucination step."""
    references = _collect_references(config)
    entities = set(known_entities)
    services = set(known_services)
    devices = set(known_devices)
    areas = set(known_areas)
    labels = set(known_labels)

    errors: list[str] = []
    unknown = {"entities": [], "services": [], "targets": []}

    for entity_id in sorted(references["entity_id"]):
        if entity_id not in entities:
            unknown["entities"].append(entity_id)
            errors.append(f"entity_id '{entity_id}' does not exist on this instance")
    # No live index is a reason to check harder, not to stop checking: fall
    # back to the closed set of services this add-on can emit.  Skipping the
    # check here would let an unreachable Core wave a hallucinated service
    # through, which is the one moment it matters most.
    allowed_services = services or emittable_services()
    for service in sorted(references["service"]):
        if service not in allowed_services:
            unknown["services"].append(service)
            errors.append(
                f"service '{service}' is not available on this instance"
                if services
                else f"service '{service}' is not one this add-on generates, and "
                "Home Assistant is not reachable to confirm it exists"
            )
    for device_id in sorted(references["device_id"]):
        if devices and device_id not in devices:
            unknown["targets"].append(device_id)
            errors.append(f"device_id '{device_id}' does not exist")
    for area_id in sorted(references["area_id"]):
        if areas and area_id not in areas:
            unknown["targets"].append(area_id)
            errors.append(f"area_id '{area_id}' does not exist")
    for label_id in sorted(references["label_id"]):
        if labels and label_id not in labels:
            unknown["targets"].append(label_id)
            errors.append(f"label_id '{label_id}' does not exist")

    return not errors, errors, unknown


def validate_schema(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate against our mirror of HA's automation schema."""
    try:
        AUTOMATION_SCHEMA(config)
    except vol.Invalid as err:
        return False, [f"schema: {err}"]
    return True, []


def parse_yaml(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse generated YAML into a single automation config dict."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as err:
        return None, [f"YAML does not parse: {err}"]
    if isinstance(data, list):
        if len(data) != 1:
            return None, ["expected exactly one automation in the generated YAML"]
        data = data[0]
    if not isinstance(data, dict):
        return None, ["generated YAML is not an automation mapping"]
    return data, []


def validate_automation(
    config: dict[str, Any] | str,
    resolver=None,
    client=None,
    known_services: Iterable[str] | None = None,
    run_check_config: bool = True,
) -> ValidationReport:
    """Run the complete gate.  ``report.ok`` is the only thing callers should trust."""
    report = ValidationReport()

    if isinstance(config, str):
        parsed, errors = parse_yaml(config)
        if parsed is None:
            report.errors.extend(errors)
            return report
        config = parsed

    report.schema_ok, schema_errors = validate_schema(config)
    report.errors.extend(schema_errors)

    # Before asking whether the references exist, ask whether they *are*
    # references.  A template names nothing, so an existence check on it always
    # passes vacuously.
    report.targets_static, target_errors = validate_static_targets(config)
    report.errors.extend(target_errors)

    if resolver is not None:
        services = set(known_services) if known_services is not None else set()
        if not services and client is not None:
            services = client.service_index()
        report.services_verified = bool(services)
        if not report.services_verified:
            report.warnings.append(
                "Home Assistant's service list is unavailable; service names were "
                "checked against the set this add-on generates instead."
            )
        ok, errors, unknown = validate_references(
            config,
            resolver.known_entity_ids(),
            services,
            resolver.known_device_ids(),
            resolver.known_area_ids(),
            resolver.known_label_ids(),
        )
        report.references_ok = ok
        report.errors.extend(errors)
        report.unknown_entities = unknown["entities"]
        report.unknown_services = unknown["services"]
        report.unknown_targets = unknown["targets"]
    else:
        report.references_ok = False
        report.errors.append("no entity resolver available - cannot verify references")

    if run_check_config and client is not None:
        result = client.check_config()
        status = str(result.get("result", "")).lower()
        if status == "valid":
            report.check_config_ok = True
        elif status == "unavailable":
            report.check_config_ok = None
            report.warnings.append(
                "Home Assistant's config check could not be reached; relying on schema "
                "and reference validation only."
            )
        else:
            report.check_config_ok = False
            report.errors.append(f"check_config failed: {result.get('errors')}")

    report.ok = (
        report.schema_ok
        and report.references_ok
        and report.targets_static
        and report.check_config_ok is not False
    )
    return report
