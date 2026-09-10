"""Optional AI feature 1 - semantic classification of the entity inventory.

:mod:`amminer.enrich.detect` labels signal roles with regular expressions over
entity ids, friendly names, platforms and models.  That works well for common
English naming and core integrations, and badly for everything else: a price
sensor called ``sensor.stroomprijs`` or a dishwasher called ``switch.geschirr``
is simply invisible, and an invisible signal cannot become a condition in any
mined rule.

An LLM is good at exactly this: reading names and models and saying what a thing
*is*.  So this module asks one, in batches, and merges the answer with the
regex detector.

Two properties make it safe to switch on:

**Additive only.**  A role the deterministic detector found is never removed by
the model.  The worst a bad classification can do is add a signal that later
fails to explain anything - which the conditional miner and the backtester
already discard on the evidence.

**Verified.**  Every returned entity id must exist in the resolver and every
returned role must be one the internal taxonomy already knows.  Anything else is
dropped and counted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..enrich.detect import SignalSet
from .provider import BaseProvider, LLMError

_LOGGER = logging.getLogger(__name__)

#: Roles the model may assign - exactly the fields of :class:`SignalSet`.
ROLE_TAXONOMY: tuple[str, ...] = tuple(SignalSet().__dict__.keys())

#: Roles that carry real weight in mining, described so the model can tell them
#: apart.  Anything not listed is still accepted if it is a SignalSet field.
ROLE_HINTS = {
    "outdoor_temperature": "measures temperature OUTSIDE the building",
    "illuminance": "measures ambient light level (lux)",
    "workday": "true/false whether today is a working day",
    "holiday": "true/false whether today is a public holiday or school holiday",
    "person": "tracks whether a specific person is home",
    "device_tracker": "tracks a device's presence (phone, car, router client)",
    "room_presence": "reports which room someone is in (mmWave, BLE, ESPresense)",
    "occupancy": "true/false whether a room is occupied",
    "motion": "PIR-style movement detection only",
    "energy_price": "current electricity price as a number",
    "price_level": "electricity price as a category such as cheap/normal/expensive",
    "solar_forecast": "PREDICTED future solar production",
    "solar_production": "CURRENT measured solar production",
    "carbon_intensity": "grid carbon intensity, gCO2eq/kWh",
    "weather_warning": "official severe-weather warning level",
    "deferrable_loads": "an appliance whose run time can be moved: dishwasher, "
    "washing machine, dryer, water heater, pool pump",
    "ev_charger": "electric-vehicle charger",
    "thermostat": "controls heating or cooling setpoint",
}

SYSTEM_PROMPT = """\
You label Home Assistant entities with the role they play in home automation.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- Only use entity_ids that appear in the input. Never invent one.
- Only use role names from the "roles" list given in the input.
- Assign a role ONLY when you are confident from the name, device class, unit,
  manufacturer or model. Leave an entity out entirely when unsure - a wrong
  label is worse than no label.
- Most entities have NO role. Returning few assignments is correct and expected.
- An entity may have more than one role.

Shape:
{"assignments": [{"entity_id": "sensor.x", "roles": ["energy_price"]}]}
"""


@dataclass
class ClassificationResult:
    """What the model contributed, and what was thrown away."""

    assignments: dict[str, list[str]] = field(default_factory=dict)
    added: dict[str, list[str]] = field(default_factory=dict)
    unknown_entities: list[str] = field(default_factory=list)
    unknown_roles: list[str] = field(default_factory=list)
    batches: int = 0
    from_cache: bool = False
    error: str | None = None

    @property
    def added_count(self) -> int:
        return sum(len(v) for v in self.added.values())

    def as_dict(self) -> dict[str, Any]:
        """A bounded summary, safe to persist in every run record.

        The full ``assignments`` and ``added`` maps are deliberately NOT here:
        this dict ends up in the ``runs.stats`` column on every nightly run, and
        on a large instance those maps would grow that table for no benefit -
        the authoritative copy already lives in the classification cache, and
        the UI only needs the counts and a sample.
        """
        return {
            "added_count": self.added_count,
            "added_sample": dict(sorted(self.added.items())[:20]),
            "assigned_entities": len(self.assignments),
            "unknown_entities": self.unknown_entities[:20],
            "unknown_roles": sorted(set(self.unknown_roles))[:20],
            "batches": self.batches,
            "from_cache": self.from_cache,
            "error": self.error,
        }


def _entity_descriptor(info) -> dict[str, Any]:
    """The minimal, non-sensitive description the model needs.

    Deliberately no state and no history - only what the thing *is*.
    """
    payload = {"entity_id": info.entity_id, "name": info.name}
    for key, value in (
        ("area", info.area_name),
        ("device", info.device_name),
        ("manufacturer", info.manufacturer),
        ("model", info.model),
        ("device_class", info.device_class),
        ("unit", info.unit_of_measurement),
        ("platform", info.platform),
    ):
        if value:
            payload[key] = value
    return payload


def classifiable_entities(resolver, options=None) -> list[Any]:
    """Entities worth asking about - excludes the obviously irrelevant."""
    interesting_domains = {
        "sensor", "binary_sensor", "switch", "climate", "person",
        "device_tracker", "weather", "calendar", "water_heater", "vacuum",
    }
    out = []
    for info in resolver.entities.values():
        if info.domain not in interesting_domains:
            continue
        if info.disabled or info.entity_category is not None:
            continue
        if options is not None and options.is_excluded(info.entity_id):
            continue
        out.append(info)
    return sorted(out, key=lambda i: i.entity_id)


def inventory_fingerprint(entities: Sequence[Any]) -> str:
    """Stable hash of the inventory, so a cached answer can be reused."""
    payload = json.dumps(
        [_entity_descriptor(info) for info in entities], sort_keys=True, default=str
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _validate(raw: dict[str, Any], known_ids: set[str]) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Keep only assignments naming a real entity and a real role."""
    assignments: dict[str, list[str]] = {}
    unknown_entities: list[str] = []
    unknown_roles: list[str] = []

    items = raw.get("assignments")
    if not isinstance(items, list):
        return {}, [], []

    for item in items:
        if not isinstance(item, dict):
            continue
        roles = item.get("roles")
        if isinstance(roles, str):
            roles = [roles]
        if not isinstance(roles, list):
            continue
        # Roles are checked first so an invented role is still counted even when
        # the entity it was attached to is rejected too - both are diagnostics
        # the Status page shows, and each should be accurate on its own.
        kept = []
        for role in roles:
            if not isinstance(role, str):
                continue
            role = role.strip().lower()
            if role in ROLE_TAXONOMY:
                kept.append(role)
            else:
                unknown_roles.append(role)

        entity_id = item.get("entity_id")
        if not isinstance(entity_id, str) or entity_id not in known_ids:
            if isinstance(entity_id, str):
                unknown_entities.append(entity_id)
            continue
        if kept:
            assignments.setdefault(entity_id, [])
            for role in kept:
                if role not in assignments[entity_id]:
                    assignments[entity_id].append(role)
    return assignments, unknown_entities, unknown_roles


def _batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def classify(
    resolver,
    provider: BaseProvider,
    options=None,
    store=None,
    batch_size: int = 60,
) -> ClassificationResult:
    """Ask the model to label the entity inventory.  Never raises."""
    result = ClassificationResult()
    entities = classifiable_entities(resolver, options)
    if not entities:
        return result
    if not provider.enabled:
        result.error = "no LLM provider configured"
        return result

    fingerprint = inventory_fingerprint(entities)
    cache_key = f"llm_classification:{fingerprint}"
    if store is not None:
        cached = store.get_meta(cache_key)
        if cached:
            try:
                result.assignments = json.loads(cached)
                result.from_cache = True
                _LOGGER.info(
                    "Reusing cached entity classification (%d entities unchanged)",
                    len(entities),
                )
                return result
            except (json.JSONDecodeError, TypeError):
                pass

    known_ids = {info.entity_id for info in entities}
    roles_payload = [
        {"role": role, "means": ROLE_HINTS[role]} if role in ROLE_HINTS else {"role": role}
        for role in ROLE_TAXONOMY
    ]

    for batch in _batches(entities, max(batch_size, 5)):
        prompt = json.dumps(
            {
                "roles": roles_payload,
                "entities": [_entity_descriptor(info) for info in batch],
            },
            indent=2,
            default=str,
        )
        try:
            raw = provider.complete_json(SYSTEM_PROMPT, prompt)
        except LLMError as err:
            result.error = str(err)
            _LOGGER.warning("Entity classification failed: %s", err)
            break
        result.batches += 1
        assignments, unknown_entities, unknown_roles = _validate(raw, known_ids)
        result.unknown_entities.extend(unknown_entities)
        result.unknown_roles.extend(unknown_roles)
        for entity_id, roles in assignments.items():
            existing = result.assignments.setdefault(entity_id, [])
            for role in roles:
                if role not in existing:
                    existing.append(role)

    if result.assignments and store is not None and not result.error:
        store.set_meta(cache_key, json.dumps(result.assignments))

    if result.unknown_entities:
        _LOGGER.info(
            "Discarded %d hallucinated entity ids from the classifier",
            len(result.unknown_entities),
        )
    return result


def apply_to_signals(signals: SignalSet, result: ClassificationResult) -> SignalSet:
    """Merge the model's labels into *signals*, additively.

    Roles the deterministic detector already found always survive; the model can
    only ever contribute an entity that was not there before.  ``result.added``
    records exactly what the model contributed, for the UI.
    """
    for entity_id, roles in result.assignments.items():
        for role in roles:
            current = getattr(signals, role, None)
            if not isinstance(current, list):
                continue
            if entity_id in current:
                continue  # the regex detector already had it
            current.append(entity_id)
            result.added.setdefault(entity_id, []).append(role)

    for role in ROLE_TAXONOMY:
        current = getattr(signals, role, None)
        if isinstance(current, list):
            setattr(signals, role, sorted(dict.fromkeys(current)))
    if result.added:
        _LOGGER.info(
            "Entity classifier added %d signal roles the regex detector missed",
            result.added_count,
        )
    return signals
