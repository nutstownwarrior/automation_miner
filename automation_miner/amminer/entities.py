"""The entity resolver: cryptic ``entity_id`` -> human meaning.

Home Assistant's entity registry only contains entities that report a
``unique_id``.  Template sensors, YAML platforms, ``scrape``, zones, groups and
plenty more are *absent* from it.  The resolver therefore UNIONs the registry
with a live ``GET /api/states`` snapshot, so mining, prompts and the UI always
see the complete entity set.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .discovery.registry import Registries

_LOGGER = logging.getLogger(__name__)


def _titleise(entity_id: str) -> str:
    object_id = entity_id.split(".", 1)[1] if "." in entity_id else entity_id
    return object_id.replace("_", " ").strip().title()


@dataclass
class EntityInfo:
    """Everything we know about one entity, from every available source."""

    entity_id: str
    domain: str = ""
    friendly_name: str | None = None
    area_id: str | None = None
    area_name: str | None = None
    #: True when the area above was guessed rather than read from the registry.
    #: Nothing may display or export the area without also saying this.
    area_inferred: bool = False
    area_inferred_reason: str = ""
    floor_id: str | None = None
    floor_name: str | None = None
    device_id: str | None = None
    device_name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    platform: str | None = None
    device_class: str | None = None
    unit_of_measurement: str | None = None
    entity_category: str | None = None
    labels: list[str] = field(default_factory=list)
    label_names: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    unique_id: str | None = None
    config_entry_id: str | None = None
    disabled: bool = False
    hidden: bool = False
    in_registry: bool = False
    in_states: bool = False
    state: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.domain and "." in self.entity_id:
            self.domain = self.entity_id.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.friendly_name or _titleise(self.entity_id)

    @property
    def is_numeric(self) -> bool:
        if self.domain not in ("sensor", "number", "input_number"):
            return False
        if self.state in (None, "unknown", "unavailable", ""):
            return bool(self.unit_of_measurement)
        try:
            float(str(self.state))
        except (TypeError, ValueError):
            return False
        return True

    def describe(self) -> str:
        """One-line human description used in UI text and LLM prompts."""
        parts = [self.name]
        if self.area_name:
            # A guess is labelled everywhere it is shown, including in the
            # prompts built from this, so nothing downstream can mistake an
            # inference for something the user actually configured.
            parts.append(
                f"in {self.area_name}" + (" (guessed)" if self.area_inferred else "")
            )
        if self.device_name and self.device_name != self.name:
            parts.append(f"on {self.device_name}")
        if self.label_names:
            parts.append(f"[{', '.join(self.label_names)}]")
        return " ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "domain": self.domain,
            "name": self.name,
            "area": self.area_name,
            "area_inferred": self.area_inferred,
            "floor": self.floor_name,
            "device": self.device_name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "platform": self.platform,
            "device_class": self.device_class,
            "unit_of_measurement": self.unit_of_measurement,
            "labels": self.label_names,
            "in_registry": self.in_registry,
            "in_states": self.in_states,
            "disabled": self.disabled,
            "hidden": self.hidden,
        }


class EntityResolver:
    """Union of the registries and the live states snapshot."""

    def __init__(self, registries: Registries | None = None) -> None:
        self.registries = registries or Registries()
        self.entities: dict[str, EntityInfo] = {}
        self.sources: dict[str, bool] = {"registry": False, "states": False}
        if registries is not None and registries.available:
            self._load_registry()

    # ------------------------------------------------------------------
    def _area_of(self, area_id: str | None) -> tuple[str | None, str | None, str | None]:
        """Return ``(area_id, area_name, floor_id)``."""
        if not area_id:
            return None, None, None
        area = self.registries.areas.get(area_id)
        if area is None:
            return area_id, None, None
        return area_id, area.name, area.floor_id

    def _label_names(self, label_ids: Iterable[str]) -> list[str]:
        return [self.registries.labels.get(lid, lid) for lid in label_ids]

    def _load_registry(self) -> None:
        reg = self.registries
        for entity_id, entry in reg.entities.items():
            device = reg.devices.get(entry.device_id) if entry.device_id else None
            # Entity area wins; otherwise inherit the device's area (HA semantics).
            area_id = entry.area_id or (device.area_id if device else None)
            area_id, area_name, floor_id = self._area_of(area_id)
            labels = list(entry.labels)
            if device is not None:
                labels.extend(lbl for lbl in device.labels if lbl not in labels)
            info = EntityInfo(
                entity_id=entity_id,
                friendly_name=entry.name_by_user or entry.name or entry.original_name,
                area_id=area_id,
                area_name=area_name,
                floor_id=floor_id,
                floor_name=reg.floors.get(floor_id) if floor_id else None,
                device_id=entry.device_id,
                device_name=device.display_name if device else None,
                manufacturer=device.manufacturer if device else None,
                model=device.model if device else None,
                platform=entry.platform,
                device_class=entry.device_class or entry.original_device_class,
                entity_category=entry.entity_category,
                labels=labels,
                label_names=self._label_names(labels),
                aliases=list(entry.aliases),
                unique_id=entry.unique_id,
                config_entry_id=entry.config_entry_id,
                disabled=not entry.enabled,
                hidden=not entry.visible,
                in_registry=True,
            )
            self.entities[entity_id] = info
        self.sources["registry"] = bool(reg.entities)

    # ------------------------------------------------------------------
    def merge_states(self, states: Iterable[dict[str, Any]]) -> int:
        """Merge a ``/api/states`` snapshot, creating entries as needed.

        Returns the number of entities that existed *only* in the states set -
        i.e. the ones a registry-only implementation would have missed.
        """
        recovered = 0
        for row in states:
            entity_id = row.get("entity_id")
            if not isinstance(entity_id, str) or "." not in entity_id:
                continue
            attributes = row.get("attributes") or {}
            info = self.entities.get(entity_id)
            if info is None:
                info = EntityInfo(entity_id=entity_id)
                self.entities[entity_id] = info
                recovered += 1
            info.in_states = True
            info.state = row.get("state")
            info.attributes = dict(attributes) if isinstance(attributes, dict) else {}
            # States are the source of truth for the *displayed* name.
            friendly = info.attributes.get("friendly_name")
            if isinstance(friendly, str) and friendly:
                info.friendly_name = friendly
            if info.device_class is None:
                device_class = info.attributes.get("device_class")
                if isinstance(device_class, str):
                    info.device_class = device_class
            unit = info.attributes.get("unit_of_measurement")
            if isinstance(unit, str):
                info.unit_of_measurement = unit
            # Entities outside the registry can still carry area/labels via
            # attributes on some integrations; never overwrite registry values.
            if info.area_id is None and isinstance(info.attributes.get("area_id"), str):
                area_id, area_name, floor_id = self._area_of(info.attributes["area_id"])
                info.area_id, info.area_name, info.floor_id = area_id, area_name, floor_id
                info.floor_name = self.registries.floors.get(floor_id) if floor_id else None
        self.sources["states"] = True
        if recovered:
            _LOGGER.info(
                "Recovered %d entities present in /api/states but absent from the registry",
                recovered,
            )
        return recovered

    def seed_from_recorder(self, entity_ids: Iterable[str]) -> int:
        """Add entities that only the recorder knows about.

        The last line of defence: when neither the registry nor ``/api/states``
        is readable, the recorder still knows every entity_id it ever stored.
        These entries carry no area/device context, but they let the miners and
        the signal detector work on names alone rather than on nothing.
        """
        added = 0
        for entity_id in entity_ids:
            if not isinstance(entity_id, str) or "." not in entity_id:
                continue
            if entity_id in self.entities:
                continue
            self.entities[entity_id] = EntityInfo(entity_id=entity_id)
            added += 1
        if added:
            self.sources["recorder"] = True
            _LOGGER.info("Seeded %d entities from the recorder (no registry/states)", added)
        return added

    # ------------------------------------------------------------------
    def get(self, entity_id: str) -> EntityInfo | None:
        return self.entities.get(entity_id)

    def resolve(self, entity_id: str) -> EntityInfo:
        """Always return an ``EntityInfo``, synthesising one when unknown."""
        info = self.entities.get(entity_id)
        if info is None:
            info = EntityInfo(entity_id=entity_id)
        return info

    def name_of(self, entity_id: str) -> str:
        return self.resolve(entity_id).name

    def describe(self, entity_id: str) -> str:
        return self.resolve(entity_id).describe()

    def exists(self, entity_id: str) -> bool:
        return entity_id in self.entities

    def known_entity_ids(self) -> set[str]:
        return set(self.entities)

    def known_device_ids(self) -> set[str]:
        return set(self.registries.devices)

    def known_area_ids(self) -> set[str]:
        return set(self.registries.areas)

    def known_label_ids(self) -> set[str]:
        return set(self.registries.labels)

    def by_domain(self, *domains: str) -> list[EntityInfo]:
        wanted = set(domains)
        return [e for e in self.entities.values() if e.domain in wanted]

    def in_area(self, area_id: str) -> list[EntityInfo]:
        return [e for e in self.entities.values() if e.area_id == area_id]

    def find_by_device_class(self, device_class: str, domain: str | None = None) -> list[EntityInfo]:
        return [
            e
            for e in self.entities.values()
            if e.device_class == device_class and (domain is None or e.domain == domain)
        ]

    def search(self, needle: str) -> list[EntityInfo]:
        needle = needle.lower().strip()
        if not needle:
            return []
        results = []
        for info in self.entities.values():
            haystack = " ".join(
                filter(
                    None,
                    [info.entity_id, info.friendly_name, info.area_name, info.device_name],
                )
            ).lower()
            if needle in haystack:
                results.append(info)
        return sorted(results, key=lambda e: e.entity_id)

    def stats(self) -> dict[str, Any]:
        registry_only = sum(1 for e in self.entities.values() if e.in_registry and not e.in_states)
        states_only = sum(1 for e in self.entities.values() if e.in_states and not e.in_registry)
        return {
            "total": len(self.entities),
            "registry_only": registry_only,
            "states_only": states_only,
            "both": sum(1 for e in self.entities.values() if e.in_registry and e.in_states),
            "with_area": sum(1 for e in self.entities.values() if e.area_id),
            "sources": dict(self.sources),
            "registries": self.registries.stats(),
        }


def build_resolver(config_dir: str, client=None) -> EntityResolver:
    """Build a resolver with the documented three-step degradation.

    1. ``.storage`` registries (fast, no Core round-trip),
    2. WebSocket ``config/*_registry/list`` when ``.storage`` is unreadable,
    3. ``/api/states`` alone (no area/device/label context, flagged in the UI).
    """
    from .discovery.registry import read_registries, registries_from_payloads

    registries = read_registries(config_dir)

    if not registries.available and client is not None:
        _LOGGER.info("Registry files unreadable; falling back to the WebSocket API")
        payloads = client.ws_registry_lists()
        if payloads:
            registries = registries_from_payloads(
                entities=payloads.get("entities", []),
                devices=payloads.get("devices", []),
                areas=payloads.get("areas", []),
                labels=payloads.get("labels", []),
                floors=payloads.get("floors", []),
                source="websocket",
            )

    resolver = EntityResolver(registries)
    if client is not None:
        states = client.get_states()
        if states:
            resolver.merge_states(states)
        elif not registries.available:
            _LOGGER.warning(
                "Neither registries nor /api/states available - entity context is unavailable"
            )
    return resolver
