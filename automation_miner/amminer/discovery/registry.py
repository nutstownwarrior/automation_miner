"""Read Home Assistant's ``.storage`` registries.

Each registry file is ``{"version": N, "key": "...", "data": {...}}``.  We read
only the **active** lists (``data.entities`` etc.) and deliberately ignore the
``deleted_*`` sections, which on a long-lived instance can dwarf the live data.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

STORAGE_DIR = ".storage"

ENTITY_REGISTRY = "core.entity_registry"
DEVICE_REGISTRY = "core.device_registry"
AREA_REGISTRY = "core.area_registry"
LABEL_REGISTRY = "core.label_registry"  # HA >= 2024.4
FLOOR_REGISTRY = "core.floor_registry"  # HA >= 2024.4


def _read_storage(config_dir: Path, key: str) -> dict[str, Any] | None:
    path = Path(config_dir) / STORAGE_DIR / key
    if not path.is_file():
        _LOGGER.debug("Registry file %s not present", path)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        _LOGGER.warning("Could not read registry %s: %s", path, err)
        return None
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    return data if isinstance(data, dict) else None


def _as_list(data: dict[str, Any] | None, *keys: str) -> list[dict[str, Any]]:
    """Pull the first present active list, ignoring ``deleted_*`` siblings."""
    if not data:
        return []
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


@dataclass
class RegistryEntity:
    entity_id: str
    unique_id: str | None = None
    platform: str | None = None
    device_id: str | None = None
    area_id: str | None = None
    config_entry_id: str | None = None
    name: str | None = None
    original_name: str | None = None
    name_by_user: str | None = None
    disabled_by: str | None = None
    hidden_by: str | None = None
    entity_category: str | None = None
    device_class: str | None = None
    original_device_class: str | None = None
    labels: list[str] = field(default_factory=list)
    categories: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.disabled_by is None

    @property
    def visible(self) -> bool:
        return self.hidden_by is None


@dataclass
class RegistryDevice:
    id: str
    name: str | None = None
    name_by_user: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    area_id: str | None = None
    via_device_id: str | None = None
    disabled_by: str | None = None
    labels: list[str] = field(default_factory=list)
    config_entries: list[str] = field(default_factory=list)
    identifiers: list[Any] = field(default_factory=list)

    @property
    def display_name(self) -> str | None:
        return self.name_by_user or self.name


@dataclass
class RegistryArea:
    id: str
    name: str | None = None
    floor_id: str | None = None
    labels: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    icon: str | None = None


@dataclass
class Registries:
    """The four (five with floors) registries, already normalised."""

    entities: dict[str, RegistryEntity] = field(default_factory=dict)
    devices: dict[str, RegistryDevice] = field(default_factory=dict)
    areas: dict[str, RegistryArea] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)  # label_id -> name
    floors: dict[str, str] = field(default_factory=dict)  # floor_id -> name
    source: str = "none"  # storage | websocket | none

    @property
    def available(self) -> bool:
        return bool(self.entities or self.devices or self.areas)

    def stats(self) -> dict[str, int | str]:
        return {
            "source": self.source,
            "entities": len(self.entities),
            "devices": len(self.devices),
            "areas": len(self.areas),
            "labels": len(self.labels),
            "floors": len(self.floors),
        }


def _entity_from_dict(raw: dict[str, Any]) -> RegistryEntity | None:
    entity_id = raw.get("entity_id")
    if not isinstance(entity_id, str) or "." not in entity_id:
        return None
    options = raw.get("options") if isinstance(raw.get("options"), dict) else {}
    return RegistryEntity(
        entity_id=entity_id,
        unique_id=raw.get("unique_id"),
        platform=raw.get("platform"),
        device_id=raw.get("device_id"),
        area_id=raw.get("area_id"),
        config_entry_id=raw.get("config_entry_id"),
        name=raw.get("name"),
        original_name=raw.get("original_name"),
        name_by_user=raw.get("name_by_user"),
        disabled_by=raw.get("disabled_by"),
        hidden_by=raw.get("hidden_by"),
        entity_category=raw.get("entity_category"),
        device_class=raw.get("device_class"),
        original_device_class=raw.get("original_device_class"),
        labels=[str(x) for x in raw.get("labels") or []],
        categories=raw.get("categories") if isinstance(raw.get("categories"), dict) else {},
        aliases=[str(x) for x in raw.get("aliases") or []]
        or [str(x) for x in (options.get("conversation", {}) or {}).get("aliases", [])],
    )


def _device_from_dict(raw: dict[str, Any]) -> RegistryDevice | None:
    device_id = raw.get("id")
    if not isinstance(device_id, str):
        return None
    return RegistryDevice(
        id=device_id,
        name=raw.get("name"),
        name_by_user=raw.get("name_by_user"),
        manufacturer=raw.get("manufacturer"),
        model=raw.get("model"),
        area_id=raw.get("area_id"),
        via_device_id=raw.get("via_device_id"),
        disabled_by=raw.get("disabled_by"),
        labels=[str(x) for x in raw.get("labels") or []],
        config_entries=[str(x) for x in raw.get("config_entries") or []],
        identifiers=list(raw.get("identifiers") or []),
    )


def _area_from_dict(raw: dict[str, Any]) -> RegistryArea | None:
    area_id = raw.get("id")
    if not isinstance(area_id, str):
        return None
    return RegistryArea(
        id=area_id,
        name=raw.get("name"),
        floor_id=raw.get("floor_id"),
        labels=[str(x) for x in raw.get("labels") or []],
        aliases=[str(x) for x in raw.get("aliases") or []],
        icon=raw.get("icon"),
    )


def registries_from_payloads(
    entities: Iterable[dict[str, Any]] = (),
    devices: Iterable[dict[str, Any]] = (),
    areas: Iterable[dict[str, Any]] = (),
    labels: Iterable[dict[str, Any]] = (),
    floors: Iterable[dict[str, Any]] = (),
    source: str = "storage",
) -> Registries:
    """Normalise raw registry payloads (from ``.storage`` *or* WebSocket)."""
    reg = Registries(source=source)
    for raw in entities:
        entity = _entity_from_dict(raw)
        if entity is not None:
            reg.entities[entity.entity_id] = entity
    for raw in devices:
        device = _device_from_dict(raw)
        if device is not None:
            reg.devices[device.id] = device
    for raw in areas:
        area = _area_from_dict(raw)
        if area is not None:
            reg.areas[area.id] = area
    for raw in labels:
        label_id = raw.get("label_id") or raw.get("id")
        if isinstance(label_id, str):
            reg.labels[label_id] = str(raw.get("name") or label_id)
    for raw in floors:
        floor_id = raw.get("floor_id") or raw.get("id")
        if isinstance(floor_id, str):
            reg.floors[floor_id] = str(raw.get("name") or floor_id)
    return reg


def read_registries(config_dir: str | Path) -> Registries:
    """Read all registries from ``<config>/.storage``.

    Returns an empty (``available == False``) result rather than raising, so the
    caller can fall back to the WebSocket API or to ``/api/states``.
    """
    config_dir = Path(config_dir)
    entity_data = _read_storage(config_dir, ENTITY_REGISTRY)
    device_data = _read_storage(config_dir, DEVICE_REGISTRY)
    area_data = _read_storage(config_dir, AREA_REGISTRY)
    label_data = _read_storage(config_dir, LABEL_REGISTRY)
    floor_data = _read_storage(config_dir, FLOOR_REGISTRY)

    reg = registries_from_payloads(
        entities=_as_list(entity_data, "entities"),
        devices=_as_list(device_data, "devices"),
        areas=_as_list(area_data, "areas"),
        labels=_as_list(label_data, "labels"),
        floors=_as_list(floor_data, "floors"),
        source="storage",
    )
    if not reg.available:
        reg.source = "none"
    _LOGGER.info("Registries from .storage: %s", reg.stats())
    return reg
