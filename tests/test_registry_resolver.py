"""Registry parsing and the entity resolver, including the states-union path."""

from __future__ import annotations

import json

from amminer.discovery.registry import read_registries
from amminer.entities import EntityResolver, build_resolver


def test_reads_active_lists_and_ignores_deleted(registry_files):
    registries = read_registries(registry_files)
    assert registries.source == "storage"
    # 3 active entities; the 500 deleted_entities must not appear.
    assert len(registries.entities) == 3
    assert not any("ghost" in e for e in registries.entities)
    assert len(registries.devices) == 2
    assert not any("gone" in d for d in registries.devices)
    assert registries.areas["area_kitchen"].name == "Kitchen"
    assert registries.labels["lbl_downstairs"] == "Downstairs"
    assert registries.floors["floor_ground"] == "Ground floor"


def test_entity_inherits_device_area_and_labels(registry_files):
    resolver = EntityResolver(read_registries(registry_files))
    kitchen = resolver.get("light.kitchen")
    assert kitchen is not None
    # The entity has no area_id of its own; it inherits the device's.
    assert kitchen.area_name == "Kitchen"
    assert kitchen.floor_name == "Ground floor"
    assert kitchen.device_name == "Hue bridge lamp"
    assert kitchen.manufacturer == "Signify"
    assert "Downstairs" in kitchen.label_names

    heater = resolver.get("switch.heater")
    assert heater.area_name == "Living room"
    # Device labels merge into the entity's.
    assert "Power hungry" in heater.label_names
    assert heater.aliases == ["the heater"]


def test_disabled_entity_flagged(registry_files):
    resolver = EntityResolver(read_registries(registry_files))
    assert resolver.get("sensor.disabled_thing").disabled is True


def test_states_union_recovers_entities_without_unique_id(registry_files, states_payload):
    resolver = EntityResolver(read_registries(registry_files))
    assert not resolver.exists("sensor.outdoor_temperature")  # not in the registry

    recovered = resolver.merge_states(states_payload)

    assert resolver.exists("sensor.outdoor_temperature")
    assert resolver.get("sensor.outdoor_temperature").in_registry is False
    assert resolver.get("sensor.outdoor_temperature").in_states is True
    assert recovered >= 6
    stats = resolver.stats()
    assert stats["states_only"] >= 6
    assert stats["both"] >= 2


def test_states_supply_friendly_name_and_unit(registry_files, states_payload):
    resolver = EntityResolver(read_registries(registry_files))
    resolver.merge_states(states_payload)
    temp = resolver.get("sensor.outdoor_temperature")
    assert temp.friendly_name == "Outdoor temperature"
    assert temp.unit_of_measurement == "°C"
    assert temp.device_class == "temperature"
    assert temp.is_numeric is True


def test_registry_wins_over_states_for_area(registry_files, states_payload):
    resolver = EntityResolver(read_registries(registry_files))
    resolver.merge_states(states_payload)
    # /api/states carries no area, so the registry value must survive the merge.
    assert resolver.get("light.kitchen").area_name == "Kitchen"


def test_describe_is_human_readable(registry_files, states_payload):
    resolver = EntityResolver(read_registries(registry_files))
    resolver.merge_states(states_payload)
    described = resolver.describe("switch.heater")
    assert "Heater" in described and "Living room" in described


def test_unknown_entity_falls_back_to_a_readable_name():
    resolver = EntityResolver()
    assert resolver.name_of("light.back_porch_lamp") == "Back Porch Lamp"
    assert resolver.exists("light.back_porch_lamp") is False


def test_build_resolver_falls_back_to_states_only(tmp_path, fake_client):
    """No .storage and no WebSocket: /api/states alone must still work."""
    (tmp_path / ".storage").mkdir()
    resolver = build_resolver(str(tmp_path), fake_client)
    assert resolver.sources["registry"] is False
    assert resolver.sources["states"] is True
    assert resolver.exists("light.kitchen")
    # Without a registry there is no area context - and that is reported.
    assert resolver.stats()["with_area"] == 0


def test_build_resolver_prefers_storage(registry_files, fake_client):
    resolver = build_resolver(str(registry_files), fake_client)
    assert resolver.sources["registry"] is True
    assert resolver.sources["states"] is True
    assert resolver.get("light.kitchen").area_name == "Kitchen"


def test_websocket_fallback_used_when_storage_unreadable(tmp_path, states_payload):
    class WSClient:
        configured = True

        def ws_registry_lists(self):
            return {
                "entities": [
                    {
                        "entity_id": "light.kitchen",
                        "unique_id": "u1",
                        "platform": "hue",
                        "device_id": None,
                        "area_id": "area_kitchen",
                        "labels": [],
                    }
                ],
                "devices": [],
                "areas": [{"id": "area_kitchen", "name": "Kitchen"}],
                "labels": [],
                "floors": [],
            }

        def get_states(self):
            return states_payload

    resolver = build_resolver(str(tmp_path), WSClient())
    assert resolver.registries.source == "websocket"
    assert resolver.get("light.kitchen").area_name == "Kitchen"


def test_corrupt_registry_file_is_survivable(tmp_path):
    storage = tmp_path / ".storage"
    storage.mkdir()
    (storage / "core.entity_registry").write_text("{not json", encoding="utf-8")
    registries = read_registries(tmp_path)
    assert registries.available is False


def test_registry_without_labels_section(tmp_path):
    """label_registry only exists from HA 2024.4; older instances must work."""
    storage = tmp_path / ".storage"
    storage.mkdir()
    (storage / "core.entity_registry").write_text(
        json.dumps(
            {
                "version": 1,
                "key": "core.entity_registry",
                "data": {"entities": [{"entity_id": "light.a", "unique_id": "x", "platform": "p"}]},
            }
        ),
        encoding="utf-8",
    )
    registries = read_registries(tmp_path)
    assert registries.available is True
    assert registries.labels == {}
