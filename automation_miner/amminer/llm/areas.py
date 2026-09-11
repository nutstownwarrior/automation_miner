"""Optional AI feature 9 - a room for entities the registry never placed.

Area is the only structural fact this add-on has about a home, and it is the one
users most often leave half-filled: a device set up in a hurry, an integration
that creates fifty entities at once, anything added before the areas existed.
Every entity without one is invisible to area-aware reasoning, and its cards
read "Switch 3" instead of "Switch 3 in the hallway".

The name is usually right there in the entity id or the device name, which is
exactly the kind of reading a model does well and a rule does badly
(``sensor.hue_motion_kitchen_2`` is the kitchen; ``binary_sensor.0x00158d``
is nothing).  So it is asked - under the narrowest mandate in this project:

* it is only shown entities whose area is **unset**.  An area the user assigned
  is never sent, never questioned and never overwritten,
* it may only answer with an area that already exists in the registry; it
  cannot invent a room, because a room the user does not have is not a place
  anything can be assigned to,
* every entity it places is flagged ``area_inferred``, so a guess is never
  displayed, exported or prompted with as though it were registry truth,
* nothing it says changes a score, a rule, or what is surfaced.

If the user has no areas at all, the feature reports that and does nothing.
Proposing a set of rooms would be inventing the structure it exists to read.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..util.text import clean_model_text
from .provider import BaseProvider, LLMError

_LOGGER = logging.getLogger(__name__)

#: Above this the prompt stops being a reading task and starts being a bulk
#: dump; entities are sent in batches of this size.
BATCH = 80

SYSTEM_PROMPT = """\
You assign home-automation entities to the room they are in.

Each entity has an id, a name and sometimes a device name. The room is usually
written in one of them. You are also given the EXACT list of rooms that exist in
this home.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- Only use "entity_id" values given in the input. Never invent one.
- "area" must be copied EXACTLY from the "areas" list. Never invent a room,
  never translate one, never merge two.
- If you cannot tell which room an entity is in, LEAVE IT OUT. A wrong room is
  worse than no room.
- An id like "binary_sensor.0x00158d0001" carries no room. Leave it out.
- Do not guess from the kind of device. A motion sensor is not automatically in
  a hallway.

Shape:
{"placements": [{"entity_id": "light.hue_kitchen_1", "area": "Kitchen",
                 "reason": "The entity id says kitchen."}]}
"""


@dataclass
class AreaResult:
    """Placements that survived validation."""

    #: entity_id -> {"area_id", "area_name", "reason"}
    placements: dict[str, dict[str, str]] = field(default_factory=dict)
    considered: int = 0
    rejected_unknown_entity: int = 0
    rejected_unknown_area: int = 0
    rejected_already_placed: int = 0
    applied: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "unplaced_entities": self.considered,
            "inferred": len(self.placements),
            "applied": self.applied,
            "rejected_unknown_entity": self.rejected_unknown_entity,
            "rejected_unknown_area": self.rejected_unknown_area,
            "rejected_already_placed": self.rejected_already_placed,
            "error": self.error,
        }


def unplaced(resolver) -> list[Any]:
    """Entities the registry gave no area, in a stable order."""
    return sorted(
        (
            info
            for info in resolver.entities.values()
            if not info.area_id and not info.area_name and not info.disabled
        ),
        key=lambda info: info.entity_id,
    )


def _area_index(resolver) -> dict[str, tuple[str, str]]:
    """Lower-cased area name -> ``(area_id, canonical name)``."""
    index: dict[str, tuple[str, str]] = {}
    for area_id, area in resolver.registries.areas.items():
        name = (area.name or "").strip()
        if name:
            index[name.lower()] = (area_id, name)
    return index


def infer(resolver, provider: BaseProvider, batch_size: int = BATCH) -> AreaResult:
    """Ask which room each unplaced entity is in.  Never raises."""
    result = AreaResult()
    candidates = unplaced(resolver)
    result.considered = len(candidates)
    if not candidates:
        return result
    if not provider.enabled:
        result.error = "no LLM provider configured"
        return result

    index = _area_index(resolver)
    if not index:
        # Without areas there is nothing to assign to, and proposing some would
        # be inventing the very structure this feature exists to read.
        result.error = "no areas are defined in Home Assistant"
        return result

    names = sorted(name for _, name in index.values())
    known_ids = {info.entity_id for info in candidates}
    step = max(int(batch_size), 1)
    for start in range(0, len(candidates), step):
        batch = candidates[start : start + step]
        prompt = json.dumps(
            {
                "areas": names,
                "entities": [
                    {
                        "entity_id": info.entity_id,
                        "name": info.name,
                        "device": info.device_name,
                    }
                    for info in batch
                ],
            },
            indent=2,
            default=str,
        )
        try:
            raw = provider.complete_json(SYSTEM_PROMPT, prompt)
        except LLMError as err:
            result.error = str(err)
            _LOGGER.warning("Area inference failed: %s", err)
            break

        placements = raw.get("placements")
        if not isinstance(placements, list):
            continue
        for placement in placements:
            if not isinstance(placement, dict):
                continue
            entity_id = placement.get("entity_id")
            if not isinstance(entity_id, str) or entity_id not in known_ids:
                result.rejected_unknown_entity += 1
                continue
            # Re-checked here and not only when the batch was built: an area
            # assigned by the user is never overwritten, whatever comes back.
            existing = resolver.entities.get(entity_id)
            if existing is None or existing.area_id or existing.area_name:
                result.rejected_already_placed += 1
                continue
            area_name = str(placement.get("area") or "").strip()
            match = index.get(area_name.lower())
            if match is None:
                result.rejected_unknown_area += 1
                continue
            area_id, canonical = match
            result.placements[entity_id] = {
                "area_id": area_id,
                "area_name": canonical,
                "reason": clean_model_text(placement.get("reason"), limit=200),
            }
    return result


def apply_inferences(resolver, result: AreaResult) -> AreaResult:
    """Fill in the inferred areas, flagged as inferred.  Overwrites nothing."""
    for entity_id, placement in result.placements.items():
        info = resolver.entities.get(entity_id)
        if info is None or info.area_id or info.area_name:
            continue
        info.area_id = placement["area_id"]
        info.area_name = placement["area_name"]
        info.floor_id, info.floor_name = _floor_of(resolver, placement["area_id"])
        info.area_inferred = True
        info.area_inferred_reason = placement["reason"]
        result.applied += 1
    if result.applied:
        _LOGGER.info(
            "Inferred an area for %d entities the registry left unplaced "
            "(each marked as a guess)",
            result.applied,
        )
    return result


def _floor_of(resolver, area_id: str) -> tuple[str | None, str | None]:
    area = resolver.registries.areas.get(area_id)
    if area is None or not area.floor_id:
        return None, None
    return area.floor_id, resolver.registries.floors.get(area.floor_id)
