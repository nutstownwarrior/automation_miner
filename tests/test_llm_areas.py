"""Inferring a room for entities the registry never placed.

Area is the only structural fact this add-on has about a home, and it is the one
users most often leave half-filled. The name is usually in the entity id, which
is a reading task - but reading is also where a model invents, so this feature
has the narrowest mandate here: it sees only entities with no area, may only
answer with a room that already exists, and everything it places is flagged as
a guess.
"""

from __future__ import annotations

from amminer.discovery.registry import Registries, RegistryArea, RegistryEntity
from amminer.entities import EntityResolver
from amminer.llm import areas as areas_mod
from amminer.llm.provider import LLMError, NullProvider


class StubLLM(NullProvider):
    name, enabled = "stub", True

    def __init__(self, payload=None, raises=False):
        self.payload = payload or {}
        self.raises = raises
        self.prompts: list[str] = []

    def complete_json(self, system, user):
        self.prompts.append(user)
        if self.raises:
            raise LLMError("stub is down")
        return self.payload


def resolver_with_areas() -> EntityResolver:
    registries = Registries(
        areas={
            "area_kitchen": RegistryArea(id="area_kitchen", name="Kitchen",
                                         floor_id="floor_ground"),
            "area_bed": RegistryArea(id="area_bed", name="Bedroom"),
        },
        floors={"floor_ground": "Ground floor"},
        entities={
            "light.hue_kitchen_1": RegistryEntity(entity_id="light.hue_kitchen_1"),
            "binary_sensor.0x00158d0001": RegistryEntity(
                entity_id="binary_sensor.0x00158d0001"),
            "light.bed_lamp": RegistryEntity(entity_id="light.bed_lamp",
                                             area_id="area_bed"),
        },
        source="storage",
    )
    return EntityResolver(registries)


def placement(entity_id="light.hue_kitchen_1", area="Kitchen", reason="The id says kitchen."):
    return {"entity_id": entity_id, "area": area, "reason": reason}


def _infer(*placements, resolver=None):
    resolver = resolver or resolver_with_areas()
    llm = StubLLM({"placements": list(placements)})
    return resolver, llm, areas_mod.infer(resolver, llm)


# --- what it may do ------------------------------------------------------
def test_an_unplaced_entity_gets_the_room_its_name_says():
    resolver, _llm, result = _infer(placement())
    areas_mod.apply_inferences(resolver, result)
    kitchen = resolver.get("light.hue_kitchen_1")
    assert (kitchen.area_id, kitchen.area_name) == ("area_kitchen", "Kitchen")
    assert result.applied == 1


def test_every_inference_is_marked_as_a_guess():
    """A guess must never be displayed, exported or prompted with as registry truth."""
    resolver, _llm, result = _infer(placement())
    areas_mod.apply_inferences(resolver, result)
    kitchen = resolver.get("light.hue_kitchen_1")
    assert kitchen.area_inferred is True
    assert kitchen.as_dict()["area_inferred"] is True
    assert "(guessed)" in kitchen.describe()
    assert resolver.get("light.bed_lamp").area_inferred is False


def test_the_floor_comes_along_with_the_area():
    resolver, _llm, result = _infer(placement())
    areas_mod.apply_inferences(resolver, result)
    assert resolver.get("light.hue_kitchen_1").floor_name == "Ground floor"


def test_an_area_matches_whatever_case_it_comes_back_in():
    resolver, _llm, result = _infer(placement(area="kitchen"))
    assert result.placements["light.hue_kitchen_1"]["area_name"] == "Kitchen"


# --- what it may not do --------------------------------------------------
def test_an_area_the_user_assigned_is_never_touched():
    resolver, _llm, result = _infer(placement(entity_id="light.bed_lamp", area="Kitchen"))
    areas_mod.apply_inferences(resolver, result)
    assert resolver.get("light.bed_lamp").area_name == "Bedroom"
    assert result.rejected_unknown_entity == 1


def test_a_placed_entity_is_never_even_shown_to_the_model():
    resolver, llm, _result = _infer()
    assert "light.bed_lamp" not in llm.prompts[0]
    assert "light.hue_kitchen_1" in llm.prompts[0]


def test_an_invented_room_is_refused():
    """A room the user does not have is not a place anything can be assigned to."""
    resolver, _llm, result = _infer(placement(area="Conservatory"))
    areas_mod.apply_inferences(resolver, result)
    assert result.placements == {} and result.rejected_unknown_area == 1
    assert resolver.get("light.hue_kitchen_1").area_name is None


def test_an_invented_entity_is_refused():
    resolver, _llm, result = _infer(placement(entity_id="light.does_not_exist"))
    assert result.placements == {} and result.rejected_unknown_entity == 1


def test_with_no_areas_defined_nothing_is_proposed():
    """Proposing rooms would be inventing the structure this exists to read."""
    resolver = resolver_with_areas()
    resolver.registries.areas.clear()
    llm = StubLLM({"placements": [placement()]})
    result = areas_mod.infer(resolver, llm)
    assert result.placements == {} and llm.prompts == []
    assert "no areas" in result.error


def test_an_entity_placed_while_the_model_was_thinking_is_refused():
    """The no-overwrite rule is re-checked against the reply, not just the prompt."""
    resolver = resolver_with_areas()

    class PlacesItFirst(StubLLM):
        def complete_json(self, system, user):
            resolver.get("light.hue_kitchen_1").area_id = "area_bed"
            return super().complete_json(system, user)

    result = areas_mod.infer(resolver, PlacesItFirst({"placements": [placement()]}))
    assert result.placements == {} and result.rejected_already_placed == 1


def test_a_provider_failure_places_nothing():
    resolver = resolver_with_areas()
    result = areas_mod.infer(resolver, StubLLM(raises=True))
    assert result.placements == {} and "stub is down" in result.error


def test_a_malformed_reply_places_nothing():
    resolver = resolver_with_areas()
    assert areas_mod.infer(resolver, StubLLM({"placements": 3})).placements == {}


def test_applying_twice_places_nothing_new():
    resolver, _llm, result = _infer(placement())
    areas_mod.apply_inferences(resolver, result)
    areas_mod.apply_inferences(resolver, result)
    assert result.applied == 1


def test_two_areas_with_the_same_name_are_not_guessed_between():
    """The model sees one name; writing it to an arbitrary one of the two is a guess."""
    resolver = resolver_with_areas()
    resolver.registries.areas["area_kitchen_2"] = RegistryArea(
        id="area_kitchen_2", name="kitchen "
    )
    llm = StubLLM({"placements": [placement()]})
    result = areas_mod.infer(resolver, llm)
    import json as _json

    assert _json.loads(llm.prompts[0])["areas"] == ["Bedroom"]
    assert result.placements == {} and result.rejected_unknown_area == 1


def test_the_reason_a_guess_was_made_is_readable():
    resolver, _llm, result = _infer(placement(reason="The entity id says kitchen."))
    areas_mod.apply_inferences(resolver, result)
    info = resolver.get("light.hue_kitchen_1")
    assert info.as_dict()["area_inferred_reason"] == "The entity id says kitchen."


def test_applying_a_placement_for_an_already_placed_entity_is_refused():
    """The guard inside `apply_inferences` is its own line of defence, so test it directly."""
    resolver = resolver_with_areas()
    result = areas_mod.AreaResult()
    result.placements["light.bed_lamp"] = {
        "area_id": "area_kitchen", "area_name": "Kitchen", "reason": "wrong",
    }
    areas_mod.apply_inferences(resolver, result)
    lamp = resolver.get("light.bed_lamp")
    assert (lamp.area_name, lamp.area_inferred, result.applied) == ("Bedroom", False, 0)
