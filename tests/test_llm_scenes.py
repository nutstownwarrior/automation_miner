"""Consolidating several suggestions into one named scene.

Six cards that all say "at about 22:40" are one habit split six ways, and the
miners have no vocabulary for that. A model does - but "these belong together"
is a claim about a rule that has never been measured, so the consolidated rule
is backtested as a unit and inherits nothing from its parts. These tests are
mostly about what happens when the grouping is wrong.
"""

from __future__ import annotations

import datetime as dt

import pytest
from amminer.config import Options
from amminer.enrich.signals import SignalStore
from amminer.llm import scenes as scenes_mod
from amminer.llm.provider import LLMError, NullProvider
from amminer.miners.base import Action, Candidate, Condition, Trigger
from amminer.recorderdb.models import Cause, StateChange
from amminer.util.timeutil import local_tz

TZ = local_tz()
START = dt.datetime(2024, 3, 1, tzinfo=TZ)
DAYS = 28
WINDOW = (START.timestamp(), (START + dt.timedelta(days=DAYS)).timestamp())


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


def at(hour, minute, entity, service="light.turn_off", title=None, **kwargs):
    return Candidate(
        miner="time_of_day",
        title=title or f"{service} {entity} at {hour:02d}:{minute:02d}",
        triggers=[Trigger(kind="time", at=f"{hour:02d}:{minute:02d}:00")],
        actions=[Action(service=service, entity_id=entity)],
        score=0.9,
        **kwargs,
    )


@pytest.fixture
def bedtime():
    """Three lights switched off within a few minutes every night."""
    changes: list[StateChange] = []
    for day in range(DAYS):
        for minutes, entity in ((0, "light.kitchen"), (2, "light.hall"), (4, "light.porch")):
            change = StateChange(
                entity, "off",
                (START + dt.timedelta(days=day, hours=22, minutes=minutes)).timestamp(),
                old_state="on",
            )
            change.cause = Cause.HUMAN
            changes.append(change)
    members = [
        at(22, 0, "light.kitchen"),
        at(22, 2, "light.hall"),
        at(22, 4, "light.porch"),
    ]
    return changes, SignalStore(), Options(llm_scenes=True), members


def _run(bedtime, payload):
    changes, store, options, members = bedtime
    return members, scenes_mod.propose_and_verify(
        members, changes, store, options, WINDOW, StubLLM(payload)
    )


# --- the shared trigger is decided by comparing triggers, not by the model ---
def test_a_real_grouping_is_consolidated_and_measured(bedtime):
    members, result = _run(bedtime, {"scenes": [
        {"name": "Bedtime", "members": [c.id for c in bedtime[3]],
         "reason": "The house is shut down for the night."},
    ]})
    assert len(result.accepted) == 1
    scene = result.accepted[0]
    assert scene.title == "Bedtime"
    assert {a.entity_id for a in scene.actions} == {
        "light.kitchen", "light.hall", "light.porch"
    }
    assert len(scene.triggers) == 1
    assert scene.backtest["passed"] is True


def test_the_scene_does_not_inherit_its_members_scores(bedtime):
    """It is a different rule from any of its parts, so it is scored from zero."""
    members, result = _run(bedtime, {"scenes": [
        {"name": "Bedtime", "members": [c.id for c in bedtime[3]], "reason": "r"},
    ]})
    assert all(m.score == 0.9 for m in bedtime[3])
    assert result.accepted[0].score <= 0.5


def test_triggers_further_apart_than_the_backtester_can_credit_are_refused(bedtime):
    changes, store, options, _ = bedtime
    far = [at(7, 0, "light.kitchen"), at(22, 0, "light.hall")]
    result = scenes_mod.propose_and_verify(
        far, changes, store, options, WINDOW,
        StubLLM({"scenes": [{"name": "Nope", "members": [c.id for c in far], "reason": "r"}]}),
    )
    assert result.accepted == [] and result.rejected_incompatible == 1


def test_the_compatibility_window_is_the_backtesters_own_tolerance(bedtime):
    """Otherwise a scene's parts get measured against a moment they never happened at."""
    changes, store, _options, _ = bedtime
    pair = [at(22, 0, "light.kitchen"), at(22, 20, "light.hall")]
    payload = {"scenes": [{"name": "S", "members": [c.id for c in pair], "reason": "r"}]}

    tight = scenes_mod.propose_and_verify(
        pair, changes, store, Options(backtest_match_tolerance_seconds=300),
        WINDOW, StubLLM(payload),
    )
    loose = scenes_mod.propose_and_verify(
        pair, changes, store, Options(backtest_match_tolerance_seconds=3600),
        WINDOW, StubLLM(payload),
    )
    assert tight.rejected_incompatible == 1
    assert loose.rejected_incompatible == 0


def test_mixed_trigger_kinds_have_no_single_stand_in(bedtime):
    changes, store, options, _ = bedtime
    mixed = [
        at(22, 0, "light.kitchen"),
        Candidate(miner="m", title="On motion", triggers=[
            Trigger(kind="state", entity_id="binary_sensor.hall", to_state="on")],
            actions=[Action(service="light.turn_on", entity_id="light.hall")]),
    ]
    result = scenes_mod.propose_and_verify(
        mixed, changes, store, options, WINDOW,
        StubLLM({"scenes": [{"name": "X", "members": [c.id for c in mixed], "reason": "r"}]}),
    )
    assert result.rejected_incompatible == 1


def test_a_state_trigger_group_keeps_the_strictest_hold():
    """A scene must never fire sooner than its strictest member would have."""
    members = [
        Candidate(miner="m", title="a", triggers=[
            Trigger(kind="state", entity_id="binary_sensor.hall", to_state="on",
                    for_seconds=30)],
            actions=[Action(service="light.turn_on", entity_id="light.a")]),
        Candidate(miner="m", title="b", triggers=[
            Trigger(kind="state", entity_id="binary_sensor.hall", to_state="on",
                    for_seconds=120)],
            actions=[Action(service="light.turn_on", entity_id="light.b")]),
    ]
    assert scenes_mod.shared_trigger(members, 900).for_seconds == 120


def test_the_shared_time_is_the_median_not_the_extreme():
    members = [at(22, 0, "light.a"), at(22, 2, "light.b"), at(22, 4, "light.c")]
    assert scenes_mod.shared_trigger(members, 900).at == "22:02:00"


# --- the rest of the checks ---------------------------------------------
def test_a_group_whose_members_fight_is_refused(bedtime):
    changes, store, options, _ = bedtime
    pair = [
        at(22, 0, "light.kitchen", service="light.turn_off"),
        at(22, 2, "light.kitchen", service="light.turn_on"),
    ]
    result = scenes_mod.propose_and_verify(
        pair, changes, store, options, WINDOW,
        StubLLM({"scenes": [{"name": "X", "members": [c.id for c in pair], "reason": "r"}]}),
    )
    assert result.accepted == [] and result.rejected_contradictory == 1


def test_a_consolidation_that_fails_the_gate_is_not_surfaced(bedtime):
    """A grouping is a claim about an unmeasured rule, and the gate decides."""
    changes, store, options, members = bedtime
    never = at(3, 0, "light.attic")
    never_2 = at(3, 2, "light.cellar")
    result = scenes_mod.propose_and_verify(
        [never, never_2], changes, store, options, WINDOW,
        StubLLM({"scenes": [{"name": "Ghost", "members": [never.id, never_2.id],
                             "reason": "r"}]}),
    )
    assert result.accepted == [] and result.rejected_by_backtest == 1


def test_invented_member_ids_are_dropped(bedtime):
    changes, store, options, members = bedtime
    result = scenes_mod.propose_and_verify(
        members, changes, store, options, WINDOW,
        StubLLM({"scenes": [{"name": "X", "members": ["nope", "also-nope"], "reason": "r"}]}),
    )
    assert result.accepted == [] and result.rejected_unknown_members == 1


def test_a_single_member_is_not_a_scene(bedtime):
    changes, store, options, members = bedtime
    result = scenes_mod.propose_and_verify(
        members, changes, store, options, WINDOW,
        StubLLM({"scenes": [{"name": "X", "members": [members[0].id], "reason": "r"}]}),
    )
    assert result.accepted == []


def test_a_member_cannot_be_spent_on_two_scenes(bedtime):
    changes, store, options, members = bedtime
    ids = [c.id for c in members]
    result = scenes_mod.propose_and_verify(
        members, changes, store, options, WINDOW,
        StubLLM({"scenes": [
            {"name": "First", "members": ids, "reason": "r"},
            {"name": "Second", "members": ids, "reason": "r"},
        ]}),
    )
    assert len(result.accepted) == 1


def test_only_conditions_every_member_carries_survive(bedtime):
    """A condition one member alone required would gate the others on nothing."""
    changes, store, options, _ = bedtime
    shared = Condition(kind="time", weekday=["mon"])
    private = Condition(kind="state", entity_id="person.alex", state="home")
    pair = [
        at(22, 0, "light.kitchen", conditions=[shared, private]),
        at(22, 2, "light.hall", conditions=[shared]),
    ]
    trigger = scenes_mod.shared_trigger(pair, 900)
    scene = scenes_mod._consolidate("Bedtime", "r", pair, trigger)
    assert [c.as_dict() for c in scene.conditions] == [shared.as_dict()]


def test_members_are_annotated_and_never_removed(bedtime):
    members, result = _run(bedtime, {"scenes": [
        {"name": "Bedtime", "members": [c.id for c in bedtime[3]], "reason": "r"},
    ]})
    scenes_mod.apply_scenes(members, result)
    assert all(m.extra["part_of_scene"]["name"] == "Bedtime" for m in members)
    assert len(members) == 3


def test_a_provider_failure_groups_nothing(bedtime):
    changes, store, options, members = bedtime
    result = scenes_mod.propose_and_verify(
        members, changes, store, options, WINDOW, StubLLM(raises=True)
    )
    assert result.accepted == [] and "stub is down" in result.error


def test_a_malformed_reply_groups_nothing(bedtime):
    changes, store, options, members = bedtime
    result = scenes_mod.propose_and_verify(
        members, changes, store, options, WINDOW, StubLLM({"scenes": "nope"})
    )
    assert result.accepted == []
