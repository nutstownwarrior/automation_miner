"""The equivalence gate: the model may reword a rule, never replace it.

Reference validation only proves that everything an automation names exists.
A completely different automation built from real entities and real services
passes that check, so these tests pin the stronger property: what gets written
is the rule whose evidence the user actually read.
"""

from __future__ import annotations

import pytest
from amminer.config import Options
from amminer.discovery.ha_config import HAConfig
from amminer.entities import EntityResolver
from amminer.llm.blueprint import candidate_to_automation
from amminer.llm.equivalence import differences, matches, semantic_form
from amminer.llm.generate import generate
from amminer.llm.provider import NullProvider
from amminer.miners.base import Action, Candidate, Condition, Trigger
from amminer.runner import Runner, candidate_from_payload, generation_digest

SERVICES = ["light.turn_off", "light.turn_on", "lock.unlock", "switch.turn_on"]


@pytest.fixture
def resolver(states_payload) -> EntityResolver:
    resolver = EntityResolver()
    resolver.merge_states(states_payload)
    return resolver


@pytest.fixture
def candidate() -> Candidate:
    return Candidate(
        miner="time_of_day",
        title="Turn on the kitchen light at 06:30 on weekdays",
        triggers=[Trigger(kind="time", at="06:30:00")],
        conditions=[Condition(kind="time", weekday=["mon", "tue", "wed", "thu", "fri"])],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )


def _llm(reply):
    class StubLLM(NullProvider):
        name = "fake"
        enabled = True

        def complete_json(self, system, user):
            return reply

    return StubLLM()


def _blueprint(candidate, resolver) -> dict:
    return candidate_to_automation(candidate, resolver)


# --- the canonical form -------------------------------------------------
def test_cosmetic_rewrites_are_not_divergences():
    """Spelling differences HA itself accepts must not read as a new rule."""
    a = {
        "trigger": [{"platform": "state", "entity_id": "binary_sensor.workday_sensor", "to": "on"}],
        "action": [{"service": "climate.set_temperature", "entity_id": "climate.living_room",
                    "data": {"temperature": 21}}],
        "mode": "single",
    }
    b = {
        # 2024.10 key spellings, target instead of shorthand, float setpoint,
        # and no explicit mode (HA defaults to single).
        "triggers": [{"trigger": "state", "entity_id": ["binary_sensor.workday_sensor"], "to": "on"}],
        "actions": [{"action": "climate.set_temperature",
                     "target": {"entity_id": "climate.living_room"},
                     "data": {"temperature": 21.0}}],
    }
    assert matches(a, b), differences(a, b)


def test_entity_order_within_one_action_is_not_a_divergence():
    a = {"action": [{"service": "light.turn_on",
                     "target": {"entity_id": ["light.kitchen", "light.hallway"]}}]}
    b = {"action": [{"service": "light.turn_on",
                     "target": {"entity_id": ["light.hallway", "light.kitchen"]}}]}
    assert matches(a, b)


@pytest.mark.parametrize(
    "field, mutation",
    [
        ("action", {"action": [{"service": "lock.unlock",
                                "target": {"entity_id": "light.kitchen"}}]}),
        ("action", {"action": [{"service": "light.turn_on",
                                "target": {"entity_id": "light.bedroom"}}]}),
        ("trigger", {"trigger": [{"platform": "time", "at": "03:00:00"}]}),
        ("condition", {"condition": []}),
        ("mode", {"mode": "queued"}),
    ],
)
def test_every_semantic_field_is_watched(candidate, resolver, field, mutation):
    reference = _blueprint(candidate, resolver)
    produced = dict(reference)
    produced.update(mutation)
    assert differences(produced, reference) == [field]


def test_alias_and_description_are_free(candidate, resolver):
    reference = _blueprint(candidate, resolver)
    produced = dict(reference, alias="Something else entirely", description="reworded")
    assert matches(produced, reference)


# --- the gate inside generate() -----------------------------------------
def test_substituted_automation_is_rejected_even_though_it_validates(
    candidate, resolver, fake_client
):
    """Every reference is real; the rule is not the one that was mined."""
    substitution = {
        "alias": "Kitchen light on weekday mornings",
        "description": "looks exactly like what was asked for",
        "mode": "single",
        "trigger": [{"platform": "time", "at": "06:30:00"}],
        "condition": [{"condition": "time", "weekday": ["mon", "tue", "wed", "thu", "fri"]}],
        # A real service on a real entity - reference validation says yes.
        "action": [{"service": "lock.unlock", "target": {"entity_id": "light.kitchen"}}],
    }
    result = generate(candidate, resolver, _llm(substitution), fake_client, SERVICES)

    assert result.source == "llm-rejected"
    assert result.llm_report.references_ok is True  # the weak gate was fooled
    assert result.llm_report.semantics_ok is False  # this one was not
    assert any("changed the automation's action" in e for e in result.llm_report.errors)
    assert any("DIFFERENT automation" in note for note in result.notes)

    # The user still gets the rule they read the evidence for.
    assert result.ok is True
    assert result.config == _blueprint(candidate, resolver)
    assert "lock.unlock" not in result.yaml_text


def test_legitimate_rewrite_still_passes(candidate, resolver, fake_client):
    reference = _blueprint(candidate, resolver)
    reply = dict(reference, alias="Kitchen light on weekday mornings",
                 description="Because you have done this every weekday.")
    result = generate(candidate, resolver, _llm(reply), fake_client, SERVICES)

    assert result.source == "llm"
    assert result.llm_report.semantics_ok is True
    assert result.config["alias"] == "Kitchen light on weekday mornings"
    assert result.ok is True


def test_dropped_condition_is_a_divergence(candidate, resolver, fake_client):
    """Silently widening a rule is the failure mode that fires at 3am."""
    reference = _blueprint(candidate, resolver)
    reply = dict(reference, condition=[])
    result = generate(candidate, resolver, _llm(reply), fake_client, SERVICES)

    assert result.source == "llm-rejected"
    assert result.llm_report.semantics_ok is False
    assert result.config["condition"] == reference["condition"]


def test_blueprint_path_reports_no_semantics_verdict(candidate, resolver, fake_client):
    """semantics_ok compares against the blueprint, so it is meaningless there."""
    result = generate(candidate, resolver, NullProvider(), fake_client, SERVICES)
    assert result.source == "blueprint"
    assert result.llm_report is None
    assert result.report.semantics_ok is None


# --- the digest ---------------------------------------------------------
def test_digest_follows_meaning_not_wording(candidate, resolver):
    config = _blueprint(candidate, resolver)
    reworded = dict(config, alias="different words", description="also different")
    changed = dict(config, action=[{"service": "light.turn_off",
                                    "target": {"entity_id": "light.kitchen"}}])

    assert generation_digest(config) == generation_digest(reworded)
    assert generation_digest(config) != generation_digest(changed)
    assert semantic_form(config)["action"] != semantic_form(changed)["action"]


# --- preview/apply binding ----------------------------------------------
@pytest.fixture
def runner(ha_config_dir, store, fake_client):
    options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir))
    runner = Runner(options, store, fake_client)
    runner.ha_config = HAConfig(ha_config_dir)
    runner.run_now()
    return runner


def _actionable(store) -> str:
    for suggestion in store.list_suggestions(status="new"):
        if (suggestion["payload"] or {}).get("actions"):
            return suggestion["id"]
    pytest.skip("fixture produced no actionable suggestion")


def _drifting_provider(monkeypatch, runner, suggestion_id):
    """A model that answers the same question differently every time.

    Each reply is a legitimate rewrite - only the alias moves - so nothing but
    the binding between preview and apply can keep them apart.
    """
    payload = runner.store.get_suggestion(suggestion_id)["payload"]
    reference = candidate_to_automation(
        candidate_from_payload(payload), runner._ensure_resolver()
    )
    calls: list[str] = []

    class DriftingLLM(NullProvider):
        name = "fake"
        enabled = True

        def complete_json(self, system, user):
            alias = f"draft {len(calls) + 1}"
            calls.append(alias)
            return dict(reference, alias=alias)

    monkeypatch.setattr("amminer.runner.build_provider", lambda options: DriftingLLM())
    return calls


def test_apply_writes_the_artifact_that_was_previewed(runner, monkeypatch, fake_client):
    suggestion_id = _actionable(runner.store)
    calls = _drifting_provider(monkeypatch, runner, suggestion_id)

    preview = runner.preview_yaml(suggestion_id)
    assert preview["config"]["alias"] == "draft 1"
    assert preview["digest"]

    result = runner.apply(suggestion_id)
    assert result["ok"] is True, result

    # The model was asked exactly once: apply reused the reviewed artifact.
    assert calls == ["draft 1"]
    written = next(iter(fake_client.written.values()))
    assert written["alias"] == "draft 1"
    assert result["generation"]["config"]["alias"] == "draft 1"
    assert generation_digest(written) == preview["digest"]


def test_reused_artifact_is_still_revalidated_against_core(runner, monkeypatch, fake_client):
    """Reuse means no new content, not no new checks."""
    suggestion_id = _actionable(runner.store)
    _drifting_provider(monkeypatch, runner, suggestion_id)
    runner.preview_yaml(suggestion_id)

    fake_client.check_config_result = "invalid"
    result = runner.apply(suggestion_id)

    assert result["ok"] is False
    assert result["errors"]
    assert not fake_client.written


def test_a_stale_preview_is_not_applied(runner, monkeypatch, fake_client):
    """If the finding was re-mined into another rule, the old approval is void."""
    suggestion_id = _actionable(runner.store)
    calls = _drifting_provider(monkeypatch, runner, suggestion_id)
    runner.preview_yaml(suggestion_id)

    # Someone's approval is now attached to an automation this suggestion no
    # longer describes.
    runner.store.save_generation(
        suggestion_id,
        "stale",
        "llm",
        {
            "alias": "stale",
            "trigger": [{"platform": "time", "at": "23:59:00"}],
            "action": [{"service": "light.turn_on", "target": {"entity_id": "light.bedroom"}}],
        },
    )
    result = runner.apply(suggestion_id)

    assert result["ok"] is True, result
    written = next(iter(fake_client.written.values()))
    assert written["alias"] != "stale"
    assert "light.bedroom" not in str(written)
    assert len(calls) == 2  # preview, then a fresh generation for the apply
