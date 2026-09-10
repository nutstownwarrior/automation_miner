"""The validation gate: hallucinated references and bad schemas must be blocked."""

from __future__ import annotations

import pytest
import yaml
from amminer.config import Options
from amminer.entities import EntityResolver
from amminer.llm.blueprint import automation_id, candidate_to_automation, render_yaml
from amminer.llm.generate import build_prompt, generate
from amminer.llm.provider import (
    LLMError,
    NullProvider,
    OllamaProvider,
    _parse_json_object,
    build_provider,
    pick_model,
)
from amminer.llm.validate import validate_automation, validate_references, validate_schema
from amminer.miners.base import Action, Candidate, Condition, Trigger

SERVICES = {"light.turn_on", "light.turn_off", "switch.turn_on", "climate.set_hvac_mode"}


@pytest.fixture
def resolver(states_payload) -> EntityResolver:
    resolver = EntityResolver()
    resolver.merge_states(states_payload)
    return resolver


@pytest.fixture
def good_candidate() -> Candidate:
    return Candidate(
        miner="time_of_day",
        title="Turn on the kitchen light at 06:30 on weekdays",
        triggers=[Trigger(kind="time", at="06:30:00")],
        conditions=[Condition(kind="time", weekday=["mon", "tue", "wed", "thu", "fri"])],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )


# --- deterministic rendering -------------------------------------------
def test_blueprint_renders_valid_automation(good_candidate, resolver):
    config = candidate_to_automation(good_candidate, resolver)
    assert config["trigger"] == [{"platform": "time", "at": "06:30:00"}]
    assert config["condition"][0]["weekday"] == ["mon", "tue", "wed", "thu", "fri"]
    assert config["action"][0]["target"]["entity_id"] == "light.kitchen"
    assert config["id"] == automation_id(good_candidate)
    parsed = yaml.safe_load(render_yaml(config))
    assert isinstance(parsed, list) and parsed[0]["alias"]


def test_blueprint_id_is_stable_across_runs(good_candidate):
    other = Candidate(
        miner="time_of_day",
        title="a completely different title",
        triggers=[Trigger(kind="time", at="06:30:00")],
        conditions=[Condition(kind="time", weekday=["mon", "tue", "wed", "thu", "fri"])],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
    )
    # The identity is the rule, not the wording.
    assert automation_id(good_candidate) == automation_id(other)


def test_blueprint_refuses_actionless_candidates():
    with pytest.raises(ValueError):
        candidate_to_automation(Candidate(miner="x", title="y", triggers=[Trigger("time", at="1")]))


# --- reference validation ----------------------------------------------
def test_hallucinated_entity_is_rejected(resolver):
    config = {
        "alias": "bad",
        "trigger": [{"platform": "time", "at": "06:30:00"}],
        "action": [{"service": "light.turn_on", "target": {"entity_id": "light.does_not_exist"}}],
    }
    report = validate_automation(config, resolver=resolver, known_services=SERVICES, run_check_config=False)
    assert report.ok is False
    assert report.references_ok is False
    assert "light.does_not_exist" in report.unknown_entities


def test_hallucinated_service_is_rejected(resolver):
    config = {
        "alias": "bad",
        "trigger": [{"platform": "time", "at": "06:30:00"}],
        "action": [{"service": "light.make_coffee", "target": {"entity_id": "light.kitchen"}}],
    }
    report = validate_automation(config, resolver=resolver, known_services=SERVICES, run_check_config=False)
    assert report.ok is False
    assert "light.make_coffee" in report.unknown_services


def test_hallucinated_area_and_device_rejected(resolver):
    config = {
        "alias": "bad",
        "trigger": [{"platform": "time", "at": "06:30:00"}],
        "action": [{"service": "light.turn_on", "target": {"area_id": "area_atlantis"}}],
    }
    ok, errors, unknown = validate_references(
        config, resolver.known_entity_ids(), SERVICES, known_areas={"area_kitchen"}
    )
    assert ok is False
    assert "area_atlantis" in unknown["targets"]


def test_valid_references_pass(resolver, good_candidate):
    config = candidate_to_automation(good_candidate, resolver)
    report = validate_automation(config, resolver=resolver, known_services=SERVICES, run_check_config=False)
    assert report.references_ok is True
    assert report.schema_ok is True
    assert report.ok is True


def test_trigger_platform_named_action_is_not_a_service(resolver):
    """`action:` inside a trigger names a platform, not a service call."""
    config = {
        "alias": "ok",
        "trigger": [{"platform": "time", "at": "06:30:00"}],
        "action": [{"action": "light.turn_on", "target": {"entity_id": "light.kitchen"}}],
    }
    report = validate_automation(config, resolver=resolver, known_services=SERVICES, run_check_config=False)
    assert report.ok is True


# --- schema validation --------------------------------------------------
@pytest.mark.parametrize(
    "config",
    [
        {"trigger": [{"platform": "time", "at": "1"}], "action": [{"service": "light.turn_on"}]},  # no alias
        {"alias": "x", "action": [{"service": "light.turn_on"}]},  # no trigger
        {"alias": "x", "trigger": [{"platform": "time"}]},  # no action
        {"alias": "x", "trigger": [], "action": [{"service": "light.turn_on"}]},  # empty trigger
        {"alias": "x", "trigger": [{"platform": "time"}], "action": [], "mode": "single"},  # empty action
        {"alias": "x", "trigger": [{"platform": "time"}], "action": [{"service": "a.b"}], "mode": "nonsense"},
    ],
)
def test_invalid_schemas_are_rejected(config):
    ok, errors = validate_schema(config)
    assert ok is False and errors


def test_broken_yaml_is_rejected(resolver):
    report = validate_automation("alias: [unclosed\n", resolver=resolver, run_check_config=False)
    assert report.ok is False
    assert any("YAML" in e for e in report.errors)


def test_yaml_string_input_is_accepted(resolver, good_candidate):
    text = render_yaml(candidate_to_automation(good_candidate, resolver))
    report = validate_automation(text, resolver=resolver, known_services=SERVICES, run_check_config=False)
    assert report.ok is True


# --- check_config -------------------------------------------------------
def test_check_config_failure_blocks(resolver, good_candidate, fake_client):
    fake_client.check_config_result = "invalid"
    config = candidate_to_automation(good_candidate, resolver)
    report = validate_automation(config, resolver=resolver, client=fake_client, known_services=SERVICES)
    assert report.check_config_ok is False
    assert report.ok is False


def test_check_config_unavailable_is_a_warning_not_a_block(resolver, good_candidate, fake_client):
    fake_client.check_config_result = "unavailable"
    config = candidate_to_automation(good_candidate, resolver)
    report = validate_automation(config, resolver=resolver, client=fake_client, known_services=SERVICES)
    assert report.check_config_ok is None
    assert report.ok is True
    assert report.warnings


def test_no_resolver_means_no_validation():
    report = validate_automation(
        {"alias": "x", "trigger": [{"platform": "time", "at": "1"}],
         "action": [{"service": "light.turn_on"}]},
        run_check_config=False,
    )
    assert report.ok is False
    assert any("resolver" in e for e in report.errors)


# --- generation ---------------------------------------------------------
def test_generate_without_llm_uses_the_blueprint(resolver, good_candidate, fake_client):
    result = generate(good_candidate, resolver, NullProvider(), fake_client, sorted(SERVICES))
    assert result.source == "blueprint"
    assert result.ok is True
    assert "light.kitchen" in result.yaml_text


def test_generate_accepts_good_llm_output(resolver, good_candidate, fake_client):
    class GoodLLM(NullProvider):
        name = "fake"
        enabled = True

        def complete_json(self, system, user):
            return {
                "alias": "Kitchen light on weekday mornings",
                "description": "nicely worded",
                "mode": "single",
                "triggers": [{"platform": "time", "at": "06:30:00"}],
                "conditions": [{"condition": "time", "weekday": ["mon", "tue", "wed", "thu", "fri"]}],
                "actions": [{"service": "light.turn_on", "target": {"entity_id": "light.kitchen"}}],
            }

    result = generate(good_candidate, resolver, GoodLLM(), fake_client, sorted(SERVICES))
    assert result.source == "llm"
    assert result.ok is True
    assert result.config["alias"] == "Kitchen light on weekday mornings"
    # Plural keys must be normalised to what HA's config API expects.
    assert "trigger" in result.config and "triggers" not in result.config


def test_hallucinating_llm_is_rejected_and_falls_back(resolver, good_candidate, fake_client):
    class HallucinatingLLM(NullProvider):
        name = "fake"
        enabled = True

        def complete_json(self, system, user):
            return {
                "alias": "Kitchen light",
                "trigger": [{"platform": "time", "at": "06:30:00"}],
                "action": [
                    {"service": "light.turn_on", "target": {"entity_id": "light.imaginary_lamp"}}
                ],
            }

    result = generate(good_candidate, resolver, HallucinatingLLM(), fake_client, sorted(SERVICES))
    assert result.source == "llm-rejected"
    assert "light.imaginary_lamp" in result.llm_report.unknown_entities
    # The user still gets a working rule from the deterministic renderer.
    assert result.ok is True
    assert "light.kitchen" in result.yaml_text
    assert any("REJECTED" in note for note in result.notes)


def test_llm_error_falls_back_cleanly(resolver, good_candidate, fake_client):
    class BrokenLLM(NullProvider):
        name = "fake"
        enabled = True

        def complete_json(self, system, user):
            raise LLMError("connection refused")

    result = generate(good_candidate, resolver, BrokenLLM(), fake_client, sorted(SERVICES))
    assert result.source == "blueprint"
    assert result.ok is True
    assert "connection refused" in result.llm_error


def test_prompt_never_contains_raw_history(resolver, good_candidate):
    good_candidate.evidence.samples = [1_700_000_000.0 + i for i in range(50)]
    prompt = build_prompt(good_candidate, resolver, sorted(SERVICES))
    assert "1700000000" not in prompt
    assert "light.kitchen" in prompt
    assert "allowed_entities" in prompt and "allowed_services" in prompt


# --- providers ----------------------------------------------------------
def test_provider_defaults_to_none():
    assert isinstance(build_provider(Options()), NullProvider)


def test_cloud_provider_needs_an_api_key():
    provider = build_provider(Options(llm_provider="openai"))
    assert isinstance(provider, NullProvider)  # opt-in means key-in


def test_cloud_provider_with_key_is_built():
    from amminer.llm.provider import CloudProvider

    provider = build_provider(Options(llm_provider="openai", llm_api_key="sk-test"))
    assert isinstance(provider, CloudProvider)
    assert provider.status().available is True


def test_unknown_provider_degrades_to_none():
    assert isinstance(build_provider(Options(llm_provider="hal9000")), NullProvider)


def test_pick_model_prefers_recommended_and_skips_think_variants():
    assert pick_model(["llama2:7b", "qwen3:8b"]) == "qwen3:8b"
    assert pick_model(["deepseek-r1:8b", "gemma3:4b"]) == "gemma3:4b"
    assert pick_model([], None) is None
    assert pick_model(["anything"], "explicit:1b") == "explicit:1b"


@pytest.mark.parametrize(
    "raw",
    [
        '{"alias": "x"}',
        '```json\n{"alias": "x"}\n```',
        'Sure! Here is the JSON:\n{"alias": "x"}\nHope that helps.',
    ],
)
def test_json_parsing_tolerates_model_chatter(raw):
    assert _parse_json_object(raw)["alias"] == "x"


def test_json_parsing_raises_on_garbage():
    with pytest.raises(LLMError):
        _parse_json_object("no json here at all")


def test_ollama_unavailable_reports_a_helpful_status(monkeypatch):
    monkeypatch.setattr("amminer.llm.provider.discover_ollama", lambda **kw: (None, []))
    status = OllamaProvider().status()
    assert status.available is False
    assert "Ollama" in status.error


# --- bypasses ----------------------------------------------------------
# Three ways an automation reached the end of the gate without its effect ever
# having been checked.  Each of these is a real config Home Assistant accepts.
def _wrap(action: dict) -> dict:
    return {
        "alias": "x",
        "trigger": [{"platform": "time", "at": "06:30:00"}],
        "action": [action],
    }


@pytest.mark.parametrize(
    "name, action",
    [
        # Names no service at all, so there is nothing to look up.
        ("bare template", {"service": "{{ svc }}", "target": {"entity_id": "light.kitchen"}}),
        # Half a name: the domain is real, the service is chosen at runtime.
        ("half template", {"service": "light.{{ s }}", "target": {"entity_id": "light.kitchen"}}),
        # Names no entity, so the existence check passes vacuously.
        ("templated target", {"service": "light.turn_on",
                              "target": {"entity_id": "{{ trigger.entity_id }}"}}),
        # Legal HA, and it means every light in the house.
        ("wildcard shorthand", {"service": "light.turn_off", "entity_id": "all"}),
        ("wildcard in a list", {"service": "light.turn_off",
                                "target": {"entity_id": ["all"]}}),
    ],
)
def test_runtime_decided_targets_are_refused(resolver, name, action):
    report = validate_automation(
        _wrap(action), resolver=resolver, known_services=SERVICES, run_check_config=False
    )
    assert report.ok is False, name
    assert report.targets_static is False, name


def test_prose_may_contain_braces(resolver):
    """Only the parts that decide behaviour are held to this."""
    config = _wrap({"service": "light.turn_on", "target": {"entity_id": "light.kitchen"}})
    config["alias"] = "Kitchen light {{ not a template, just words }}"
    config["description"] = "uses {% raw %} in the docs"
    report = validate_automation(
        config, resolver=resolver, known_services=SERVICES, run_check_config=False
    )
    assert report.targets_static is True
    assert report.ok is True


def test_an_unreachable_core_does_not_wave_services_through(resolver):
    """No service index is a reason to check harder, not to stop checking."""
    config = _wrap({"service": "shell_command.wipe", "target": {"entity_id": "light.kitchen"}})
    report = validate_automation(
        config, resolver=resolver, known_services=set(), run_check_config=False
    )
    assert report.services_verified is False
    assert report.references_ok is False
    assert report.ok is False
    assert "shell_command.wipe" in report.unknown_services


def test_the_fallback_still_admits_what_this_addon_generates(resolver):
    """Degradation must not block the deterministic renderer's own output."""
    config = _wrap({"service": "light.turn_on", "target": {"entity_id": "light.kitchen"}})
    report = validate_automation(
        config, resolver=resolver, known_services=set(), run_check_config=False
    )
    assert report.services_verified is False
    assert report.ok is True
    assert any("service list is unavailable" in w for w in report.warnings)


def test_every_service_a_miner_can_emit_is_in_the_fallback_set():
    """The fallback is only safe while it covers what service_for() returns."""
    from amminer.config import ACTIONABLE_DOMAINS
    from amminer.llm.validate import emittable_services
    from amminer.miners.time_of_day import service_for

    allowed = emittable_services()
    states = ("on", "off", "open", "closed", "locked", "unlocked", "playing",
              "cleaning", "docked", "heat", "cool", "option_a")
    for domain in ACTIONABLE_DOMAINS:
        for state in states:
            emitted = service_for(f"{domain}.thing", state)
            if emitted is not None:
                assert emitted[0] in allowed, f"{domain}/{state} -> {emitted[0]}"


def test_a_cloud_key_never_reaches_an_error_message():
    """Error strings from here are stored in the database and rendered in the UI."""
    from amminer.llm.provider import CloudProvider

    secret = "AIzaSy-NOT-A-REAL-KEY-000000"
    provider = CloudProvider("google", secret, model="gemini-2.0-flash")
    with pytest.raises(LLMError) as raised:
        provider.complete_json("system", "user")
    assert secret not in str(raised.value)


def test_google_authenticates_with_a_header_not_a_query_string():
    from amminer.llm.provider import CloudProvider

    provider = CloudProvider("google", "k", model="gemini-2.0-flash")
    captured = {}

    def fake_post(url, json, headers, timeout):  # noqa: A002
        captured["url"] = url
        captured["headers"] = headers
        raise RuntimeError("stop here")

    import httpx

    original = httpx.post
    httpx.post = fake_post
    try:
        with pytest.raises(RuntimeError):
            provider.complete_json("system", "user")
    finally:
        httpx.post = original
    assert "key=" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "k"
