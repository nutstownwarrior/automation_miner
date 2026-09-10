"""The one module that writes to a user's live configuration.

Everything here was covered only indirectly, through a single web-layer test.
That test happens to exercise the refusal path; nothing exercised what happens
when the write succeeds and the reload does not, or when there is no client at
all, or when the automation has no id.
"""

from __future__ import annotations

import pytest
from amminer.apply import apply_generation
from amminer.llm.generate import GenerationResult
from amminer.llm.validate import ValidationReport

GOOD_CONFIG = {
    "id": "amminer_abc123",
    "alias": "Kitchen light on weekday mornings",
    "trigger": [{"platform": "time", "at": "06:30:00"}],
    "action": [{"service": "light.turn_on", "target": {"entity_id": "light.kitchen"}}],
}


def _passing(config: dict | None = None) -> GenerationResult:
    generation = GenerationResult()
    generation.config = dict(config or GOOD_CONFIG)
    generation.report = ValidationReport(
        ok=True, schema_ok=True, references_ok=True, check_config_ok=True
    )
    return generation


class FakeClient:
    """A Core API that records what it was asked to do."""

    def __init__(self, *, configured=True, write_ok=True, reload_ok=True):
        self.configured = configured
        self.write_ok = write_ok
        self.reload_ok = reload_ok
        self.written: dict[str, dict] = {}
        self.reloads = 0
        self.last_error = "something went wrong"

    def upsert_automation(self, automation_id, body):
        if not self.write_ok:
            return False
        self.written[automation_id] = body
        return True

    def reload_automations(self):
        if not self.reload_ok:
            return False
        self.reloads += 1
        return True


def test_a_validated_automation_is_written_and_reloaded():
    client = FakeClient()
    result = apply_generation(_passing(), client)

    assert result.ok is True
    assert result.written is True
    assert result.reloaded is True
    assert result.automation_id == "amminer_abc123"
    assert result.errors == []
    # The id travels in the URL, per Home Assistant's config API.
    assert "id" not in client.written["amminer_abc123"]
    assert client.written["amminer_abc123"]["alias"] == GOOD_CONFIG["alias"]


@pytest.mark.parametrize(
    "generation",
    [
        pytest.param(GenerationResult(), id="nothing generated"),
        pytest.param(
            GenerationResult(
                config=dict(GOOD_CONFIG),
                report=ValidationReport(ok=False, errors=["entity does not exist"]),
            ),
            id="failed the gate",
        ),
        pytest.param(
            GenerationResult(config=None, report=ValidationReport(ok=True)),
            id="passed with no config",
        ),
    ],
)
def test_nothing_that_failed_validation_is_ever_written(generation):
    client = FakeClient()
    result = apply_generation(generation, client)

    assert result.ok is False
    assert result.written is False
    assert client.written == {}
    assert any("refusing to apply" in error for error in result.errors)


def test_the_reason_it_was_refused_is_carried_through():
    generation = GenerationResult(
        config=dict(GOOD_CONFIG),
        report=ValidationReport(ok=False, errors=["entity_id 'light.nope' does not exist"]),
    )
    result = apply_generation(generation, FakeClient())
    assert "light.nope" in " ".join(result.errors)


def test_no_api_access_says_so_instead_of_failing_obscurely():
    result = apply_generation(_passing(), FakeClient(configured=False))
    assert result.ok is False
    assert result.written is False
    assert any("SUPERVISOR_TOKEN" in error for error in result.errors)


def test_no_client_at_all_is_the_same_answer():
    result = apply_generation(_passing(), None)
    assert result.ok is False
    assert any("SUPERVISOR_TOKEN" in error for error in result.errors)


def test_an_automation_without_an_id_is_refused():
    config = {k: v for k, v in GOOD_CONFIG.items() if k != "id"}
    client = FakeClient()
    result = apply_generation(_passing(config), client)

    assert result.ok is False
    assert client.written == {}
    assert any("no id" in error for error in result.errors)


def test_a_failed_write_reports_the_client_error_and_stops():
    client = FakeClient(write_ok=False)
    client.last_error = "403 Forbidden"
    result = apply_generation(_passing(), client)

    assert result.ok is False
    assert result.written is False
    assert result.reloaded is False
    assert client.reloads == 0  # nothing to reload
    assert "403 Forbidden" in " ".join(result.errors)


def test_a_written_automation_that_could_not_be_reloaded_is_still_applied():
    """The user's config has changed; telling them it failed would be wrong."""
    client = FakeClient(reload_ok=False)
    client.last_error = "service call timed out"
    result = apply_generation(_passing(), client)

    assert result.written is True
    assert result.reloaded is False
    assert result.ok is True  # it IS in their configuration now
    assert result.errors == []
    assert any("reload manually" in note for note in result.notes)
    assert any("service call timed out" in note for note in result.notes)


def test_the_result_serialises_for_the_ui():
    result = apply_generation(_passing(), FakeClient())
    data = result.as_dict()
    assert data["ok"] is True
    assert data["automation_id"] == "amminer_abc123"
    assert set(data) == {"ok", "written", "reloaded", "automation_id", "errors", "notes"}
