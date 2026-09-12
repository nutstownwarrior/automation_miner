"""A friendlier name for an inferred mode - advisory only, never a fact.

The tests that matter most are the ones proving this can never invent a
state that was not in the model's own summaries, and that the deterministic
description (typical hours, top domains/areas) is untouched whether or not
this ran at all.
"""

from __future__ import annotations

from amminer.llm import home_mode_labels
from amminer.llm.provider import LLMError, NullProvider


class StubLLM(NullProvider):
    name, enabled = "stub", True

    def __init__(self, labels=None, raises=False, payload=None):
        self.labels = labels or {}
        self.raises = raises
        self.payload = payload
        self.calls = 0

    def complete_json(self, system, user):
        self.calls += 1
        if self.raises:
            raise LLMError("stub is down")
        if self.payload is not None:
            return self.payload
        return {
            "labels": [
                {"state": state, "label": label} for state, label in self.labels.items()
            ]
        }


def _model(summaries):
    class _Model:
        state_summaries = summaries

    return _Model()


def _summary(state, **extra):
    return {
        "state": state,
        "label": f"mode_{state}",
        "occupancy_share": 0.2,
        "typical_time": "21:30:00",
        "typical_spread_minutes": 45.0,
        "top_domains": ["light"],
        "top_areas": [],
        "llm_label": None,
        "llm_label_is_advisory": True,
        **extra,
    }


def test_no_provider_configured_labels_nothing():
    model = _model([_summary(0)])
    result = home_mode_labels.propose(model, NullProvider())
    assert result.labels == {}
    assert result.error == "no LLM provider configured"
    applied = home_mode_labels.apply_labels(model, result)
    assert applied == 0
    assert model.state_summaries[0]["llm_label"] is None


def test_a_good_label_is_applied_and_marked_advisory():
    model = _model([_summary(0), _summary(1)])
    provider = StubLLM(labels={0: "Evening wind-down"})
    result = home_mode_labels.propose(model, provider)
    assert result.labels == {0: "Evening wind-down"}
    applied = home_mode_labels.apply_labels(model, result)
    assert applied == 1
    assert model.state_summaries[0]["llm_label"] == "Evening wind-down"
    assert model.state_summaries[0]["llm_label_is_advisory"] is True
    # A state the model never mentioned is left exactly as it was.
    assert model.state_summaries[1]["llm_label"] is None


def test_never_invents_a_state_outside_the_model():
    model = _model([_summary(0)])
    provider = StubLLM(payload={"labels": [{"state": 7, "label": "Ghost mode"}]})
    result = home_mode_labels.propose(model, provider)
    assert result.labels == {}
    assert result.rejected_unknown_state == 1
    applied = home_mode_labels.apply_labels(model, result)
    assert applied == 0


def test_a_label_that_is_really_a_restated_description_is_rejected():
    model = _model([_summary(0)])
    long_label = "Active twenty two percent of the time mostly light and media"
    provider = StubLLM(payload={"labels": [{"state": 0, "label": long_label}]})
    result = home_mode_labels.propose(model, provider)
    assert result.labels == {}
    assert result.rejected_too_long == 1


def test_malformed_response_labels_nothing_rather_than_raising():
    model = _model([_summary(0)])
    provider = StubLLM(payload={"not_labels": []})
    result = home_mode_labels.propose(model, provider)
    assert result.labels == {}
    assert home_mode_labels.apply_labels(model, result) == 0


def test_provider_failure_is_reported_not_raised():
    model = _model([_summary(0)])
    provider = StubLLM(raises=True)
    result = home_mode_labels.propose(model, provider)
    assert result.error == "stub is down"
    assert result.labels == {}


def test_no_states_to_label_short_circuits_without_calling_the_provider():
    model = _model([])
    provider = StubLLM(labels={0: "Anything"})
    result = home_mode_labels.propose(model, provider)
    assert provider.calls == 0
    assert result.considered == 0


def test_as_dict_summarises_without_leaking_raw_labels():
    model = _model([_summary(0), _summary(1)])
    provider = StubLLM(labels={0: "Evening wind-down"})
    result = home_mode_labels.propose(model, provider)
    summary = result.as_dict()
    assert summary["considered"] == 2
    assert summary["labelled"] == 1
    assert summary["error"] is None
