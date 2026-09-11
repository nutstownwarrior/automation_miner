"""Gap proposals from world knowledge, and the precondition rule that gates them.

The feature exists because of a real report: the detector recommended a dynamic
electricity price sensor to someone on a fixed-price contract. Nothing it can
see - entities - reveals a contract, so the honest form of that advice names the
condition under which it is worth anything. A model proposing gaps has exactly
the same blind spot, so a proposal without a stated precondition is rejected
rather than shown.
"""

from __future__ import annotations

from amminer.enrich.detect import SignalSet
from amminer.gaps import GapSuggestion
from amminer.llm.gaps import MIN_REQUIRES_CHARS, build_prompt, propose
from amminer.llm.provider import LLMError, NullProvider

KNOWN = {"sensor.power", "binary_sensor.hall", "light.kitchen"}


class StubLLM(NullProvider):
    name, enabled = "stub", True

    def __init__(self, proposals=None, raises=False, payload=None):
        self.payload = payload if payload is not None else {"proposals": proposals or []}
        self.raises = raises
        self.prompts: list[str] = []

    def complete_json(self, system, user):
        self.prompts.append(user)
        if self.raises:
            raise LLMError("stub is down")
        return self.payload


def _proposal(**overrides):
    base = {
        "title": "Add an energy dashboard",
        "gap": "You measure power but nothing accumulates it into energy.",
        "recommendation": "Configure Home Assistant's Energy dashboard.",
        "benefit": "You can see where the power actually goes.",
        "requires": "a sensor reporting cumulative energy in kWh, not just instantaneous watts.",
    }
    base.update(overrides)
    return base


def _existing():
    return [GapSuggestion("integration", "Add a dynamic electricity price sensor", "g", "r", "b")]


def _run(proposals, **kwargs):
    signals = SignalSet()
    signals.motion = ["binary_sensor.hall"]
    return propose(signals, {"light": 40}, _existing(), StubLLM(proposals), known_entities=KNOWN, **kwargs)


# --- the precondition rule ----------------------------------------------
def test_a_good_proposal_is_accepted_and_labelled():
    result = _run([_proposal()])
    assert len(result.accepted) == 1
    gap = result.accepted[0]
    assert gap.title == "Add an energy dashboard"
    assert "kWh" in gap.requires
    assert gap.source == "ai", "the card must say a model proposed this"
    assert gap.score < 0.5, "a proposal ranks below a detected gap"


def test_a_proposal_without_a_precondition_is_rejected():
    """The failure this feature was built to stop."""
    result = _run([_proposal(requires="")])
    assert result.accepted == []
    assert result.missing_requires == 1


def test_a_gestured_precondition_is_not_a_precondition():
    """'yes' is not a condition anyone can check about their own life."""
    result = _run([_proposal(requires="yes")])
    assert result.accepted == []
    assert result.missing_requires == 1
    assert len("yes") < MIN_REQUIRES_CHARS


def test_every_detector_gap_still_states_its_own_precondition():
    """The deterministic side had the same blind spot; the reported one especially."""
    import inspect

    from amminer import gaps as gap_module

    source = inspect.getsource(gap_module.suggest)
    pricing = source.split('title="Add a dynamic electricity price sensor')[1].split(
        "GapSuggestion("
    )[0]
    assert "requires=" in pricing
    assert "fixed-price" in pricing or "fixed price" in pricing


# --- what it may not do -------------------------------------------------
def test_it_cannot_restate_a_detector_gap():
    reworded = _proposal(title="Add a dynamic ELECTRICITY price sensor!!")
    result = _run([reworded])
    assert result.accepted == []
    assert result.duplicates == 1


def test_it_cannot_cite_an_entity_that_does_not_exist():
    result = _run([_proposal(gap="Your sensor.imaginary_meter reports nothing.")])
    assert result.accepted == []
    assert result.invented_entities == ["sensor.imaginary_meter"]


def test_it_may_cite_an_entity_that_does():
    result = _run([_proposal(gap="Your sensor.power reports watts only.")])
    assert len(result.accepted) == 1


def test_an_incomplete_proposal_is_rejected():
    result = _run([_proposal(recommendation="")])
    assert result.accepted == []
    assert result.incomplete == 1


def test_it_cannot_flood_the_page():
    from amminer.llm.gaps import MAX_PROPOSALS

    many = [_proposal(title=f"Idea number {i}") for i in range(20)]
    result = _run(many)
    assert len(result.accepted) <= MAX_PROPOSALS


def test_two_proposals_with_the_same_title_collapse():
    result = _run([_proposal(), _proposal()])
    assert len(result.accepted) == 1
    assert result.duplicates == 1


# --- it must never cost the detector ------------------------------------
def test_a_provider_outage_yields_nothing_and_reports_it():
    signals = SignalSet()
    result = propose(signals, {}, _existing(), StubLLM(raises=True), known_entities=KNOWN)
    assert result.accepted == []
    assert "stub is down" in result.error


def test_no_provider_is_reported_not_raised():
    result = propose(SignalSet(), {}, _existing(), NullProvider(), known_entities=KNOWN)
    assert result.accepted == []
    assert result.error == "no LLM provider configured"


def test_junk_instead_of_proposals_is_survived():
    signals = SignalSet()
    for payload in ({}, {"proposals": "not a list"}, {"proposals": ["a string"]}):
        result = propose(
            signals, {}, _existing(), StubLLM(payload=payload), known_entities=KNOWN
        )
        assert result.accepted == []
        assert result.error is None


# --- what the model is shown --------------------------------------------
def test_the_prompt_carries_no_raw_history():
    signals = SignalSet()
    signals.motion = ["binary_sensor.hall"]
    prompt = build_prompt(signals, {"light": 40}, _existing())
    assert "1700000" not in prompt and "timestamp" not in prompt.lower()
    # It does carry what it needs to reason.
    assert "manual_actions_by_domain" in prompt
    assert "already_suggested_by_the_detector" in prompt
    assert "Add a dynamic electricity price sensor" in prompt


def test_the_summary_reports_every_rejection_reason():
    result = _run([
        _proposal(title="Good one"),
        _proposal(title="No precondition", requires=""),
        _proposal(title="Add a dynamic electricity price sensor"),
        _proposal(title="Incomplete", benefit=""),
    ])
    summary = result.as_dict()
    assert summary["proposed"] == 4
    assert summary["accepted"] == 1
    assert summary["rejected_no_precondition"] == 1
    assert summary["rejected_duplicate"] == 1
    assert summary["rejected_incomplete"] == 1


def test_the_feature_is_off_by_default():
    from amminer.config import Options

    assert Options().llm_gaps is False
    assert Options().ai_features_requested["gap_proposals"] is False
