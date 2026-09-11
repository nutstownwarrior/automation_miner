"""The model reviewing audit findings, and the authority it is denied.

The deterministic audit decides what is reported. This feature may hide or
soften one of those findings; the tests that matter most are the ones proving
it can never add a conflict or make one louder, because a model that could do
that would make the audit less trustworthy than having no AI at all.
"""

from __future__ import annotations

import json

from amminer.automations import normalise_automation
from amminer.llm.audit import apply_verdicts, review
from amminer.llm.provider import LLMError, NullProvider


class StubLLM(NullProvider):
    name, enabled = "stub", True

    def __init__(self, verdicts=None, raises=False, payload=None):
        self.verdicts = verdicts or {}
        self.raises = raises
        self.payload = payload
        self.calls = 0
        self.prompts: list[str] = []

    def complete_json(self, system, user):
        self.calls += 1
        self.prompts.append(user)
        if self.raises:
            raise LLMError("stub is down")
        if self.payload is not None:
            return self.payload
        ids = [w["id"] for w in json.loads(user)["warnings"]]
        return {
            "reviews": [
                {"id": i, "verdict": self.verdicts.get(i, "unsure"), "reason": "because"}
                for i in ids
            ]
        }


def _rule(alias, condition, service):
    return normalise_automation({
        "id": alias, "alias": alias,
        "trigger": [{"platform": "state", "entity_id": "binary_sensor.motion", "to": "on"}],
        "condition": [condition],
        "action": [{"service": service, "target": {"entity_id": "light.kitchen"}}],
    })


RULES = [
    _rule("Away mode", {"condition": "state", "entity_id": "person.alex", "state": "not_home"},
          "light.turn_off"),
    _rule("When dark", {"condition": "numeric_state", "entity_id": "sensor.lux", "below": 20},
          "light.turn_on"),
]


def _finding(severity="warning", **extra):
    base = {
        "kind": "value_inconsistency",
        "severity": severity,
        "message": "'Away mode' and 'When dark' both act on light.kitchen.",
        "automations": ["Away mode", "When dark"],
        "entities": ["light.kitchen"],
    }
    base.update(extra)
    return base


# --- what it may do -----------------------------------------------------
def test_a_dismissed_finding_is_hidden_with_its_reasoning():
    findings = [_finding()]
    result = review(findings, RULES, StubLLM({"f0": "not_a_conflict"}))
    kept = apply_verdicts(findings, result)
    assert kept == []
    assert result.dismissed == ["f0"]


def test_an_unsure_verdict_softens_rather_than_hides():
    findings = [_finding(severity="error")]
    result = review(findings, RULES, StubLLM({"f0": "unsure"}))
    kept = apply_verdicts(findings, result)
    assert len(kept) == 1
    assert kept[0]["severity"] == "warning"
    assert kept[0]["ai_review"]["verdict"] == "unsure"


def test_a_real_verdict_changes_nothing():
    findings = [_finding(severity="error")]
    result = review(findings, RULES, StubLLM({"f0": "real"}))
    kept = apply_verdicts(findings, result)
    assert kept[0]["severity"] == "error"
    assert result.dismissed == [] and result.softened == []


# --- what it may never do -----------------------------------------------
def test_it_can_never_raise_a_severity():
    """Every verdict, against every severity: nothing may get louder."""
    order = {"info": 0, "warning": 1, "error": 2}
    for severity in ("info", "warning", "error"):
        for verdict in ("real", "unsure", "not_a_conflict"):
            findings = [_finding(severity=severity)]
            result = review(findings, RULES, StubLLM({"f0": verdict}))
            for kept in apply_verdicts(findings, result):
                assert order[kept["severity"]] <= order[severity], (severity, verdict)


def test_it_cannot_invent_a_finding():
    """A warning the deterministic audit never produced is never shown."""
    findings = [_finding()]
    inventing = StubLLM(payload={"reviews": [
        {"id": "f0", "verdict": "real", "reason": "fine"},
        {"id": "made-up", "verdict": "real", "reason": "a conflict nobody found"},
    ]})
    result = review(findings, RULES, inventing)
    assert result.unknown_ids == ["made-up"]
    assert len(apply_verdicts(findings, result)) == 1


def test_an_invalid_verdict_is_counted_and_ignored():
    findings = [_finding(severity="error")]
    result = review(findings, RULES, StubLLM(payload={"reviews": [
        {"id": "f0", "verdict": "definitely-not-a-conflict-trust-me", "reason": "x"}
    ]}))
    assert result.invalid_verdicts == 1
    assert apply_verdicts(findings, result)[0]["severity"] == "error"


def test_a_finding_it_never_mentions_is_untouched():
    findings = [_finding(severity="error"), _finding(severity="info")]
    result = review(findings, RULES, StubLLM(payload={"reviews": []}))
    kept = apply_verdicts(findings, result)
    assert [f["severity"] for f in kept] == ["error", "info"]
    assert all("ai_review" not in f for f in kept)


# --- it must never cost the audit ---------------------------------------
def test_a_provider_outage_leaves_every_finding_as_it_was():
    findings = [_finding(severity="error")]
    result = review(findings, RULES, StubLLM(raises=True))
    assert "stub is down" in result.error
    assert apply_verdicts(findings, result) == findings


def test_no_provider_is_reported_not_raised():
    findings = [_finding()]
    result = review(findings, RULES, NullProvider())
    assert result.error == "no LLM provider configured"
    assert apply_verdicts(findings, result) == findings


def test_findings_that_name_one_rule_are_not_reviewed():
    """A loop or a self-conflict has nothing to compare conditions against."""
    findings = [_finding(automations=["Away mode"])]
    stub = StubLLM()
    result = review(findings, RULES, stub)
    assert stub.calls == 0
    assert apply_verdicts(findings, result) == findings


# --- what the model is shown --------------------------------------------
def test_the_model_sees_both_rules_in_full():
    """It cannot judge whether conditions overlap without reading them."""
    stub = StubLLM()
    review([_finding()], RULES, stub)
    sent = json.loads(stub.prompts[0])["warnings"][0]
    assert len(sent["rules"]) == 2
    conditions = json.dumps(sent["rules"])
    assert "person.alex" in conditions and "sensor.lux" in conditions


def test_a_pathological_reason_is_capped():
    findings = [_finding()]
    result = review(findings, RULES, StubLLM(payload={"reviews": [
        {"id": "f0", "verdict": "unsure", "reason": "x" * 20_000}
    ]}))
    assert len(result.verdicts["f0"]["reason"]) < 1000


def test_the_summary_reports_what_changed():
    findings = [_finding(), _finding(severity="error")]
    result = review(findings, RULES, StubLLM({"f0": "not_a_conflict", "f1": "unsure"}))
    apply_verdicts(findings, result)
    summary = result.as_dict()
    assert summary["reviewed"] == 2
    assert summary["dismissed"] == 1
    assert summary["softened"] == 1
    assert summary["error"] is None


def test_the_feature_is_off_by_default():
    from amminer.config import Options

    assert Options().llm_audit is False
    assert Options().ai_features_requested["audit_review"] is False
