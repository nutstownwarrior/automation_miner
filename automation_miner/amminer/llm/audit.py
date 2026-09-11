"""Optional AI feature 4 - does this audit finding actually make sense?

The audit compares existing automations by trigger, target and condition.  The
deterministic part now refuses to report a pair whose conditions *provably*
cannot both hold - "when I am home" against "when I am out", "below 20 lux"
against "above 500".  What it cannot decide is the rest: two rules conditioned
on different entities entirely, where whether they ever coincide depends on what
those entities mean in this house.

That is a semantic question, so it is asked here - and given no authority:

* it may **dismiss** a finding, which hides it with its reasoning recorded,
* it may **soften** one from error to warning,
* it may never raise a severity, and never invent a finding: a pair the
  deterministic audit did not report is never shown, whatever the model says,
* a finding it never mentions is left exactly as it was.

The worst a wrong verdict can do is hide one questionable finding, with an
arguable reason attached and the deterministic finding still in the run report.
Getting that backwards - letting a model *add* conflicts - would make the audit
less trustworthy than having no AI at all.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..util.text import clean_model_text
from .provider import BaseProvider, LLMError

_LOGGER = logging.getLogger(__name__)

VERDICTS = ("real", "not_a_conflict", "unsure")

#: Severity can only ever move this way.
SOFTER = {"error": "warning", "warning": "info", "info": "info"}

SYSTEM_PROMPT = """\
You review warnings about a person's existing home automations.

Each warning says two automations act on the same entity and might interfere.
That was decided by comparing triggers and targets. You are given both rules in
full, including their conditions, and asked one question: in a real home, can
these two ever actually apply at the same time?

Two automations doing the same thing under DIFFERENT conditions are normal and
correct - a person deliberately covering two situations. That is not a conflict.
A conflict is when both can fire in the SAME situation and fight over an entity.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- Only use the "id" values given in the input. Never invent one.
- verdict must be exactly one of: "real", "not_a_conflict", "unsure".
- Use "not_a_conflict" when the conditions describe situations that do not
  overlap, or when the two rules plainly serve different purposes.
- Use "real" when both really could fire together and disagree.
- Use "unsure" when you cannot tell from what you were given. That is useful.
- "reason" must be one short sentence a homeowner would understand.

Shape:
{"reviews": [{"id": "f1", "verdict": "not_a_conflict",
              "reason": "One runs only when nobody is home, the other only when someone is."}]}
"""


@dataclass
class AuditReviewResult:
    """Verdicts that survived validation, and what they changed."""

    verdicts: dict[str, dict[str, str]] = field(default_factory=dict)
    dismissed: list[str] = field(default_factory=list)
    softened: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)
    invalid_verdicts: int = 0
    reviewed: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for entry in self.verdicts.values():
            counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
        return {
            "reviewed": self.reviewed,
            "verdicts": counts,
            "dismissed": len(self.dismissed),
            "softened": len(self.softened),
            "unknown_ids": self.unknown_ids[:20],
            "invalid_verdicts": self.invalid_verdicts,
            "error": self.error,
        }


def finding_id(finding: dict[str, Any], index: int) -> str:
    """A stable handle for one finding within one run."""
    return finding.get("id") or f"f{index}"


def _rule(automation: Any) -> dict[str, Any]:
    """One automation as the model needs to see it: what it does and when."""
    raw = getattr(automation, "raw", {}) or {}
    return {
        "alias": getattr(automation, "alias", "") or "unnamed",
        "triggers": raw.get("trigger") or raw.get("triggers") or [],
        "conditions": raw.get("condition") or raw.get("conditions") or [],
        "actions": raw.get("action") or raw.get("actions") or [],
    }


def _describe(
    finding: dict[str, Any], index: int, by_alias: dict[str, Any]
) -> dict[str, Any]:
    """One finding plus the full text of both rules it is about."""
    rules = [
        _rule(by_alias[alias])
        for alias in finding.get("automations", [])
        if alias in by_alias
    ]
    return {
        "id": finding_id(finding, index),
        "concern": finding.get("message", ""),
        "entities": finding.get("entities", []),
        "rules": rules,
    }


def review(
    findings: Sequence[dict[str, Any]],
    automations: Sequence[Any],
    provider: BaseProvider,
    batch_size: int = 15,
) -> AuditReviewResult:
    """Ask whether each audit finding is a real conflict.  Never raises."""
    result = AuditReviewResult()
    # Only findings about a *pair* of rules are answerable this way.
    reviewable = [
        (index, finding)
        for index, finding in enumerate(findings)
        if len(finding.get("automations") or []) == 2
    ]
    if not reviewable:
        return result
    if not provider.enabled:
        result.error = "no LLM provider configured"
        return result

    by_alias = {getattr(a, "alias", ""): a for a in automations}
    known_ids = {finding_id(finding, index) for index, finding in reviewable}

    step = max(batch_size, 1)
    for start in range(0, len(reviewable), step):
        batch = reviewable[start : start + step]
        prompt = json.dumps(
            {"warnings": [_describe(f, i, by_alias) for i, f in batch]},
            indent=2,
            default=str,
        )
        try:
            raw = provider.complete_json(SYSTEM_PROMPT, prompt)
        except LLMError as err:
            result.error = str(err)
            _LOGGER.warning("Audit review failed: %s", err)
            break
        result.reviewed += len(batch)

        reviews = raw.get("reviews")
        if not isinstance(reviews, list):
            continue
        for entry in reviews:
            if not isinstance(entry, dict):
                continue
            verdict = str(entry.get("verdict") or "").strip().lower()
            if verdict not in VERDICTS:
                result.invalid_verdicts += 1
                continue
            found_id = entry.get("id")
            if not isinstance(found_id, str) or found_id not in known_ids:
                if isinstance(found_id, str):
                    result.unknown_ids.append(found_id)
                continue
            result.verdicts[found_id] = {
                "verdict": verdict,
                "reason": clean_model_text(entry.get("reason")),
            }
    return result


def apply_verdicts(
    findings: Sequence[dict[str, Any]], result: AuditReviewResult
) -> list[dict[str, Any]]:
    """Hide dismissed findings and soften unsure ones.  Never escalates."""
    kept: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        entry = result.verdicts.get(finding_id(finding, index))
        if entry is None:
            kept.append(finding)
            continue

        finding = dict(finding)
        finding["ai_review"] = entry
        verdict = entry["verdict"]
        if verdict == "not_a_conflict":
            # Hidden, not deleted: the run report still counts it, so a model
            # quietly hiding real conflicts is visible on the Status page.
            result.dismissed.append(finding_id(finding, index))
            continue
        if verdict == "unsure":
            current = str(finding.get("severity") or "info")
            softer = SOFTER.get(current, current)
            if softer != current:
                finding["severity"] = softer
                result.softened.append(finding_id(finding, index))
        # "real" changes nothing: the deterministic finding already said that.
        kept.append(finding)
    return kept
