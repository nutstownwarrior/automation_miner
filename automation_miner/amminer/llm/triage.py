"""Optional AI feature 3 - semantic plausibility triage.

The miners find patterns that are statistically real and semantically absurd.
A live example from this project's own fixture: ``climate.living_room`` going
off and ``light.bedroom`` coming on have a lift of 4.3 - because both happen in
the evening, not because either causes the other.  Backtesting does not always
catch this; a coincidence that recurs reliably backtests perfectly well.

Judging that is a semantic question, and a model is good at it.  So it is asked
- and given no authority whatsoever:

* it may mark a rule implausible, which applies a score **penalty** and shows
  its reasoning on the card,
* it may not promote anything: a "plausible" verdict changes nothing,
* it may not remove anything: every rule stays visible, with its evidence and
  its backtest intact, and the user decides,
* a rule it never mentions is left exactly as it was.

That asymmetry is the point.  The worst a wrong verdict can do is push a good
suggestion down the list with a visible, arguable reason attached.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..miners.base import Candidate
from ..util.text import clean_model_text
from .provider import BaseProvider, LLMError

_LOGGER = logging.getLogger(__name__)

VERDICTS = ("plausible", "implausible", "unsure")

SYSTEM_PROMPT = """\
You review proposed home automations for whether they make SENSE.

Each rule was found by statistical analysis of real usage, so the correlation is
already known to be real. Your only question is whether it reflects something a
person would actually intend, or whether it is a coincidence - two things that
merely happen at similar times of day.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- Only use the "id" values given in the input. Never invent one.
- verdict must be exactly one of: "plausible", "implausible", "unsure".
- Use "implausible" ONLY for rules where the trigger has no believable
  connection to the action. Being unusual is not implausible.
- Use "unsure" when you genuinely cannot tell. That is a useful answer.
- "reason" must be one short sentence a homeowner would understand.
- Judge intent, not statistics. Do not comment on the numbers.

Shape:
{"reviews": [{"id": "abc123", "verdict": "implausible",
              "reason": "Turning the kettle on has no bearing on the garage door."}]}
"""


@dataclass
class TriageResult:
    """Verdicts that survived validation, and what they changed."""

    verdicts: dict[str, dict[str, str]] = field(default_factory=dict)
    demoted: list[str] = field(default_factory=list)
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
            "demoted": len(self.demoted),
            "unknown_ids": self.unknown_ids[:20],
            "invalid_verdicts": self.invalid_verdicts,
            "error": self.error,
        }


def _describe(candidate: Candidate, resolver=None) -> dict[str, Any]:
    """The rule in plain language, with no statistics attached.

    The numbers are withheld on purpose: the model is asked whether the rule
    makes sense, and showing it a high confidence score invites it to agree with
    the statistics instead of judging the semantics independently.
    """
    payload: dict[str, Any] = {
        "id": candidate.id,
        "rule": candidate.describe(resolver),
    }
    entities = {}
    for entity_id in candidate.entities:
        info = resolver.resolve(entity_id) if resolver else None
        if info is None:
            continue
        described = {"name": info.name}
        if info.area_name:
            described["area"] = info.area_name
        if info.device_class:
            described["device_class"] = info.device_class
        entities[entity_id] = described
    if entities:
        payload["entities"] = entities
    return payload


def triage(
    candidates: Sequence[Candidate],
    provider: BaseProvider,
    resolver=None,
    batch_size: int = 25,
) -> TriageResult:
    """Ask for a plausibility verdict on each candidate.  Never raises."""
    result = TriageResult()
    reviewable = [c for c in candidates if c.actions and c.triggers]
    if not reviewable:
        return result
    if not provider.enabled:
        result.error = "no LLM provider configured"
        return result

    known_ids = {c.id for c in reviewable}
    for start in range(0, len(reviewable), max(batch_size, 1)):
        batch = reviewable[start : start + max(batch_size, 1)]
        prompt = json.dumps(
            {"rules": [_describe(c, resolver) for c in batch]}, indent=2, default=str
        )
        try:
            raw = provider.complete_json(SYSTEM_PROMPT, prompt)
        except LLMError as err:
            result.error = str(err)
            _LOGGER.warning("Triage failed: %s", err)
            break
        result.reviewed += len(batch)

        reviews = raw.get("reviews")
        if not isinstance(reviews, list):
            continue
        for review in reviews:
            if not isinstance(review, dict):
                continue
            verdict = str(review.get("verdict") or "").strip().lower()
            if verdict not in VERDICTS:
                result.invalid_verdicts += 1
                continue
            candidate_id = review.get("id")
            if not isinstance(candidate_id, str) or candidate_id not in known_ids:
                if isinstance(candidate_id, str):
                    result.unknown_ids.append(candidate_id)
                continue
            result.verdicts[candidate_id] = {
                "verdict": verdict,
                "reason": clean_model_text(review.get("reason")),
            }
    return result


def apply_verdicts(
    candidates: Sequence[Candidate], result: TriageResult, penalty: float = 0.5
) -> Sequence[Candidate]:
    """Attach verdicts and demote the implausible ones, in place.

    Nothing is removed and nothing is promoted; ``penalty`` only ever scales a
    score down.  Candidates the model did not mention are untouched.
    """
    penalty = min(max(float(penalty), 0.0), 1.0)
    for candidate in candidates:
        entry = result.verdicts.get(candidate.id)
        if entry is None:
            continue
        candidate.extra["triage"] = dict(entry)
        if entry["verdict"] == "implausible":
            candidate.score = round(candidate.score * penalty, 4)
            candidate.extra["triage"]["score_penalty"] = penalty
            result.demoted.append(candidate.id)
    if result.demoted:
        _LOGGER.info(
            "Triage demoted %d candidates as semantically implausible "
            "(none were removed)",
            len(result.demoted),
        )
    return candidates
