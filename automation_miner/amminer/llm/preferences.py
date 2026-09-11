"""Optional AI feature 6 - learning what you have already said no to.

``dismissals.reason`` has been written since the first release and read by
nothing.  Dismissing is therefore a mute keyed to one exact rule: reword the
rule and it comes back, and the sentence the person typed explaining *why* -
the only place they say anything about their own home in their own words -
is stored and ignored.

This reads those sentences back and generalises them into standing preferences
("nothing in the guest room", "never touch the bedroom lights before 07:00"),
then suppresses new candidates that match one.

The asymmetry is the same as everywhere else in this project, and matters more
here because suppression removes things from view:

* a preference can only **hide** a candidate, never promote or reorder one,
* it never alters evidence, a backtest, or a score,
* every suppression records which preference caused it and the sentence that
  preference came from, and lands where the user can undo it,
* preferences are derived, never authoritative: they are rebuilt from the
  dismissals each run, and the user can switch one off.

It is never sent raw history - only the titles the user dismissed and the
reasons they gave for dismissing them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..miners.base import Candidate
from ..util.text import clean_model_text
from .provider import BaseProvider, LLMError

_LOGGER = logging.getLogger(__name__)

#: Below this there is nothing to generalise from, and a "preference" learned
#: from one dismissal is just that dismissal with a wider blast radius.
MIN_DISMISSALS = 3

#: A handful of standing rules is a preference set; thirty is a filter nobody
#: asked for.
MAX_PREFERENCES = 8

LEARN_PROMPT = """\
You read the reasons a person gave for rejecting home-automation suggestions,
and generalise them into standing preferences.

Each input is a suggestion they dismissed and the sentence they typed saying
why. Find the preferences that would have predicted several of those, and say
them as a rule that could be applied to a NEW suggestion.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- A preference must be supported by at least two dismissals. Cite their ids.
- At most 8 preferences. Fewer is much better.
- Write each as a rule about what NOT to suggest, in one sentence.
- Be specific. "They dislike some suggestions" is useless. "Nothing that
  automates the guest room" is a rule.
- Do not invent reasons they did not give. If several dismissals share no
  theme, return fewer preferences or none.

Shape:
{"preferences": [{"rule": "Do not suggest anything that automates the guest room.",
                  "from": ["abc123", "def456"]}]}
"""

MATCH_PROMPT = """\
You apply a person's standing preferences to new home-automation suggestions.

For each suggestion, say whether it matches a preference - meaning the person
would reject it for the same reason they rejected things before.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- Only use the "id" values given. Never invent one.
- Only match when you are confident. Leaving a suggestion visible is the safe
  answer; hiding one the person wanted is not.
- Give the preference id that matched, exactly as given.

Shape:
{"matches": [{"id": "cand1", "preference": "p1",
              "reason": "This automates the guest room."}]}
"""


@dataclass
class Preference:
    rule: str
    evidence: list[str] = field(default_factory=list)
    #: Left empty when the model has just proposed the rule, and derived from
    #: its text.  Carried explicitly for one already in the store, because a
    #: rule the user has since rewritten no longer hashes to its own id - and
    #: the id is what the suggestions it hid point at.
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "p" + hashlib.sha1(self.rule.encode()).hexdigest()[:10]

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Preference:
        return cls(
            rule=row["rule"], evidence=list(row.get("evidence") or []), id=row["id"]
        )

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "rule": self.rule, "evidence": self.evidence}


@dataclass
class PreferenceResult:
    """What was learned, and what it hid."""

    learned: list[Preference] = field(default_factory=list)
    suppressed: dict[str, dict[str, str]] = field(default_factory=dict)
    unsupported: int = 0
    unknown_ids: list[str] = field(default_factory=list)
    considered: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "learned": [p.as_dict() for p in self.learned],
            "suppressed": len(self.suppressed),
            "considered": self.considered,
            "rejected_unsupported": self.unsupported,
            "unknown_ids": self.unknown_ids[:20],
            "error": self.error,
        }


def learn(
    dismissals: Sequence[dict[str, Any]], provider: BaseProvider
) -> tuple[list[Preference], str | None]:
    """Generalise standing preferences from dismissal reasons."""
    if len(dismissals) < MIN_DISMISSALS:
        return [], None
    if not provider.enabled:
        return [], "no LLM provider configured"

    prompt = json.dumps(
        {
            "dismissals": [
                {
                    "id": d.get("suggestion_id"),
                    "suggestion": d.get("title") or "",
                    "their_reason": clean_model_text(d.get("reason"), limit=300),
                }
                for d in dismissals
            ]
        },
        indent=2,
        default=str,
    )
    try:
        raw = provider.complete_json(LEARN_PROMPT, prompt)
    except LLMError as err:
        _LOGGER.warning("Learning preferences failed: %s", err)
        return [], str(err)

    entries = raw.get("preferences")
    if not isinstance(entries, list):
        return [], None

    known = {d.get("suggestion_id") for d in dismissals}
    out: list[Preference] = []
    seen: set[str] = set()
    for entry in entries[:MAX_PREFERENCES]:
        if not isinstance(entry, dict):
            continue
        rule = clean_model_text(entry.get("rule"), limit=200)
        cited = [c for c in (entry.get("from") or []) if c in known]
        # One dismissal is not a pattern; generalising from it just widens
        # that single mute without the user asking for it.
        if not rule or len(cited) < 2:
            continue
        preference = Preference(rule=rule, evidence=cited)
        if preference.id in seen:
            continue
        seen.add(preference.id)
        out.append(preference)
    return out, None


def apply_preferences(
    candidates: Sequence[Candidate],
    preferences: Sequence[Preference],
    provider: BaseProvider,
    resolver=None,
) -> PreferenceResult:
    """Decide which candidates a standing preference hides.  Never raises."""
    result = PreferenceResult(learned=list(preferences))
    if not candidates or not preferences:
        return result
    if not provider.enabled:
        result.error = "no LLM provider configured"
        return result

    by_id = {c.id: c for c in candidates}
    known_preferences = {p.id: p for p in preferences}
    result.considered = len(candidates)

    prompt = json.dumps(
        {
            "preferences": [p.as_dict() for p in preferences],
            "suggestions": [
                {"id": c.id, "title": c.title, "describes": c.describe(resolver)}
                for c in candidates
            ],
        },
        indent=2,
        default=str,
    )
    try:
        raw = provider.complete_json(MATCH_PROMPT, prompt)
    except LLMError as err:
        result.error = str(err)
        _LOGGER.warning("Applying preferences failed: %s", err)
        return result

    matches = raw.get("matches")
    if not isinstance(matches, list):
        return result

    for match in matches:
        if not isinstance(match, dict):
            continue
        candidate_id = match.get("id")
        preference_id = match.get("preference")
        if not isinstance(candidate_id, str) or candidate_id not in by_id:
            if isinstance(candidate_id, str):
                result.unknown_ids.append(candidate_id)
            continue
        if preference_id not in known_preferences:
            # A suppression has to name a preference the user can read and
            # switch off, or it is an unaccountable disappearance.
            result.unsupported += 1
            continue
        result.suppressed[candidate_id] = {
            "preference": preference_id,
            "rule": known_preferences[preference_id].rule,
            "reason": clean_model_text(match.get("reason"), limit=300),
        }
    return result
