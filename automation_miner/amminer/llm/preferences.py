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


DRAFT_PROMPT = """\
You help someone word a standing preference about their own home automation.

A preference is one sentence saying what NOT to suggest. You are given what they
want changed, and - if they are amending one that already exists - its current
wording.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- "rule" is ONE sentence, written as a rule about what not to suggest.
- Apply exactly what they asked for and change nothing else. If they narrow a
  rule, keep the rest of it intact.
- Do not add conditions, rooms, times or devices they did not mention. You are
  wording their instruction, not improving it.
- Do not refer to "the user" or to yourself. Write the rule plainly.
- If what they said cannot be expressed as a preference, return an empty rule.

Shape:
{"rule": "Do not suggest anything that automates the guest room lamp."}
"""


def draft(instruction: str, rule: str, provider: BaseProvider) -> tuple[str, str | None]:
    """Word a preference from an instruction.  Returns ``(proposal, error)``.

    This deliberately returns a proposal and writes nothing.  A model that could
    edit a stored preference directly would be able to reword the rules that
    hide things, which is the one power this whole feature is built to withhold.
    Here it can only fill in a text box that a person then reads and saves, so
    the stored rule is the user's either way.
    """
    instruction = clean_model_text(instruction, limit=500)
    if not instruction:
        return "", "say what you would like changed"
    if not provider.enabled:
        return "", "no LLM provider configured"

    prompt = json.dumps(
        {"current_rule": clean_model_text(rule, limit=200), "change_requested": instruction},
        indent=2,
    )
    try:
        raw = provider.complete_json(DRAFT_PROMPT, prompt)
    except LLMError as err:
        _LOGGER.warning("Drafting a preference failed: %s", err)
        return "", str(err)
    # The same cap the store applies, so what is shown is what can be saved.
    return clean_model_text(raw.get("rule"), limit=200), None


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
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rule = clean_model_text(entry.get("rule"), limit=200)
        cited = _citations(entry.get("from"), known)
        # One dismissal is not a pattern; generalising from it just widens
        # that single mute without the user asking for it.  Counted over
        # *distinct* ids, because the same dismissal named twice is still one
        # dismissal and would otherwise clear this bar on its own.
        if not rule or len(cited) < 2:
            continue
        preference = Preference(rule=rule, evidence=cited)
        if preference.id in seen:
            continue
        seen.add(preference.id)
        out.append(preference)
        # Sliced after validation, not before: taking the first eight entries
        # and then discarding the invalid ones silently threw away good
        # preferences further down the list.
        if len(out) >= MAX_PREFERENCES:
            break
    return out, None


def _citations(raw: object, known: set) -> list[str]:
    """The real dismissal ids an entry cites, deduplicated and in order.

    Everything here is model output, so a citation may be any JSON value at
    all: a bare number where a list was asked for, or a nested object whose
    membership test would raise on an unhashable type.
    """
    if not isinstance(raw, list):
        return []
    cited: list[str] = []
    for item in raw:
        if isinstance(item, str) and item in known and item not in cited:
            cited.append(item)
    return cited


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
                # Capped like every other piece of model text here: this is
                # persisted into the run report, and an id is a short string
                # or it is not an id.
                result.unknown_ids.append(clean_model_text(candidate_id, limit=100))
            continue
        # The isinstance check is not decoration: a non-string here is any
        # JSON value the model felt like sending, and an unhashable one turns
        # the membership test below into a TypeError.
        if not isinstance(preference_id, str) or preference_id not in known_preferences:
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
