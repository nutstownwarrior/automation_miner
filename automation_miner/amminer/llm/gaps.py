"""Optional AI feature 5 - gaps the detector has no rule for.

The deterministic gap rules each look for one named pattern: deferrable loads
with no price signal, motion sensors with no presence sensor, and so on.  They
are reliable and they are a fixed list.  What they cannot do is notice that
someone with a heat pump, an EV and no energy dashboard might want one - that
takes knowing what those things are for, which is world knowledge.

So a model is asked, and is held to the rule that prompted this feature: a
recommendation inferred from entities cannot see a contract, a roof or a
commute, so **every proposal must state the precondition that makes it worth
doing**.  "Add a dynamic price sensor" is sound advice on a variable tariff and
useless on a fixed one, and a suggestion that does not say which is a guess
presented as a recommendation.  A proposal arriving without that is rejected,
not patched up.

What it is not allowed to do:

* it may not remove, reword or reorder anything the detector produced,
* it may not claim the user *has* something - it proposes, and says under what
  condition the proposal applies,
* it may not cite an entity that does not exist,
* it may not restate a gap the detector already found,
* everything it proposes is labelled as model-proposed on the card.

It is never sent raw history: only which signal roles were detected, how much
manual activity there was per domain, and the titles the detector already used.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..gaps import GapSuggestion
from ..util.text import clean_model_text
from .provider import BaseProvider, LLMError

_LOGGER = logging.getLogger(__name__)

#: Proposals per run.  A gap list is a shopping list; an unbounded one is noise.
MAX_PROPOSALS = 5

#: Long enough to actually name a condition rather than gesture at one.
MIN_REQUIRES_CHARS = 20

SYSTEM_PROMPT = """\
You suggest integrations or hardware a smart-home owner might be missing.

You are given what their instance can already sense, how much they operate by
hand, and the suggestions an automated detector has already made. Propose only
things the detector has NOT covered.

The rule that matters most:

Everything you know about this home was inferred from its devices. Devices
cannot see an electricity contract, a roof, a car, a job or a family. So every
proposal MUST carry a "requires" field naming the real-world condition under
which it is worth doing - and it must be a condition the person can check about
their own life, not a restatement of the suggestion.

Good: "requires": "an electricity contract whose price varies through the day;
on a fixed tariff this saves nothing."
Bad:  "requires": "that you want cheaper electricity."

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- At most 5 proposals. Fewer is better. Propose nothing if nothing fits.
- Never claim the person HAS something. You do not know.
- Never name an entity_id that was not given to you.
- "requires" is mandatory and must name a checkable real-world condition.
- "gap", "recommendation" and "benefit" are one or two plain sentences each.
- Do not repeat a detector suggestion under a different name.

Shape:
{"proposals": [{"title": "...", "gap": "...", "recommendation": "...",
                "benefit": "...", "requires": "..."}]}
"""


@dataclass
class GapProposalResult:
    """Proposals that survived validation, and what was thrown away."""

    accepted: list[GapSuggestion] = field(default_factory=list)
    missing_requires: int = 0
    duplicates: int = 0
    invented_entities: list[str] = field(default_factory=list)
    incomplete: int = 0
    proposed: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposed": self.proposed,
            "accepted": len(self.accepted),
            "rejected_no_precondition": self.missing_requires,
            "rejected_duplicate": self.duplicates,
            "rejected_incomplete": self.incomplete,
            "invented_entities": self.invented_entities[:20],
            "error": self.error,
        }


def _normalise(title: str) -> str:
    """Titles compared loosely, so a reworded duplicate is still a duplicate."""
    return "".join(c for c in title.lower() if c.isalnum() or c == " ").strip()


def build_prompt(
    signals: Any, action_counts: dict[str, int], existing: Sequence[GapSuggestion]
) -> str:
    """What the model is shown.  No raw history, no timestamps, no values."""
    roles = signals.as_dict() if hasattr(signals, "as_dict") else {}
    return json.dumps(
        {
            "signal_roles_detected": sorted(k for k, v in roles.items() if v),
            "signal_roles_absent": sorted(k for k, v in roles.items() if not v),
            "manual_actions_by_domain": dict(sorted(action_counts.items())),
            "already_suggested_by_the_detector": [g.title for g in existing],
        },
        indent=2,
        default=str,
    )


def propose(
    signals: Any,
    action_counts: dict[str, int],
    existing: Sequence[GapSuggestion],
    provider: BaseProvider,
    known_entities: set[str] | None = None,
) -> GapProposalResult:
    """Ask for gaps the detector has no rule for.  Never raises."""
    result = GapProposalResult()
    if not provider.enabled:
        result.error = "no LLM provider configured"
        return result

    try:
        raw = provider.complete_json(
            SYSTEM_PROMPT, build_prompt(signals, action_counts, existing)
        )
    except LLMError as err:
        result.error = str(err)
        _LOGGER.warning("Gap proposals failed: %s", err)
        return result

    proposals = raw.get("proposals")
    if not isinstance(proposals, list):
        return result

    seen = {_normalise(g.title) for g in existing}
    known = known_entities or set()

    for entry in proposals[:MAX_PROPOSALS]:
        if not isinstance(entry, dict):
            continue
        result.proposed += 1

        title = clean_model_text(entry.get("title"), limit=120)
        gap = clean_model_text(entry.get("gap"), limit=400)
        recommendation = clean_model_text(entry.get("recommendation"), limit=400)
        benefit = clean_model_text(entry.get("benefit"), limit=300)
        requires = clean_model_text(entry.get("requires"), limit=400)

        if not (title and gap and recommendation and benefit):
            result.incomplete += 1
            continue

        # The whole reason this feature exists.  A proposal that does not say
        # what must be true is exactly the failure it was built to stop.
        if len(requires) < MIN_REQUIRES_CHARS:
            result.missing_requires += 1
            continue

        if _normalise(title) in seen:
            result.duplicates += 1
            continue

        invented = [
            word.strip(".,;:()[]\"'")
            for text in (gap, recommendation, benefit, requires)
            for word in text.split()
            if word.count(".") == 1
            and word.strip(".,;:()[]\"'").replace("_", "a").replace(".", "a").isalnum()
            and word.strip(".,;:()[]\"'").split(".")[0] in _HA_DOMAINS
        ]
        unknown = sorted({e for e in invented if known and e not in known})
        if unknown:
            result.invented_entities.extend(unknown)
            continue

        seen.add(_normalise(title))
        result.accepted.append(
            GapSuggestion(
                kind="integration",
                title=title,
                gap=gap,
                recommendation=recommendation,
                benefit=benefit,
                requires=requires,
                evidence=["Proposed by the language model from what this instance can sense."],
                # Below every detector suggestion: a proposal from world
                # knowledge is a weaker signal than a detected gap.
                score=0.3,
                source="ai",
            )
        )
    return result


#: Domains a word like "sensor.foo" could plausibly be, for the citation check.
_HA_DOMAINS = frozenset(
    {
        "sensor", "binary_sensor", "light", "switch", "climate", "cover", "lock",
        "media_player", "person", "device_tracker", "input_boolean", "input_number",
        "input_select", "number", "select", "fan", "vacuum", "water_heater", "valve",
        "siren", "humidifier", "camera", "weather", "sun", "calendar", "automation",
        "script", "scene", "zone", "alarm_control_panel", "lawn_mower", "todo",
    }
)
