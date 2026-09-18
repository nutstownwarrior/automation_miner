"""Optional AI feature - a friendlier name for an inferred household mode.

``amminer.learn.home_mode`` fits unlabelled, unsupervised states from
activity: it can tell you a mode is active 22% of the time, typically around
21:40, mostly `light` and `media_player` - it has no idea a person would call
that "winding down". A language model reading the same structural description
often can, and a short name is genuinely easier to recognise at a glance than
a table of numbers.

This is asked under the same narrow mandate every other advisory AI feature in
this project gets:

* it is given only the honest, already-computed description of one mode
  (typical hours, top domains, top areas, occupancy share) - never anything
  that could let it invent activity that was not observed,
* it may propose a short label, or decline - it may not describe a mode the
  fitted model did not report, because there is no id to attach a decline to,
* every label is stored and shown clearly marked as AI-suggested, never as
  fact, and never replaces the structural description itself,
* nothing it says changes which candidates are surfaced, how they are scored,
  or how many modes were found - see ``amminer.learn.home_mode``'s own
  "Honesty about what the modes are" for why: the states are unlabelled and
  unsupervised, and a name is not evidence.

This feature is off by default (``llm_home_mode_labels``) and the rest of the
household-mode feature works identically with no LLM configured at all - see
``amminer.learn.home_mode.HomeModeModel.state_summaries``, which already
carries a complete, honest description of every mode before this module ever
runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..util.text import clean_model_text
from .provider import BaseProvider, LLMError

_LOGGER = logging.getLogger(__name__)

#: A label longer than this is not "a name", it is a restated description -
#: and this feature exists to add a name, not a second copy of the structural
#: summary already on the card.
MAX_LABEL_LENGTH = 40

SYSTEM_PROMPT = """\
You suggest a short, friendly name for an inferred pattern in a household's
activity. You are given only structural facts about the pattern: roughly what
share of the observed time it covers, the time of day it is typically active,
and which kinds of devices are most active during it.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- Only use the "state" values given in the input. Never invent one.
- "label" must be 1-4 words, in plain language a homeowner would use
  ("Evening wind-down", "Quiet overnight", "Away during the day").
- Never claim certainty the facts do not support. Do not say "asleep" or
  "away" unless the facts already say so directly (e.g. a person entity or
  device tracker was named as a top domain/area) - prefer a description of the
  activity itself ("Quiet hours") when they do not.
- If you cannot suggest anything better than restating the facts, leave that
  state out entirely. No label is better than a bad one.

Shape:
{"labels": [{"state": 0, "label": "Evening wind-down"}]}
"""


@dataclass
class HomeModeLabelResult:
    """Labels that survived validation, keyed by state index."""

    labels: dict[int, str] = field(default_factory=dict)
    considered: int = 0
    rejected_unknown_state: int = 0
    rejected_too_long: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "labelled": len(self.labels),
            "rejected_unknown_state": self.rejected_unknown_state,
            "rejected_too_long": self.rejected_too_long,
            "error": self.error,
        }


def _describe(summary: dict[str, Any]) -> dict[str, Any]:
    """The honest facts only - no evidence numbers beyond what is already public."""
    described: dict[str, Any] = {
        "state": summary["state"],
        "occupancy_share": summary["occupancy_share"],
    }
    if summary.get("typical_time"):
        described["typical_time"] = summary["typical_time"][:5]
        described["typical_spread_minutes"] = summary.get("typical_spread_minutes")
    if summary.get("top_domains"):
        described["top_domains"] = summary["top_domains"]
    if summary.get("top_areas"):
        described["top_areas"] = summary["top_areas"]
    return described


def propose(model: Any, provider: BaseProvider) -> HomeModeLabelResult:
    """Ask for a label per state. Never raises; a failed call just labels nothing."""
    result = HomeModeLabelResult()
    summaries = list(getattr(model, "state_summaries", []) or [])
    if not summaries or not provider.enabled:
        if not provider.enabled:
            result.error = "no LLM provider configured"
        return result

    known_states = {s["state"] for s in summaries}
    result.considered = len(summaries)
    prompt = json.dumps({"modes": [_describe(s) for s in summaries]}, indent=2, default=str)
    try:
        raw = provider.complete_json(SYSTEM_PROMPT, prompt)
    except LLMError as err:
        result.error = str(err)
        _LOGGER.warning("Home mode labelling failed: %s", err)
        return result

    proposals = raw.get("labels")
    if not isinstance(proposals, list):
        return result
    for entry in proposals:
        if not isinstance(entry, dict):
            continue
        state = entry.get("state")
        if not isinstance(state, int) or state not in known_states:
            result.rejected_unknown_state += 1
            continue
        label = clean_model_text(str(entry.get("label") or ""))
        if not label:
            continue
        if len(label) > MAX_LABEL_LENGTH:
            result.rejected_too_long += 1
            continue
        result.labels[state] = label
    return result


def apply_labels(model: Any, result: HomeModeLabelResult) -> int:
    """Attach surviving labels to ``model.state_summaries``, in place.

    Additive only, exactly like ``amminer.llm.classify.apply_to_signals``:
    every summary already has ``llm_label: None`` and
    ``llm_label_is_advisory: True`` from ``amminer.learn.home_mode`` itself,
    so a state this never mentions is left exactly as it was. Returns how
    many labels were applied, for the run report.
    """
    applied = 0
    for summary in getattr(model, "state_summaries", []) or []:
        label = result.labels.get(summary.get("state"))
        if label is None:
            continue
        summary["llm_label"] = label
        summary["llm_label_is_advisory"] = True
        applied += 1
    return applied
