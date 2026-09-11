"""Optional AI feature 2 - LLM proposes conditions, the backtester decides.

:mod:`amminer.miners.conditional` explains an action by testing each available
signal on its own with a single threshold.  That is deliberate and it is
verifiable, but it cannot express a *combination* ("when it is dark AND it is a
workday"), and it has no idea which combinations are worth trying.  A model does
have that idea, because it knows how houses are used: blinds go down against low
evening sun, heating comes on when it is cold and someone is home, the
dishwasher runs after dinner.

So the model is used as a **hypothesis generator** and nothing more.  For every
candidate the backtest rejected, it proposes condition sets that might explain
when the action really happens.  Each proposal is then rebuilt as a normal
candidate and put through the *same* backtester, with the same thresholds, as
everything else.  Proposals that do not clear the gate are discarded, and the
model is never told it was right.

The model therefore cannot surface anything.  It can only suggest something for
the deterministic engine to measure, which is the one job it is better at than
the statistics.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..backtest import backtest
from ..miners.base import Candidate, Condition
from ..util.text import clean_model_text
from ..util.timeutil import parse_time_of_day
from .provider import BaseProvider, LLMError

_LOGGER = logging.getLogger(__name__)

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

#: Condition kinds the backtester can actually simulate.  Anything else is
#: unverifiable, so the model is not allowed to propose it.
SUPPORTED_KINDS = ("state", "numeric_state", "time")

SYSTEM_PROMPT = """\
You suggest what might explain WHEN a household action happens.

You are given one action a person performs by hand, and the signals available in
their home. A statistical check has already shown that the action does NOT
happen on a simple schedule, so something else must explain it.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- Only use entity_ids from the "available_signals" list. Never invent one.
- Only use condition kinds: "state", "numeric_state", "time".
- "state" needs entity_id and state. "numeric_state" needs entity_id and
  above and/or below (numbers). "time" needs weekday and/or after/before
  as "HH:MM:SS".
- Each hypothesis is a SET of conditions that must ALL hold. Keep sets small:
  one or two conditions. Three is the maximum.
- Propose genuinely different explanations, not variations of one.
- Base them on how homes are actually used. If nothing plausible comes to mind,
  return an empty list - that is a valid and useful answer.

Shape:
{"hypotheses": [
  {"reason": "people heat when it is cold outside and someone is home",
   "conditions": [
     {"kind": "numeric_state", "entity_id": "sensor.outdoor_temp", "below": 10},
     {"kind": "state", "entity_id": "person.alex", "state": "home"}
   ]}
]}
"""


@dataclass
class Hypothesis:
    """One proposed explanation, plus what measuring it showed."""

    reason: str
    conditions: list[Condition] = field(default_factory=list)
    accepted: bool = False
    backtest: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "conditions": [c.as_dict() for c in self.conditions],
            "accepted": self.accepted,
            "backtest": self.backtest,
        }


@dataclass
class HypothesisResult:
    """Outcome of the propose-then-measure loop over rejected candidates."""

    considered: int = 0
    proposed: int = 0
    rejected_invalid: int = 0
    accepted: list[Candidate] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "proposed": self.proposed,
            "accepted": len(self.accepted),
            "rejected_invalid": self.rejected_invalid,
            "error": self.error,
            "attempts": self.attempts[:20],
        }


def _signal_catalogue(store, resolver, limit: int = 60) -> list[dict[str, Any]]:
    """Describe the signals a hypothesis may reference, with real value ranges.

    Giving the model the observed range of a numeric signal is what stops it
    proposing ``below: 10`` for a sensor that reads in kilowatts.
    """
    catalogue: list[dict[str, Any]] = []
    for entity_id in sorted(store.series):
        series = store.get(entity_id)
        if series is None or series.empty:
            continue
        info = resolver.resolve(entity_id) if resolver else None
        entry: dict[str, Any] = {
            "entity_id": entity_id,
            "name": info.name if info else entity_id,
        }
        if info is not None and info.area_name:
            entry["area"] = info.area_name
        if series.numeric:
            values = [v for v in series.values if isinstance(v, (int, float))]
            if values:
                entry["kind"] = "numeric"
                entry["min"] = round(min(values), 2)
                entry["max"] = round(max(values), 2)
                if info is not None and info.unit_of_measurement:
                    entry["unit"] = info.unit_of_measurement
        else:
            entry["kind"] = "categorical"
            entry["observed_states"] = [str(v) for v in series.distinct_values(8)]
        catalogue.append(entry)
        if len(catalogue) >= limit:
            break
    return catalogue


def parse_conditions(
    raw_conditions: Any, allowed_entities: set[str]
) -> list[Condition] | None:
    """Turn proposed conditions into real ones, or ``None`` if unusable.

    Rejects unknown entities, unsupported kinds and conditions with no actual
    constraint - a ``numeric_state`` with neither bound would silently match
    everything and make a hypothesis look better than it is.
    """
    if not isinstance(raw_conditions, list) or not raw_conditions:
        return None
    if len(raw_conditions) > 3:
        return None

    conditions: list[Condition] = []
    for raw in raw_conditions:
        if not isinstance(raw, dict):
            return None
        kind = str(raw.get("kind") or "").strip().lower()
        if kind not in SUPPORTED_KINDS:
            return None

        if kind == "time":
            weekday = raw.get("weekday")
            if isinstance(weekday, str):
                weekday = [weekday]
            weekday = [
                str(d).strip().lower()[:3]
                for d in (weekday or [])
                if str(d).strip().lower()[:3] in WEEKDAYS
            ]
            # A time bound that does not parse is rejected rather than
            # dropped: the backtester cannot evaluate it, and left in place it
            # would be rendered straight into the applied automation's YAML.
            after_raw, before_raw = raw.get("after"), raw.get("before")
            after = parse_time_of_day(after_raw)
            before = parse_time_of_day(before_raw)
            if after_raw is not None and after is None:
                return None
            if before_raw is not None and before is None:
                return None
            if not weekday and not after and not before:
                return None
            conditions.append(
                Condition(
                    kind="time",
                    weekday=weekday,
                    after=after,
                    before=before,
                    source="LLM hypothesis",
                )
            )
            continue

        entity_id = raw.get("entity_id")
        if not isinstance(entity_id, str) or entity_id not in allowed_entities:
            return None

        if kind == "state":
            state = raw.get("state")
            if not isinstance(state, (str, int, float)) or str(state) == "":
                return None
            conditions.append(
                Condition(
                    kind="state",
                    entity_id=entity_id,
                    state=str(state),
                    source="LLM hypothesis",
                )
            )
        else:  # numeric_state
            above, below = raw.get("above"), raw.get("below")
            above = float(above) if isinstance(above, (int, float)) else None
            below = float(below) if isinstance(below, (int, float)) else None
            if above is None and below is None:
                return None
            conditions.append(
                Condition(
                    kind="numeric_state",
                    entity_id=entity_id,
                    above=above,
                    below=below,
                    source="LLM hypothesis",
                )
            )
    return conditions or None


def _variant(candidate: Candidate, hypothesis: Hypothesis) -> Candidate:
    """A copy of *candidate* with the hypothesised conditions added."""
    variant = Candidate(
        miner=f"{candidate.miner}+hypothesis",
        title=candidate.title,
        triggers=list(candidate.triggers),
        conditions=list(candidate.conditions) + list(hypothesis.conditions),
        actions=list(candidate.actions),
        evidence=candidate.evidence,
        score=candidate.score,
        extra=dict(candidate.extra),
    )
    variant.extra["hypothesis"] = {
        "reason": hypothesis.reason,
        "origin_candidate": candidate.id,
        "conditions": [c.as_dict() for c in hypothesis.conditions],
    }
    return variant


def propose_and_verify(
    rejected: Sequence[Candidate],
    changes: Sequence[Any],
    store,
    options,
    window: tuple[float, float],
    provider: BaseProvider,
    resolver=None,
    overrides: Sequence[Any] = (),
) -> HypothesisResult:
    """Ask for explanations of rejected candidates, then measure each.  Never raises."""
    result = HypothesisResult()
    if not provider.enabled:
        result.error = "no LLM provider configured"
        return result

    workable = [
        candidate
        for candidate in rejected
        if candidate.actions
        and candidate.backtest
        and candidate.backtest.get("simulated")
        # A rule that never fires cannot be repaired by ADDING a condition;
        # conditions only ever remove fires.
        and (candidate.backtest.get("false_fires") or 0) > 0
    ]
    workable.sort(key=lambda c: c.score, reverse=True)
    workable = workable[: max(int(options.llm_hypothesis_candidates), 0)]
    if not workable:
        return result

    catalogue = _signal_catalogue(store, resolver)
    if not catalogue:
        result.error = "no signals available to build a hypothesis from"
        return result
    allowed = {entry["entity_id"] for entry in catalogue}

    for candidate in workable:
        result.considered += 1
        # Never let a rule be "explained" by the entity it acts on.
        usable = [e for e in catalogue if e["entity_id"] not in candidate.target_entities]
        prompt = json.dumps(
            {
                "action": candidate.describe(resolver),
                "target_entities": candidate.target_entities,
                "why_rejected": candidate.backtest.get("reason"),
                "measured": {
                    "correct_fires": candidate.backtest.get("true_fires"),
                    "unwanted_fires": candidate.backtest.get("false_fires"),
                    "missed_actions": candidate.backtest.get("missed"),
                },
                "max_hypotheses": int(options.llm_hypotheses_per_candidate),
                "available_signals": usable,
            },
            indent=2,
            default=str,
        )
        try:
            raw = provider.complete_json(SYSTEM_PROMPT, prompt)
        except LLMError as err:
            result.error = str(err)
            _LOGGER.warning("Hypothesis generation failed: %s", err)
            break

        proposals = raw.get("hypotheses")
        if not isinstance(proposals, list):
            continue
        allowed_here = allowed - set(candidate.target_entities)

        for proposal in proposals[: max(int(options.llm_hypotheses_per_candidate), 0)]:
            if not isinstance(proposal, dict):
                continue
            conditions = parse_conditions(proposal.get("conditions"), allowed_here)
            if conditions is None:
                result.rejected_invalid += 1
                continue
            result.proposed += 1
            hypothesis = Hypothesis(
                # The reason is persisted into the suggestion payload and
                # rendered on the card, so it is capped where it enters.
                reason=clean_model_text(
                    proposal.get("reason"), fallback="no reason given"
                ),
                conditions=conditions,
            )
            variant = _variant(candidate, hypothesis)
            outcome = backtest(variant, changes, store, options, window, overrides)
            hypothesis.backtest = outcome.as_dict()
            hypothesis.accepted = outcome.passed
            variant.backtest = hypothesis.backtest
            variant.extra["hypothesis"]["backtest"] = hypothesis.backtest

            result.attempts.append(
                {
                    "origin": candidate.title,
                    "reason": hypothesis.reason,
                    "accepted": hypothesis.accepted,
                    "summary": outcome.summary(),
                }
            )
            if outcome.passed:
                # Re-score on measured precision, exactly like every other
                # surfaced candidate; the model's confidence plays no part.
                if outcome.precision is not None:
                    variant.score = round((candidate.score + outcome.precision) / 2.0, 4)
                variant.description = (
                    f"{candidate.describe(resolver)} - refined with a condition suggested "
                    f"by the assistant ({hypothesis.reason}) and then verified against "
                    f"your history."
                )
                result.accepted.append(variant)

    _LOGGER.info(
        "Hypotheses: %d proposed over %d rejected candidates, %d verified and surfaced",
        result.proposed,
        result.considered,
        len(result.accepted),
    )
    return result
