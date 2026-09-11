"""Optional AI feature 7 - the evidence in a sentence a person reads.

A card's evidence line today reads::

    30 of 34, consistency 88%, confidence 100%, lift 2.00, ±6 min

That is a statistician's sentence, and worse: as :mod:`amminer.miners.base`
records, ``confidence`` and ``consistency`` carry *different quantities*
depending on which miner filled them in, so the numbers are not even comparable
between two cards on the same page.

So a model is asked to write the "why" in plain language.  This is rendering,
not judgement, and it is held to that:

* it may not change a score, a verdict, a backtest or any evidence value,
* it may not state a number that is not in the evidence it was given - a
  plausible-sounding invented figure is worse than the raw line it replaces,
* the original numbers stay on the card underneath.

A sentence that fails the number check is dropped, not corrected: a wrong number
in a sentence about why you should trust something is the one error that cannot
be tolerated here.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..miners.base import Candidate
from ..util.text import clean_model_text
from .provider import BaseProvider, LLMError

_LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You rewrite the evidence behind a home-automation suggestion as one plain
sentence a homeowner would understand.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- Only use the "id" values given. Never invent one.
- ONE sentence per suggestion. Two at most. No bullet points.
- You may ONLY use numbers that appear in the evidence you were given. Do not
  round them, do not compute new ones, do not estimate.
- Prefer plain counts over ratios: "on 30 of the 34 weekdays" reads better than
  "88% consistency".
- Do not say whether the suggestion is good. You are explaining the evidence,
  not judging it.
- Do not repeat the title.

Shape:
{"explanations": [{"id": "abc123",
                   "text": "You switched this on at about 06:30 on 30 of the 34 weekdays in the window."}]}
"""

#: Numbers in the sentence must come from here.  A leading minus is part of the
#: token: none of these quantities is ever negative, so "-30" must not pass on
#: the strength of "30".
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

#: A digit-group separator inside a number, so "1,234" is read as one figure
#: instead of as "1" and "234" - which rejected a perfectly correct sentence.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")

#: "30 of 34" / "30 out of the 34".  These two numbers are a *pair* and have to
#: be checked as one, because each of them can be individually permitted while
#: the pair says something the evidence never said.
_PAIR = re.compile(
    r"(-?\d+(?:\.\d+)?)\s+(?:of|out of)\s+(?:the\s+)?(-?\d+(?:\.\d+)?)", re.IGNORECASE
)

#: Written-out numbers sidestep every check above, so a sentence containing one
#: is dropped rather than waved through: unverifiable is not the same as
#: correct, and this feature's whole job is to refuse the difference.
_NUMBER_WORDS = frozenset(
    """zero one two three four five six seven eight nine ten eleven twelve
    thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty
    forty fifty sixty seventy eighty ninety hundred thousand half twice double
    dozen""".split()
)
_WORD = re.compile(r"[a-z]+")


def _forms(number: float, ratio: bool) -> set[str]:
    """Every way a permitted quantity may reasonably be written.

    Rounded forms are deliberately withheld from a raw 0-1 ratio.  Adding them
    put "1" (and often "0") into the allowed set for almost every candidate -
    ``f"{0.88:.0f}"`` is ``"1"`` - which let the single most common invented
    figure in a sentence like "this happened only 1 time" through unchallenged.
    A ratio belongs in a sentence as a percentage or as itself, never rounded to
    a bare integer.
    """
    forms = {f"{number:g}"}
    if ratio:
        return forms
    forms.update({f"{number:.0f}", f"{number:.1f}", f"{number:.2f}"})
    forms.add(str(int(number)) if number == int(number) else f"{number}")
    return forms


def _permitted_numbers(candidate: Candidate) -> set[str]:
    """Every number a sentence about this candidate is allowed to contain."""
    evidence = candidate.evidence
    values: list[tuple[Any, bool]] = [
        (evidence.occurrences, False),
        (evidence.opportunities, False),
        (evidence.window_days, False),
        (evidence.spread_minutes, False),
    ]
    for ratio in (evidence.consistency, evidence.support, evidence.confidence):
        if ratio is not None:
            values.extend([(ratio, True), (ratio * 100, False)])
    if evidence.lift is not None:
        values.append((evidence.lift, False))
    if candidate.backtest:
        for key in ("true_fires", "false_fires", "missed", "total_fires", "window_days"):
            values.append((candidate.backtest.get(key), False))
        for key in ("precision", "recall"):
            ratio = candidate.backtest.get(key)
            if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
                values.extend([(ratio, True), (ratio * 100, False)])

    allowed: set[str] = set()
    for value, ratio in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        allowed.update(_forms(number, ratio))
    # Times of day and small ordinals in the rule itself are fair game.
    for trigger in candidate.triggers:
        for part in _NUMBER.findall(str(trigger.at or "")):
            allowed.add(part.lstrip("0") or "0")
            allowed.add(part)
    return allowed


def _permitted_pairs(candidate: Candidate) -> set[tuple[str, str]]:
    """The "N of M" figures the evidence actually supports, as pairs."""
    pairs: list[tuple[Any, Any]] = [
        (candidate.evidence.occurrences, candidate.evidence.opportunities)
    ]
    if candidate.backtest:
        pairs.append(
            (candidate.backtest.get("true_fires"), candidate.backtest.get("total_fires"))
        )
    out: set[tuple[str, str]] = set()
    for left, right in pairs:
        if left is None or right is None:
            continue
        try:
            out.add((f"{float(left):g}", f"{float(right):g}"))
        except (TypeError, ValueError):
            continue
    return out


def _normalise(token: str) -> str:
    try:
        return f"{float(token):g}"
    except ValueError:
        return token


def invented_numbers(text: str, candidate: Candidate) -> list[str]:
    """Numbers in *text* that the evidence does not support.

    Three separate ways a sentence can be wrong about a number, all of which
    drop the sentence: a figure that appears nowhere in the evidence, a figure
    that appears but is written as a word so it cannot be checked, and a pair of
    individually-permitted figures asserted about each other.
    """
    text = _THOUSANDS.sub("", text)
    allowed = _permitted_numbers(candidate)
    out: list[str] = []

    for raw in _NUMBER.findall(text):
        normalised = {raw, raw.lstrip("0") or "0", _normalise(raw)}
        if not (normalised & allowed):
            out.append(raw)

    # "30 of 34" is a claim about a ratio, not two independent numbers.  Both
    # halves can be permitted and the pair still assert something the evidence
    # never said - quoting the window length as the number of occurrences, say.
    pairs = _permitted_pairs(candidate)
    for left, right in _PAIR.findall(text):
        if (_normalise(left), _normalise(right)) not in pairs:
            out.extend(part for part in (left, right) if part not in out)

    words = set(_WORD.findall(text.lower()))
    out.extend(sorted(words & _NUMBER_WORDS))
    return out


@dataclass
class ExplanationResult:
    """Sentences that survived the number check."""

    texts: dict[str, str] = field(default_factory=dict)
    rejected_numbers: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)
    explained: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "explained": len(self.texts),
            "asked_for": self.explained,
            "rejected_invented_numbers": len(self.rejected_numbers),
            "invented_numbers": self.rejected_numbers[:20],
            "unknown_ids": self.unknown_ids[:20],
            "error": self.error,
        }


#: The evidence fields a model may see.  An allowlist, not a blocklist: popping
#: ``samples`` was not enough, because ``extra`` carries raw epoch timestamps of
#: its own for some miners (a motif's ``start_ts``, a stale automation's
#: ``last_triggered``), and the window bounds are exact timestamps too.  A
#: blocklist here has to be updated every time a miner adds a field, and is
#: wrong in the meantime.
EVIDENCE_FIELDS = (
    "occurrences",
    "opportunities",
    "consistency",
    "support",
    "confidence",
    "lift",
    "spread_minutes",
    "window_days",
    "notes",
    "summary",
)


def _describe(candidate: Candidate, resolver=None) -> dict[str, Any]:
    full = candidate.evidence.as_dict()
    evidence = {key: full[key] for key in EVIDENCE_FIELDS if key in full}
    return {
        "id": candidate.id,
        "title": candidate.title,
        "rule": candidate.describe(resolver),
        "evidence": evidence,
        "backtest": {
            key: candidate.backtest.get(key)
            for key in ("true_fires", "false_fires", "missed", "precision", "recall",
                        "window_days")
        }
        if candidate.backtest
        else {},
    }


def explain(
    candidates: Sequence[Candidate],
    provider: BaseProvider,
    resolver=None,
    batch_size: int = 20,
) -> ExplanationResult:
    """Write a plain-language why for each candidate.  Never raises."""
    result = ExplanationResult()
    if not candidates:
        return result
    if not provider.enabled:
        result.error = "no LLM provider configured"
        return result

    by_id = {c.id: c for c in candidates}
    step = max(batch_size, 1)
    for start in range(0, len(candidates), step):
        batch = list(candidates)[start : start + step]
        prompt = json.dumps(
            {"suggestions": [_describe(c, resolver) for c in batch]},
            indent=2,
            default=str,
        )
        try:
            raw = provider.complete_json(SYSTEM_PROMPT, prompt)
        except LLMError as err:
            result.error = str(err)
            _LOGGER.warning("Explanations failed: %s", err)
            break
        result.explained += len(batch)

        entries = raw.get("explanations")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            candidate_id = entry.get("id")
            if not isinstance(candidate_id, str) or candidate_id not in by_id:
                if isinstance(candidate_id, str):
                    result.unknown_ids.append(candidate_id)
                continue
            text = clean_model_text(entry.get("text"), limit=400)
            if not text:
                continue
            invented = invented_numbers(text, by_id[candidate_id])
            if invented:
                # Dropped rather than repaired: a wrong number in the sentence
                # explaining why to trust something is the worst possible error.
                result.rejected_numbers.extend(invented)
                continue
            result.texts[candidate_id] = text
    return result


def apply_explanations(
    candidates: Sequence[Candidate], result: ExplanationResult
) -> Sequence[Candidate]:
    """Attach sentences in place.  Changes nothing else about a candidate."""
    for candidate in candidates:
        text = result.texts.get(candidate.id)
        if text:
            candidate.extra["explanation"] = text
    return candidates
