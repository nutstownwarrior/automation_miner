"""Miner C - association-rule mining over co-occurring state changes.

Transactions are sliding windows over the event stream: everything that happened
within ``association_window_seconds`` forms one basket.  ``mlxtend``'s FP-Growth
then yields ``{A} -> {B}`` rules with support/confidence/lift.

Rules are kept only when the consequent is an *actionable* entity the user
touched by hand, and the antecedent is a different entity - i.e. rules that can
become a real "when A, then B" automation.

A basket is a set, so it says nothing about order.  FP-Growth on a co-occurring
pair yields ``{A} -> {B}`` and ``{B} -> {A}`` with *identical* support,
confidence and lift, by construction, for every strong pair - so the miner would
otherwise propose "turn the pump on because the light came on" with exactly the
evidence of the rule that is actually true, and a user reading two cards with
the same numbers could accept both and build a loop.  Order is therefore
recorded separately, while the baskets are being built, and a rule has to show
that its antecedent really did come first.
"""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from ..config import ACTIONABLE_DOMAINS, Options
from ..recorderdb.models import Cause, StateChange
from .base import Action, Candidate, Evidence, Trigger
from .time_of_day import service_for

_LOGGER = logging.getLogger(__name__)

#: Domains usable as an antecedent (a trigger) even though you cannot act on them.
TRIGGER_DOMAINS = (
    "binary_sensor",
    "person",
    "device_tracker",
    "sensor",
    "input_boolean",
    "sun",
    "calendar",
    "light",
    "switch",
    "cover",
    "lock",
    "climate",
    "media_player",
    "fan",
    "alarm_control_panel",
)


#: How much more often the antecedent must precede the consequent than follow it
#: before the rule is allowed to claim a direction.
MIN_DIRECTION_RATIO = 2.0


def _item(change: StateChange) -> str:
    return f"{change.entity_id}={change.state.lower()}"


def is_categorical(change: StateChange) -> bool:
    """True when a state value is a usable discrete symbol.

    A numeric sensor reading ("3.6") is a terrible association item - every
    reading is its own symbol, so support collapses and any "rule" found is an
    artefact.  Numeric series belong to the motif and conditional miners.
    """
    return change.numeric is None


def build_transactions(
    changes: Sequence[StateChange],
    window_seconds: float,
    options: Options,
) -> tuple[list[list[str]], dict[str, int], dict[tuple[str, str], int]]:
    """Group changes into overlapping baskets, one per "interesting" change.

    Anchoring a basket on each *human* change (rather than on fixed clock
    windows) keeps the transaction count proportional to user activity and
    avoids thousands of empty midnight baskets.

    Returns ``(baskets, item counts, ordered-pair counts)``.  The last one is
    what the baskets themselves cannot express: for each ordered pair of items
    in a basket, how many baskets had the first strictly before the second.
    Collect it here or not at all - once a basket is a set, the timestamps are
    gone.
    """
    usable = [
        change
        for change in changes
        if change.is_transition
        and not options.is_excluded(change.entity_id)
        and change.domain in set(TRIGGER_DOMAINS) | set(ACTIONABLE_DOMAINS)
        and change.state.lower() not in ("unknown", "unavailable", "")
        and change.cause in (Cause.HUMAN, Cause.DEVICE, Cause.AUTOMATION)
        and is_categorical(change)
    ]
    usable.sort(key=lambda c: c.ts)

    anchors = [i for i, change in enumerate(usable) if change.cause is Cause.HUMAN]
    transactions: list[list[str]] = []
    item_counts: dict[str, int] = defaultdict(int)

    order_counts: dict[tuple[str, str], int] = defaultdict(int)

    for anchor in anchors:
        centre = usable[anchor].ts
        # First time each item appears in this basket; a repeated item is one
        # symbol, and its earliest occurrence is what ordering means.
        first_seen: dict[str, float] = {}
        i = anchor
        while i >= 0 and centre - usable[i].ts <= window_seconds:
            first_seen[_item(usable[i])] = usable[i].ts
            i -= 1
        i = anchor + 1
        while i < len(usable) and usable[i].ts - centre <= window_seconds:
            first_seen.setdefault(_item(usable[i]), usable[i].ts)
            i += 1
        basket = set(first_seen)
        if len(basket) >= 2:
            transactions.append(sorted(basket))
            for item in basket:
                item_counts[item] += 1
            ordered = sorted(first_seen.items(), key=lambda kv: kv[1])
            for position, (earlier, earlier_ts) in enumerate(ordered):
                for later, later_ts in ordered[position + 1:]:
                    if later_ts > earlier_ts:
                        order_counts[(earlier, later)] += 1
    return transactions, dict(item_counts), dict(order_counts)


def one_hot_encode(transactions: Sequence[Sequence[str]], pd) -> Any:
    """One-hot encode baskets into the boolean frame FP-Growth expects.

    This replaces ``mlxtend.preprocessing.TransactionEncoder``, whose module
    imports scikit-learn.  scikit-learn publishes no musllinux wheels, so
    depending on it would force a from-source build of scikit-learn (and pull in
    matplotlib) inside the Alpine add-on image.  ``mlxtend.frequent_patterns``
    itself needs only numpy/pandas/scipy, so encoding here keeps the image
    wheel-only.  Columns are sorted so the frame is deterministic.
    """
    items = sorted({item for transaction in transactions for item in transaction})
    index = {item: position for position, item in enumerate(items)}
    rows = []
    for transaction in transactions:
        row = [False] * len(items)
        for item in transaction:
            row[index[item]] = True
        rows.append(row)
    return pd.DataFrame(rows, columns=items, dtype=bool)


def _rules_kwargs(options: Options, transaction_count: int) -> dict[str, Any]:
    """Arguments for ``association_rules``, adapted to the installed mlxtend.

    mlxtend 0.23.x made ``num_itemsets`` a *required* argument and 0.24+ made it
    optional again with a misleading default of 1.  Passing the real transaction
    count whenever the parameter exists is correct on every version, and keeps a
    dependency resolving to a different release from breaking the miner.
    """
    kwargs: dict[str, Any] = {
        "metric": "confidence",
        "min_threshold": options.min_confidence,
    }
    try:
        from mlxtend.frequent_patterns import association_rules

        if "num_itemsets" in inspect.signature(association_rules).parameters:
            kwargs["num_itemsets"] = transaction_count
    except (ImportError, TypeError, ValueError):  # pragma: no cover - defensive
        pass
    return kwargs


def _rules_from_transactions(
    transactions: list[list[str]], options: Options
) -> list[dict[str, Any]]:
    """Run FP-Growth + association rules, degrading if mlxtend is unavailable."""
    if len(transactions) < 10:
        return []
    try:
        import pandas as pd
        from mlxtend.frequent_patterns import association_rules, fpgrowth
    except ImportError:  # pragma: no cover - mlxtend is a hard requirement in the image
        _LOGGER.warning("mlxtend/pandas unavailable; skipping association mining")
        return []

    frame = one_hot_encode(transactions, pd)
    frequent = fpgrowth(frame, min_support=options.min_support, use_colnames=True, max_len=2)
    if frequent.empty:
        return []
    try:
        rules = association_rules(frequent, **_rules_kwargs(options, len(transactions)))
    except (ValueError, KeyError, TypeError):
        return []
    if rules.empty:
        return []
    rules = rules[rules["lift"] >= options.min_lift]

    out: list[dict[str, Any]] = []
    for _, row in rules.iterrows():
        antecedents = sorted(row["antecedents"])
        consequents = sorted(row["consequents"])
        if len(antecedents) != 1 or len(consequents) != 1:
            continue
        out.append(
            {
                "antecedent": antecedents[0],
                "consequent": consequents[0],
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
            }
        )
    return out


def mine(
    changes: Sequence[StateChange],
    options: Options,
    window: tuple[float, float] | None = None,
    resolver=None,
) -> list[Candidate]:
    """Mine "when A happens, B usually follows" candidates."""
    transactions, _counts, order_counts = build_transactions(
        changes, options.association_window_seconds, options
    )
    rules = _rules_from_transactions(transactions, options)
    if not rules:
        _LOGGER.info("association miner produced 0 candidates (%d transactions)", len(transactions))
        return []

    # How often each item was *human*-caused, so we only automate real actions.
    human_items: dict[str, int] = defaultdict(int)
    for change in changes:
        if change.cause is Cause.HUMAN and change.is_transition:
            human_items[_item(change)] += 1

    start_ts = window[0] if window else min((c.ts for c in changes), default=0.0)
    end_ts = window[1] if window else max((c.ts for c in changes), default=0.0)

    candidates: list[Candidate] = []
    seen: set[tuple[str, str, str]] = set()
    for rule in rules:
        antecedent, consequent = rule["antecedent"], rule["consequent"]
        if antecedent == consequent:
            continue
        a_entity, _, a_state = antecedent.partition("=")
        c_entity, _, c_state = consequent.partition("=")
        if a_entity == c_entity:
            continue
        if c_entity.split(".", 1)[0] not in ACTIONABLE_DOMAINS:
            continue
        if human_items.get(consequent, 0) < max(options.min_occurrences // 2, 2):
            # The consequent is not something the user actually does by hand.
            continue
        forward = order_counts.get((antecedent, consequent), 0)
        backward = order_counts.get((consequent, antecedent), 0)
        if forward < backward:
            # The antecedent consistently arrives *after* the consequent, so it
            # is the effect being offered as the cause - "when the power draw
            # rises, switch the heater on".
            continue
        # If the mirror rule could also be emitted, both cards would carry the
        # same support, confidence and lift, and a user could accept both and
        # build a loop.  That case needs a clear arrow, not merely a tie.  When
        # the mirror rule is impossible anyway - the antecedent is a sensor or a
        # person, which nothing can act on - simultaneity carries no
        # information and is not held against the rule.
        mirror_possible = (
            a_entity.split(".", 1)[0] in ACTIONABLE_DOMAINS
            and human_items.get(antecedent, 0) >= max(options.min_occurrences // 2, 2)
        )
        if mirror_possible and forward < backward * MIN_DIRECTION_RATIO:
            continue
        if (a_entity, a_state, c_entity) in seen:
            continue
        service = service_for(c_entity, c_state)
        if service is None:
            continue
        seen.add((a_entity, a_state, c_entity))
        service_name, service_data = service

        a_name = resolver.name_of(a_entity) if resolver else a_entity
        c_name = resolver.name_of(c_entity) if resolver else c_entity
        verb = service_name.split(".", 1)[-1].replace("_", " ")

        candidates.append(
            Candidate(
                miner="association",
                title=f"When {a_name} becomes '{a_state}', {verb} {c_name}",
                triggers=[Trigger(kind="state", entity_id=a_entity, to_state=a_state)],
                actions=[
                    Action(service=service_name, entity_id=c_entity, data=dict(service_data))
                ],
                evidence=Evidence(
                    occurrences=int(rule["support"] * len(transactions)),
                    opportunities=len(transactions),
                    support=rule["support"],
                    confidence=rule["confidence"],
                    lift=rule["lift"],
                    window_start_ts=start_ts,
                    window_end_ts=end_ts,
                    window_days=(end_ts - start_ts) / 86400.0,
                    notes=[
                        f"'{antecedent}' and '{consequent}' co-occurred within "
                        f"{options.association_window_seconds}s in "
                        f"{rule['support']:.1%} of {len(transactions)} activity windows.",
                        f"Lift {rule['lift']:.2f} means this is {rule['lift']:.1f}x more often "
                        "than chance would predict.",
                    ],
                    extra={"rule": rule},
                ),
                score=round(min(rule["confidence"] * min(rule["lift"] / 3.0, 1.0), 1.0), 4),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    _LOGGER.info(
        "association miner produced %d candidates from %d transactions",
        len(candidates),
        len(transactions),
    )
    return candidates
