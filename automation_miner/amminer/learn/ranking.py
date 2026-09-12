"""Learn a calibrated acceptance probability, and use it only to order cards.

Every miner scores its own candidates with its own arithmetic - association
rules by ``confidence * lift``, time-of-day habits by ``consistency * hits``,
staleness by raw age.  :class:`~amminer.miners.base.Evidence` documents at
length that these numbers are not comparable across miners.  Sorting them
against each other on the index page is a collation, not a ranking.

This module replaces that ordering - never the numbers themselves, which stay
on the card as supporting detail - with one calibrated quantity: the estimated
probability that *this user* accepts *this* suggestion, learned from their own
accept/dismiss history.

Two things make this safe to turn on for an install with no history at all:

**A hand-set prior.**  A brand-new instance has zero labels, and the ordering
has to start somewhere sane rather than random.  :data:`PRIOR_WEIGHTS` encodes
beliefs this project already holds elsewhere in the codebase (a holdout-
validated candidate is better than an in-sample one, more conflicts are worse,
a risky-domain action deserves more caution) as a starting point, shrunk
towards the personal fit as decisions accumulate.

**A guard that never lets a personal model be worse than doing nothing.**  A
few noisy labels can easily fit worse than the prior they are meant to
improve on, so the personal model has to beat the prior on held-out log-loss
before it is used at all; otherwise the prior stands, honestly labelled as
such.

The one rule that matters most: **this number is for ranking and display
only.**  Nothing here may decide whether a suggestion may be surfaced or
applied - that is the backtest gate's job (:mod:`amminer.backtest`) and the
conflict checker's job (:mod:`amminer.conflicts`), and only theirs.  A
candidate that failed its backtest has no probability computed for it at all,
because nothing here ever sees `rejected` candidates; a candidate that passed
keeps whatever severity of conflict it has regardless of how likely it is to
be accepted.  See ``tests/test_ranking.py`` for the test that pins this down.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import RISKY_DOMAINS
from ..miners.base import Candidate

#: Bumped whenever the feature vector's shape or meaning changes.  A weight
#: vector fit under an older version is not merely stale, it is meaningless -
#: index 7 might have been ``lift`` and is now ``holdout_days`` - so a mismatch
#: must discard the stored model outright rather than apply it.
FEATURE_SCHEMA_VERSION = 1

#: A small, fixed vocabulary.  Bounded so the one-hot encoding has a stable
#: width forever; anything else falls into "other" rather than growing the
#: feature vector every time a new integration shows up.
DOMAIN_VOCAB: tuple[str, ...] = (
    "light",
    "switch",
    "climate",
    "cover",
    "lock",
    "fan",
    "media_player",
    "sensor",
    "binary_sensor",
    "person",
    "device_tracker",
)

#: The miners that currently exist, plus "scene" (built from several of their
#: outputs).  A name outside this list - most often ``"<miner>+hypothesis"``,
#: which an AI-proposed condition tacks onto whatever miner it rescued - falls
#: into "other" rather than growing the vocabulary per combination, which
#: would otherwise be unbounded.
MINER_VOCAB: tuple[str, ...] = (
    "time_of_day",
    "association",
    "sequence",
    "conditional",
    "motif",
    "energy_shift",
    "stale_automation",
    "unused_entity",
    "scene",
)

_SEVERITY_RANK: dict[str, int] = {"info": 1, "warning": 2, "error": 3}

#: Numeric features.  Each contributes two columns: the (possibly log-scaled)
#: value, and a missing-indicator that is 1 exactly when the underlying field
#: was ``None`` - never a bare 0, which would be indistinguishable from a real
#: zero (an association rule really can have zero prior occurrences; a
#: candidate with no backtest yet does not, and those are different facts).
_NUMERIC_FEATURES: tuple[str, ...] = (
    "occurrences",
    "opportunities",
    "consistency",
    "support",
    "confidence",
    "lift",
    "spread_minutes",
    "precision",
    "recall",
    "false_fires_per_week",
    "true_fires",
    "holdout_days",
    "n_triggers",
    "n_conditions",
    "n_actions",
    "n_entities",
    "conflict_count",
    "worst_conflict_severity",
    "seen_count",
)

#: Heavy-tailed counts get ``log1p`` so one candidate with 400 occurrences does
#: not dwarf every other feature; ratios and small structural counts do not
#: need it.
_LOG_SCALE: frozenset[str] = frozenset(
    {
        "occurrences",
        "opportunities",
        "lift",
        "spread_minutes",
        "false_fires_per_week",
        "true_fires",
        "holdout_days",
        "n_entities",
        "conflict_count",
        "seen_count",
    }
)

#: Binary/categorical features carry their own meaning at 0 - "no backtest" is
#: exactly ``has_backtest = 0``, not a missing value - so none of these need a
#: separate missing-indicator column.
_BINARY_FEATURES: tuple[str, ...] = ("holdout_validated", "has_backtest", "risky_domain")


def feature_names() -> tuple[str, ...]:
    """The feature vector's columns, in the fixed order every array uses."""
    names: list[str] = []
    for name in _NUMERIC_FEATURES:
        names.append(name)
        names.append(f"{name}_missing")
    names.extend(_BINARY_FEATURES)
    names.extend(f"domain_{d}" for d in DOMAIN_VOCAB)
    names.append("domain_other")
    names.extend(f"miner_{m}" for m in MINER_VOCAB)
    names.append("miner_other")
    return tuple(names)


_FEATURE_NAMES: tuple[str, ...] = feature_names()
_FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(_FEATURE_NAMES)}


def extract_features(candidate: Candidate, seen_count: int | None = None) -> np.ndarray:
    """A miner-agnostic feature vector for *candidate*.

    ``seen_count`` is passed in rather than read off the candidate, because it
    lives on the store's suggestion row, not on :class:`Candidate` - and,
    critically, the caller controls *which* seen_count this is.  Scoring a
    candidate about to be shown uses the live count; building a training
    example from a past decision must use the count as it stood at the moment
    of that decision, or the model would be learning from something that only
    existed after the label was made (see ``Store.ranking_labels``).
    """
    values = np.zeros(len(_FEATURE_NAMES), dtype=np.float64)

    def put(name: str, raw: float | int | None) -> None:
        idx = _FEATURE_INDEX[name]
        if raw is None:
            values[idx] = 0.0
            values[_FEATURE_INDEX[f"{name}_missing"]] = 1.0
            return
        v = float(raw)
        values[idx] = math.log1p(max(v, 0.0)) if name in _LOG_SCALE else v
        values[_FEATURE_INDEX[f"{name}_missing"]] = 0.0

    ev = candidate.evidence
    put("occurrences", ev.occurrences)
    put("opportunities", ev.opportunities)
    put("consistency", ev.consistency)
    put("support", ev.support)
    put("confidence", ev.confidence)
    put("lift", ev.lift)
    put("spread_minutes", ev.spread_minutes)

    backtest = candidate.backtest or {}
    has_backtest = bool(candidate.backtest)
    put("precision", backtest.get("precision") if has_backtest else None)
    put("recall", backtest.get("recall") if has_backtest else None)
    put("false_fires_per_week", backtest.get("false_fires_per_week") if has_backtest else None)
    put("true_fires", backtest.get("true_fires") if has_backtest else None)
    put("holdout_days", backtest.get("holdout_days") if has_backtest else None)

    put("n_triggers", len(candidate.triggers))
    put("n_conditions", len(candidate.conditions))
    put("n_actions", len(candidate.actions))
    put("n_entities", len(candidate.entities))

    conflicts = candidate.conflicts or []
    put("conflict_count", len(conflicts))
    worst = 0
    for conflict in conflicts:
        worst = max(worst, _SEVERITY_RANK.get(conflict.get("severity"), 0))
    put("worst_conflict_severity", worst)

    put("seen_count", seen_count)

    values[_FEATURE_INDEX["has_backtest"]] = 1.0 if has_backtest else 0.0
    values[_FEATURE_INDEX["holdout_validated"]] = (
        1.0 if (has_backtest and backtest.get("validation") == "holdout") else 0.0
    )

    domains = {e.split(".", 1)[0] for e in candidate.entities if "." in e}
    values[_FEATURE_INDEX["risky_domain"]] = 1.0 if domains & set(RISKY_DOMAINS) else 0.0
    for domain in DOMAIN_VOCAB:
        values[_FEATURE_INDEX[f"domain_{domain}"]] = 1.0 if domain in domains else 0.0
    values[_FEATURE_INDEX["domain_other"]] = 1.0 if (domains - set(DOMAIN_VOCAB)) else 0.0

    miner = candidate.miner if candidate.miner in MINER_VOCAB else None
    for name in MINER_VOCAB:
        values[_FEATURE_INDEX[f"miner_{name}"]] = 1.0 if miner == name else 0.0
    values[_FEATURE_INDEX["miner_other"]] = 1.0 if miner is None else 0.0

    return values


# ----------------------------------------------------------------------
# The hand-set prior.  This is not fit on anything - see the module docstring
# and ``testing/synthetic.py``'s own warning that it generates behaviour, not
# preferences.  Every weight below is a belief this codebase already holds
# somewhere else, written down once so a new install has a sane starting
# ordering instead of an arbitrary one.
#: Most suggestions shown are not accepted; a negative bias means "assume
#: dismissal" until the features argue otherwise, which matches this
#: project's whole posture (surface only what cleared a strict gate, then let
#: the user decide).
PRIOR_BIAS = -1.0

PRIOR_WEIGHTS: dict[str, float] = {
    # --- evidence: miners/base.py:158 warns these mean different things per
    # miner, but the *direction* of each is still something this project
    # believes regardless of which miner produced it. ---------------------
    # More raw occurrences is weak evidence of a real habit, but says nothing
    # about quality without consistency alongside it - a small, cautious
    # weight rather than none at all.
    "occurrences": 0.15,
    "occurrences_missing": 0.0,
    # Opportunities alone conflates "observed for a long time" with "a strong
    # habit"; whatever it contributes already flows through consistency and
    # confidence, so there is no independent prior belief here.
    "opportunities": 0.0,
    "opportunities_missing": 0.0,
    # Consistency (hits / eligible days) is time_of_day's own headline number
    # and the clearest per-miner signal of "this really is a habit".
    "consistency": 1.2,
    "consistency_missing": 0.0,
    "support": 0.4,
    "support_missing": 0.0,
    # confidence means a different conditional probability per miner, but in
    # every case a higher one means "more often right when it could fire".
    "confidence": 1.0,
    "confidence_missing": 0.0,
    "lift": 0.3,
    "lift_missing": 0.0,
    # A wider spread of times the habit actually happened makes for a noisier
    # trigger time and a less trustworthy automation.
    "spread_minutes": -0.3,
    "spread_minutes_missing": 0.0,
    # --- backtest: the one number this project already treats as comparable
    # across miners (miners/base.py:172), so these carry the heaviest weights.
    "precision": 2.0,
    # No backtest at all (an audit finding with nothing to replay) is treated
    # as worse than an average backtest result, not neutral - the whole point
    # of this project is not suggesting the unmeasured.
    "precision_missing": -0.5,
    "recall": 0.8,
    "recall_missing": 0.0,
    "false_fires_per_week": -0.6,
    "false_fires_per_week_missing": 0.0,
    "true_fires": 0.25,
    "true_fires_missing": 0.0,
    "holdout_days": 0.2,
    "holdout_days_missing": 0.0,
    # --- structure -------------------------------------------------------
    "n_triggers": 0.0,
    "n_triggers_missing": 0.0,
    "n_conditions": 0.1,  # an explicit condition suggests a deliberate rule
    "n_conditions_missing": 0.0,
    "n_actions": 0.0,
    "n_actions_missing": 0.0,
    # Touching more entities is more surface area for something to be wrong.
    "n_entities": -0.1,
    "n_entities_missing": 0.0,
    "conflict_count": -0.9,
    "conflict_count_missing": 0.0,
    "worst_conflict_severity": -0.4,
    "worst_conflict_severity_missing": 0.0,
    # --- context -----------------------------------------------------------
    # Deliberately neutral.  How many times a suggestion has already been
    # *shown* says nothing on its own about whether it deserves to be shown
    # again - llm/preferences.py explicitly declines to treat "surfaced but
    # never acted on" as an implicit rejection, and giving this a negative
    # prior would smuggle exactly that assumption back in through the model
    # instead of through data.  Left at 0 for the personal model to learn
    # from, if a user's own history ever actually supports it.
    "seen_count": 0.0,
    "seen_count_missing": 0.0,
    # --- binary flags ------------------------------------------------------
    # config.py:150 - the whole reason a holdout is carved out is that a rule
    # graded on the data it was mined from is being re-described, not tested.
    "holdout_validated": 1.0,
    "has_backtest": 0.3,
    # config.py:67 - risky domains (locks, covers, valves, alarms, water
    # heaters, sirens, mowers, vacuums) get things wrong more expensively than
    # a light left on; the prior is cautious about them independent of
    # anything a personal model learns.
    "risky_domain": -1.0,
}
# No established belief that any one domain is more acceptable than another
# beyond the risky-domain flag above, or that any one miner is inherently
# better than the others - "this user always dismisses the association
# miner" is a personal fact, not a general one, and is exactly the kind of
# thing the personal fit - not this hand-set prior - should discover.
PRIOR_WEIGHTS.update({f"domain_{d}": 0.0 for d in DOMAIN_VOCAB})
PRIOR_WEIGHTS["domain_other"] = 0.0
PRIOR_WEIGHTS.update({f"miner_{m}": 0.0 for m in MINER_VOCAB})
PRIOR_WEIGHTS["miner_other"] = 0.0

# A drifted feature name here (a rename on one side, not the other) would
# silently apply the wrong weight to the wrong column - fail at import time
# instead, where it is loud and always caught by the test suite importing
# this module.
assert set(PRIOR_WEIGHTS) == set(_FEATURE_NAMES), (
    "PRIOR_WEIGHTS and feature_names() have drifted apart"
)

PRIOR_VECTOR: np.ndarray = np.array([PRIOR_WEIGHTS[name] for name in _FEATURE_NAMES])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def shrinkage_lambda(n_labels: int, k: int) -> float:
    """How much weight the personal fit gets, out of prior-and-personal.

    ``n_labels=0`` gives 0 (pure prior); ``n_labels == k`` gives exactly 0.5;
    ``n_labels >> k`` approaches 1 (mostly personal).  ``k`` is the number of
    labels at which the two are trusted equally - the add-on option that
    controls it is documented as such.
    """
    if n_labels <= 0:
        return 0.0
    return n_labels / (n_labels + max(k, 1))


class _RidgeLogisticRegression:
    """L2-regularised logistic regression, fit by Newton's method (IRLS).

    Deterministic on purpose: weights always start at zero, the update rule
    involves no randomness, and the number of iterations is fixed - the same
    ``(X, y)`` always converges to the same coefficients.  That matters twice
    over here: this runs nightly and feeds an ordering people see, and the
    test suite pins an exact result.

    Pure numpy, per this project's constraint that scikit-learn is
    deliberately not a dependency (musllinux wheels for aarch64 do not exist
    for it).
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8) -> None:
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.intercept_: float = 0.0
        self.coef_: np.ndarray = np.zeros(0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> _RidgeLogisticRegression:
        n_samples, n_features = x.shape
        design = np.hstack([np.ones((n_samples, 1)), x])
        w = np.zeros(n_features + 1)
        reg = np.eye(n_features + 1) * self.l2
        reg[0, 0] = 0.0  # never shrink the intercept towards 0
        for _ in range(self.max_iter):
            z = design @ w
            p = _sigmoid(z)
            grad = design.T @ (p - y) + reg @ w
            weight = np.clip(p * (1.0 - p), 1e-6, None)
            hessian = design.T @ (design * weight[:, None]) + reg
            try:
                step = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
            # A run of near-perfectly-separated labels (every label in a fold
            # the same class, easy with a dozen samples) drives Newton's
            # method towards an infinite intercept.  Clipping keeps the fit
            # finite and deterministic without meaningfully changing any
            # well-conditioned result, where weights are of order 1.
            w_new = np.clip(w - step, -25.0, 25.0)
            if np.max(np.abs(w_new - w)) < self.tol:
                w = w_new
                break
            w = w_new
        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.intercept_ + x @ self.coef_)


def _blend(prior: np.ndarray, personal: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - lam) * prior + lam * personal


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


@dataclass(frozen=True)
class LabelExample:
    """One past decision, reconstructed as a training example.

    Built from a *snapshot* taken at the moment of the decision
    (``Store.ranking_labels``), never from a suggestion's live row - a
    suggestion that stayed accepted keeps being re-mined and re-backtested
    every night, and its live payload keeps being overwritten with numbers
    the user never saw when they actually decided.
    """

    candidate: Candidate
    label: int  # 1 == accepted, 0 == dismissed
    seen_count: int | None = None


@dataclass
class RankingModel:
    """A fitted (or prior-only) acceptance model, ready to score candidates."""

    feature_schema_version: int
    weights: dict[str, float]
    bias: float
    n_labels: int
    prior_k: int
    trained_ts: float = field(default_factory=time.time)
    fallback_to_prior: bool = True
    fallback_reason: str | None = None

    def probability(self, candidate: Candidate, seen_count: int | None = None) -> float:
        """The estimated probability this user accepts *candidate*.

        For ranking and display only - see the module docstring.  This never
        reads whether the candidate passed its backtest or has a blocking
        conflict as anything other than an input feature; it has no way to
        promote a candidate into ``passed`` or push one out of it, because it
        is never called with the ``rejected`` list at all.
        """
        x = extract_features(candidate, seen_count)
        w = np.array([self.weights[name] for name in _FEATURE_NAMES])
        return float(_sigmoid(np.array([self.bias + float(x @ w)]))[0])

    def confidence_note(self) -> str:
        """One honest sentence about how much this rests on the prior vs you.

        Never a bare number with no context: two decisions and two hundred
        must not read the same way, and a fallback must say why it fell back
        rather than presenting the prior as if it were personalised.
        """
        if self.n_labels == 0:
            return "based entirely on general patterns - you have not accepted or dismissed anything yet"
        if self.fallback_to_prior:
            plural = "s" if self.n_labels != 1 else ""
            return (
                f"based on general patterns - your {self.n_labels} decision{plural} did not "
                "yet make the ordering more accurate, so they were not used"
            )
        lam = shrinkage_lambda(self.n_labels, self.prior_k)
        if lam < 0.25:
            return f"mostly general patterns, lightly adjusted by {self.n_labels} of your decisions"
        if lam < 0.6:
            return f"a mix of general patterns and {self.n_labels} of your own decisions"
        return f"based mainly on your own {self.n_labels} accept/dismiss decisions"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": self.feature_schema_version,
            "weights": dict(self.weights),
            "bias": self.bias,
            "n_labels": self.n_labels,
            "prior_k": self.prior_k,
            "trained_ts": self.trained_ts,
            "fallback_to_prior": self.fallback_to_prior,
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> RankingModel | None:
        """Rebuild a model from a persisted row, discarding it if stale.

        A weight vector fit under an earlier :data:`FEATURE_SCHEMA_VERSION` is
        not stale data to patch around, it is meaningless: the same index
        could have named a different feature.  Any mismatch here must be
        treated exactly like there being no model at all, never "close
        enough".
        """
        if not row:
            return None
        if int(row.get("feature_schema_version", -1)) != FEATURE_SCHEMA_VERSION:
            return None
        weights = row.get("weights")
        if not isinstance(weights, dict) or set(weights) != set(_FEATURE_NAMES):
            return None
        return cls(
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            weights={k: float(v) for k, v in weights.items()},
            bias=float(row.get("bias", PRIOR_BIAS)),
            n_labels=int(row.get("n_labels", 0)),
            prior_k=int(row.get("prior_k", DEFAULT_PRIOR_K)),
            trained_ts=float(row.get("trained_ts", 0.0)),
            fallback_to_prior=bool(row.get("fallback_to_prior", True)),
            fallback_reason=row.get("fallback_reason"),
        )


#: Below this many labels there is nothing to meaningfully fit or
#: cross-validate at all - a "model" fit on one or two points is just those
#: points with extra steps.  The prior is used outright and the run is
#: reported as a cold start, not as a personal model that lost a comparison it
#: was never really in.
MIN_LABELS_TO_FIT = 4

#: Default label count at which the personal fit and the prior are trusted
#: equally.  Overridable via the ``ranking_prior_k`` add-on option.
DEFAULT_PRIOR_K = 20


def _fit_blended(
    x: np.ndarray, y: np.ndarray, k: int, l2: float
) -> tuple[np.ndarray, float]:
    """Fit the personal model on (x, y) and shrink it towards the prior."""
    model = _RidgeLogisticRegression(l2=l2).fit(x, y)
    personal = np.concatenate(([model.intercept_], model.coef_))
    prior_full = np.concatenate(([PRIOR_BIAS], PRIOR_VECTOR))
    lam = shrinkage_lambda(len(y), k)
    blended = _blend(prior_full, personal, lam)
    return blended[1:], float(blended[0])


def _prior_only_model(n_labels: int, k: int, reason: str) -> RankingModel:
    return RankingModel(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        weights=dict(PRIOR_WEIGHTS),
        bias=PRIOR_BIAS,
        n_labels=n_labels,
        prior_k=k,
        fallback_to_prior=True,
        fallback_reason=reason,
    )


def train_from_labels(
    examples: Sequence[LabelExample], k: int = DEFAULT_PRIOR_K, l2: float = 1.0
) -> RankingModel:
    """Fit a personal ranking model, or fall back to the prior.

    Fitting happens in two stages:

    1. Shrink a personal logistic-regression fit towards :data:`PRIOR_WEIGHTS`
       by :func:`shrinkage_lambda`, so a handful of labels nudges the ordering
       gently and a large history dominates it.
    2. Leave-one-out cross-validate that *shrunk* model's held-out log-loss
       against the prior's alone.  If the personal fit does not beat the
       prior, the prior is used outright and this is recorded - a noisy
       handful of labels must never be allowed to produce a worse ordering
       than the add-on shipped with.
    """
    n = len(examples)
    if n < MIN_LABELS_TO_FIT:
        if n == 0:
            reason = "cold start - no decisions recorded yet"
        else:
            plural = "s" if n != 1 else ""
            reason = f"only {n} labelled decision{plural} so far - too few to fit a personal model"
        return _prior_only_model(n, k, reason)

    x = np.vstack([extract_features(e.candidate, e.seen_count) for e in examples])
    y = np.array([float(e.label) for e in examples])

    personal_losses = np.empty(n)
    prior_losses = np.empty(n)
    prior_full = np.concatenate(([PRIOR_BIAS], PRIOR_VECTOR))
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w_i, b_i = _fit_blended(x[mask], y[mask], k, l2)
        p_personal = _sigmoid(np.array([b_i + x[i] @ w_i]))
        p_prior = _sigmoid(np.array([prior_full[0] + x[i] @ prior_full[1:]]))
        personal_losses[i] = _log_loss(y[i : i + 1], p_personal)
        prior_losses[i] = _log_loss(y[i : i + 1], p_prior)

    personal_cv_loss = float(np.mean(personal_losses))
    prior_cv_loss = float(np.mean(prior_losses))

    if personal_cv_loss >= prior_cv_loss:
        return _prior_only_model(
            n,
            k,
            f"the personal model's held-out log-loss ({personal_cv_loss:.3f}) did not beat "
            f"the prior's ({prior_cv_loss:.3f}) over {n} labels",
        )

    weights_vector, bias = _fit_blended(x, y, k, l2)
    return RankingModel(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        weights={name: float(weights_vector[i]) for i, name in enumerate(_FEATURE_NAMES)},
        bias=bias,
        n_labels=n,
        prior_k=k,
        fallback_to_prior=False,
        fallback_reason=None,
    )


def rank_candidates(
    candidates: Sequence[Candidate], model: RankingModel, seen_counts: dict[str, int] | None = None
) -> None:
    """Attach an acceptance probability to every candidate, in place.

    Deliberately the only thing this function does.  It never removes a
    candidate, never reorders the list it is given (the caller decides how to
    use the attached number), and never reads or writes ``passed``/
    ``rejected`` - those are decided entirely by the backtest gate before this
    is ever called.  ``candidate.extra["ranking"]`` is additive, the same
    pattern ``llm/triage.py`` and ``llm/classify.py`` use for their own
    advisory annotations.
    """
    seen_counts = seen_counts or {}
    for candidate in candidates:
        probability = model.probability(candidate, seen_counts.get(candidate.id))
        candidate.extra["ranking"] = {
            "probability": round(probability, 4),
            "n_labels": model.n_labels,
            "fallback_to_prior": model.fallback_to_prior,
            "confidence_note": model.confidence_note(),
        }


def labels_from_rows(rows: Sequence[dict[str, Any]]) -> list[LabelExample]:
    """Turn :meth:`Store.ranking_labels` rows into training examples.

    ``rows`` are already filtered to accepted/dismissed decisions with a
    decision-time snapshot recorded (see that method's docstring on why the
    live suggestion row is not used) - anything without one is left out by
    the store, not guessed at here.
    """
    # Deferred: runner.py imports pipeline.py, which is what calls into this
    # module, so importing runner at module scope would be a cycle. Same
    # pattern as pipeline._log_shadow_fires.
    from ..runner import candidate_from_payload

    examples: list[LabelExample] = []
    for row in rows:
        payload = row.get("payload")
        if not payload:
            continue
        try:
            candidate = candidate_from_payload(payload)
        except (KeyError, TypeError, ValueError):
            continue
        examples.append(
            LabelExample(
                candidate=candidate,
                label=1 if row.get("status") == "accepted" else 0,
                seen_count=row.get("seen_count_at_decision"),
            )
        )
    return examples
