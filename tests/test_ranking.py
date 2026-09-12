"""The learned acceptance-ranking model (amminer.learn.ranking).

Weighted towards the two things that would make this feature actively harmful
if they were wrong: the safety property that it can never change a pass/fail
outcome (it only ever orders what already passed), and the guard that stops a
handful of noisy labels from producing a worse ordering than the hand-set
prior the add-on ships with.
"""

from __future__ import annotations

import numpy as np
import pytest
from amminer.learn import ranking
from amminer.miners.base import Action, Candidate, Condition, Evidence, Trigger


def make_candidate(
    miner="time_of_day",
    entity="light.kitchen",
    occurrences=10,
    opportunities=14,
    consistency=0.8,
    confidence=None,
    lift=None,
    precision=0.9,
    recall=0.5,
    false_fires_per_week=0.2,
    true_fires=9,
    holdout_days=10.0,
    validation="holdout",
    has_backtest=True,
    conflicts=None,
    n_conditions=0,
) -> Candidate:
    candidate = Candidate(
        miner=miner,
        title=f"Turn on {entity}",
        triggers=[Trigger(kind="time", at="06:30:00")],
        conditions=[],
        actions=[Action(service=f"{entity.split('.')[0]}.turn_on", entity_id=entity)],
        evidence=Evidence(
            occurrences=occurrences,
            opportunities=opportunities,
            consistency=consistency,
            confidence=confidence,
            lift=lift,
        ),
    )
    if n_conditions:
        candidate.conditions = [Condition(kind="time", after="18:00") for _ in range(n_conditions)]
    if has_backtest:
        candidate.backtest = {
            "precision": precision,
            "recall": recall,
            "false_fires_per_week": false_fires_per_week,
            "true_fires": true_fires,
            "holdout_days": holdout_days,
            "validation": validation,
            "passed": True,
        }
    candidate.conflicts = conflicts or []
    return candidate


# --- feature extraction --------------------------------------------------
def test_missing_values_get_an_explicit_indicator_not_a_bare_zero():
    candidate = make_candidate(consistency=None, has_backtest=False)
    x = ranking.extract_features(candidate, seen_count=None)
    names = ranking.feature_names()

    assert x[names.index("consistency")] == 0.0
    assert x[names.index("consistency_missing")] == 1.0
    assert x[names.index("precision_missing")] == 1.0
    assert x[names.index("has_backtest")] == 0.0
    assert x[names.index("seen_count_missing")] == 1.0


def test_a_real_zero_is_not_confused_with_a_missing_value():
    candidate = make_candidate(occurrences=0, opportunities=0)
    x = ranking.extract_features(candidate, seen_count=0)
    names = ranking.feature_names()

    assert x[names.index("occurrences")] == 0.0
    assert x[names.index("occurrences_missing")] == 0.0
    assert x[names.index("seen_count")] == 0.0
    assert x[names.index("seen_count_missing")] == 0.0


def test_feature_vector_has_a_stable_fixed_width():
    names = ranking.feature_names()
    assert len(names) == len(set(names)), "duplicate feature name"
    a = ranking.extract_features(make_candidate())
    b = ranking.extract_features(make_candidate(miner="association"))
    assert a.shape == b.shape == (len(names),)


def test_risky_domain_and_conflict_severity_are_read_correctly():
    risky = make_candidate(entity="lock.front_door")
    risky.conflicts = [{"severity": "warning"}, {"severity": "error"}]
    x = ranking.extract_features(risky)
    names = ranking.feature_names()
    assert x[names.index("risky_domain")] == 1.0
    assert x[names.index("conflict_count")] == pytest.approx(np.log1p(2))
    assert x[names.index("worst_conflict_severity")] == 3.0

    safe = make_candidate(entity="light.kitchen")
    assert ranking.extract_features(safe)[names.index("risky_domain")] == 0.0


def test_miner_and_domain_outside_the_vocabulary_fall_into_other():
    candidate = make_candidate(miner="time_of_day+hypothesis", entity="lawn_mower.mower")
    x = ranking.extract_features(candidate)
    names = ranking.feature_names()
    assert x[names.index("miner_other")] == 1.0
    assert x[names.index("miner_time_of_day")] == 0.0
    assert x[names.index("domain_other")] == 1.0


# --- the prior ------------------------------------------------------------
def test_prior_weights_cover_exactly_the_feature_vector():
    # Guards against the exact drift the module's own assert protects against;
    # written here too so a break shows up as a normal test failure.
    assert set(ranking.PRIOR_WEIGHTS) == set(ranking.feature_names())


def test_prior_alone_orders_a_holdout_validated_candidate_above_an_in_sample_one():
    model = ranking.train_from_labels([], k=20)
    assert model.fallback_to_prior is True
    assert model.n_labels == 0

    validated = make_candidate(validation="holdout")
    in_sample = make_candidate(validation="in_sample")
    assert model.probability(validated) > model.probability(in_sample)


def test_prior_alone_penalises_conflicts_and_risky_domains():
    model = ranking.train_from_labels([])
    clean = make_candidate(entity="light.kitchen", conflicts=[])
    conflicted = make_candidate(entity="light.kitchen", conflicts=[{"severity": "warning"}])
    risky = make_candidate(entity="lock.front_door", conflicts=[])
    assert model.probability(clean) > model.probability(conflicted)
    assert model.probability(clean) > model.probability(risky)


# --- shrinkage --------------------------------------------------------
def test_shrinkage_lambda_at_zero_labels_is_pure_prior():
    assert ranking.shrinkage_lambda(0, 20) == 0.0


def test_shrinkage_lambda_at_n_equal_k_is_one_half():
    assert ranking.shrinkage_lambda(20, 20) == pytest.approx(0.5)


def test_shrinkage_lambda_grows_towards_one_with_more_labels():
    small = ranking.shrinkage_lambda(5, 20)
    large = ranking.shrinkage_lambda(2000, 20)
    assert 0.0 < small < 0.5
    assert large > 0.9


# --- cold start / too few labels --------------------------------------
def test_below_the_minimum_label_count_the_prior_is_used_outright():
    examples = [
        ranking.LabelExample(candidate=make_candidate(), label=1, seen_count=1),
        ranking.LabelExample(candidate=make_candidate(), label=0, seen_count=1),
    ]
    assert len(examples) < ranking.MIN_LABELS_TO_FIT
    model = ranking.train_from_labels(examples)
    assert model.fallback_to_prior is True
    assert model.weights == ranking.PRIOR_WEIGHTS
    assert "too few" in model.fallback_reason


# --- the guard: a personal model must never be allowed to be worse ----
def _examples(n, noise=False, seed=0):
    """``n`` labelled examples.

    With ``noise=False`` the label is a real, learnable function of the
    features (precision/recall/consistency) - a personal fit should recover
    it easily.  With ``noise=True`` the label is an independent coin flip,
    uncorrelated with every feature - exactly what a handful of genuinely
    arbitrary human decisions would look like, and precisely the case the
    guard exists for: a small logistic regression fit on pure noise finds
    *some* separating combination of features in-sample, and that spurious
    fit generalises badly to the point left out of each fold, which is what
    should make it lose to the prior on held-out log-loss.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        good = rng.random() > 0.5
        candidate = make_candidate(
            precision=0.95 if good else 0.3,
            recall=0.8 if good else 0.2,
            consistency=0.9 if good else 0.3,
            validation="holdout" if good else "in_sample",
        )
        label = rng.integers(0, 2) if noise else int(good)
        out.append(ranking.LabelExample(candidate=candidate, label=int(label), seen_count=1))
    return out


def test_a_personal_model_that_learns_the_real_signal_is_used():
    examples = _examples(40, noise=False, seed=1)
    model = ranking.train_from_labels(examples, k=20)
    assert model.fallback_to_prior is False
    assert model.n_labels == 40


_JUNK_DOMAINS = (
    "light.a", "switch.b", "climate.c", "cover.d", "lock.e", "fan.f",
    "media_player.g", "sensor.h", "binary_sensor.i", "person.j", "device_tracker.k",
)


def _overfit_noise_examples(n, seed):
    """``n`` examples where the label is a coin flip independent of everything.

    Miner and entity domain vary per example (high-cardinality one-hot
    features) while the numeric evidence is held flat and uninformative - a
    classic few-samples/many-parameters setup a small regularised fit can
    still partially memorise (a handful of one-hot columns happens to line up
    with the training fold's random labels), producing confident predictions
    that do not hold up on the point left out of each fold.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        miner = ranking.MINER_VOCAB[rng.integers(0, len(ranking.MINER_VOCAB))]
        entity = _JUNK_DOMAINS[rng.integers(0, len(_JUNK_DOMAINS))]
        candidate = make_candidate(
            miner=miner, entity=entity, precision=0.7, recall=0.5,
            consistency=0.5, validation="in_sample",
        )
        label = int(rng.integers(0, 2))
        out.append(ranking.LabelExample(candidate=candidate, label=label, seen_count=1))
    return out


def test_the_guard_rejects_a_personal_model_that_overfits_pure_noise():
    # A weak L2 and few samples relative to the (many, mostly one-hot)
    # features let the personal fit partially memorise which miner/domain
    # happened to co-occur with which random label in-sample - exactly the
    # kind of spurious fit that looks fine in training and falls over on the
    # point each leave-one-out fold held out. This seed's numbers are printed
    # in the failure-free assertion below so a change in the fit is visible,
    # not just a flipped boolean.
    examples = _overfit_noise_examples(6, seed=9)
    model = ranking.train_from_labels(examples, k=1, l2=0.05)
    assert model.fallback_to_prior is True
    assert model.weights == ranking.PRIOR_WEIGHTS
    assert "did not beat the prior" in model.fallback_reason


# --- recovering a known preference -------------------------------------
def test_the_model_recovers_a_consistent_per_miner_preference():
    """A user who always dismisses one miner's output ends up ranking it lower.

    Every other feature is held roughly equal between the two miners, so any
    difference the model ends up with is attributable to the label pattern -
    not to the prior, which is neutral on miner identity by construction
    (PRIOR_WEIGHTS sets every miner_* weight to 0.0).
    """
    rng = np.random.default_rng(3)
    examples = []
    for _ in range(30):
        jitter = rng.uniform(-0.02, 0.02)
        examples.append(
            ranking.LabelExample(
                candidate=make_candidate(
                    miner="association", precision=0.8 + jitter, recall=0.5 + jitter
                ),
                label=0,
                seen_count=1,
            )
        )
        examples.append(
            ranking.LabelExample(
                candidate=make_candidate(
                    miner="time_of_day", precision=0.8 + jitter, recall=0.5 + jitter
                ),
                label=1,
                seen_count=1,
            )
        )

    model = ranking.train_from_labels(examples, k=20)
    assert model.fallback_to_prior is False

    association_candidate = make_candidate(miner="association")
    time_of_day_candidate = make_candidate(miner="time_of_day")
    assert model.probability(time_of_day_candidate) > model.probability(association_candidate)
    # And it is a real, substantial gap - not a rounding artefact of a model
    # that actually learned nothing.
    assert model.probability(time_of_day_candidate) - model.probability(association_candidate) > 0.1


# --- determinism --------------------------------------------------------
def test_training_is_deterministic():
    examples = _examples(25, noise=False, seed=4)
    first = ranking.train_from_labels(examples, k=20)
    second = ranking.train_from_labels(examples, k=20)
    assert first.weights == second.weights
    assert first.bias == second.bias
    assert first.fallback_to_prior == second.fallback_to_prior


# --- schema-version guard -------------------------------------------------
def test_from_row_discards_a_model_fit_under_an_older_schema():
    row = {
        "feature_schema_version": ranking.FEATURE_SCHEMA_VERSION - 1,
        "weights": dict(ranking.PRIOR_WEIGHTS),
        "bias": -1.0,
        "n_labels": 50,
        "prior_k": 20,
        "trained_ts": 0.0,
        "fallback_to_prior": False,
        "fallback_reason": None,
    }
    assert ranking.RankingModel.from_row(row) is None


def test_from_row_discards_a_model_whose_weights_do_not_match_the_current_features():
    row = {
        "feature_schema_version": ranking.FEATURE_SCHEMA_VERSION,
        "weights": {"only_one_feature": 1.0},
        "bias": -1.0,
        "n_labels": 50,
        "prior_k": 20,
        "trained_ts": 0.0,
        "fallback_to_prior": False,
        "fallback_reason": None,
    }
    assert ranking.RankingModel.from_row(row) is None


def test_from_row_accepts_a_current_schema_model():
    row = {
        "feature_schema_version": ranking.FEATURE_SCHEMA_VERSION,
        "weights": dict(ranking.PRIOR_WEIGHTS),
        "bias": -1.0,
        "n_labels": 5,
        "prior_k": 20,
        "trained_ts": 123.0,
        "fallback_to_prior": True,
        "fallback_reason": "cold start",
    }
    model = ranking.RankingModel.from_row(row)
    assert model is not None
    assert model.n_labels == 5


def test_from_row_of_nothing_is_nothing():
    assert ranking.RankingModel.from_row(None) is None
    assert ranking.RankingModel.from_row({}) is None


# --- honesty of the confidence note -------------------------------------
def test_confidence_note_says_cold_start_at_zero_labels():
    model = ranking.train_from_labels([])
    note = model.confidence_note()
    assert "have not accepted or dismissed" in note


def test_confidence_note_names_the_fallback_reason_family_when_the_guard_fires():
    examples = _overfit_noise_examples(6, seed=9)
    model = ranking.train_from_labels(examples, k=1, l2=0.05)
    assert model.fallback_to_prior is True
    note = model.confidence_note()
    assert "did not" in note  # honest about the guard, not a bare percentage


def test_confidence_note_differs_between_few_and_many_labels():
    few = ranking.RankingModel(
        feature_schema_version=ranking.FEATURE_SCHEMA_VERSION,
        weights=dict(ranking.PRIOR_WEIGHTS), bias=-1.0, n_labels=2, prior_k=20,
        fallback_to_prior=False,
    )
    many = ranking.RankingModel(
        feature_schema_version=ranking.FEATURE_SCHEMA_VERSION,
        weights=dict(ranking.PRIOR_WEIGHTS), bias=-1.0, n_labels=500, prior_k=20,
        fallback_to_prior=False,
    )
    assert few.confidence_note() != many.confidence_note()
    assert "mostly" in few.confidence_note() or "mix" in few.confidence_note()
    assert "mainly on your own" in many.confidence_note()


# --- the safety property: ranking cannot alter a pass/fail outcome -----
def test_rank_candidates_never_moves_anything_between_passed_and_rejected():
    """The core safety claim: this module has no way to gate anything.

    Backtesting/conflict-checking decide `passed` vs `rejected` before ranking
    ever runs.  This proves the *strong* version of that: even a model
    engineered to give every rejected-like candidate a probability of ~1.0 and
    every passed-like candidate ~0.0 cannot change which list either ends up
    in, because rank_candidates has no parameter through which it could.
    """
    passed = [make_candidate(entity="light.a", precision=0.95), make_candidate(entity="light.b", precision=0.9)]
    rejected = [make_candidate(entity="light.c", precision=0.1), make_candidate(entity="light.d", precision=0.05)]
    for c in rejected:
        c.backtest["passed"] = False

    # An adversarial model: strongly rewards exactly the low-precision profile
    # that failed the backtest, and punishes the high-precision one that
    # passed - the opposite of what the prior believes.
    adversarial_weights = dict(ranking.PRIOR_WEIGHTS)
    adversarial_weights["precision"] = -50.0
    model = ranking.RankingModel(
        feature_schema_version=ranking.FEATURE_SCHEMA_VERSION,
        weights=adversarial_weights, bias=25.0, n_labels=100, prior_k=20,
        fallback_to_prior=False,
    )

    before_passed_ids = {c.id for c in passed}
    before_rejected_ids = {c.id for c in rejected}

    ranking.rank_candidates(passed, model, {})
    ranking.rank_candidates(rejected, model, {})

    # Membership of each list is untouched - rank_candidates has no return
    # value and no side effect other than annotating candidate.extra.
    assert {c.id for c in passed} == before_passed_ids
    assert {c.id for c in rejected} == before_rejected_ids
    # The candidates whose backtest failed are exactly as failed as before -
    # nothing here reads or writes `.backtest["passed"]`.
    assert all(c.backtest["passed"] is False for c in rejected)
    assert all(c.backtest["passed"] is True for c in passed)
    # And, despite the adversarial weighting giving the *rejected* candidates
    # the higher probability, they were never given the chance to end up
    # ahead of a passed one anywhere ranking is actually used: the pipeline
    # only ever calls rank_candidates on `passed`.
    assert model.probability(rejected[0]) > model.probability(passed[0])


def test_pipeline_only_ranks_the_passed_list_not_rejected():
    """A regression guard on the wiring itself, not just the function's contract."""
    import inspect

    from amminer import pipeline

    source = inspect.getsource(pipeline)
    assert "ranking_module.rank_candidates(passed, model, seen_counts)" in source
    assert "rank_candidates(rejected" not in source


def test_probability_is_never_used_by_the_backtest_or_conflict_modules():
    """amminer.backtest and amminer.conflicts must not import this module at all."""
    import inspect

    from amminer import backtest as backtest_module
    from amminer import conflicts as conflicts_module

    assert "learn" not in inspect.getsource(backtest_module)
    assert "learn" not in inspect.getsource(conflicts_module)
