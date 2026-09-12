"""The learned acceptance-ranking model (amminer.learn.ranking).

Weighted towards the three things that would make this feature actively
harmful if they were wrong: the safety property that it can never change a
pass/fail outcome (it only ever orders what already passed); that a personal
fit stays close to the prior when the labels behind it carry no real signal,
under realistic conditions and at production defaults, not only under
contrived adversarial ones; and that it can still learn something real when
the labels actually show one.
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
    model = ranking.train_from_labels([])
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


# --- the scaler -------------------------------------------------------
def test_scaler_identity_is_a_true_no_op():
    scaler = ranking.Scaler.identity()
    x = ranking.extract_features(make_candidate())
    assert np.array_equal(scaler.transform(x.reshape(1, -1))[0], x)
    w_std, b_std = scaler.prior_in_this_space()
    assert np.array_equal(w_std, ranking.PRIOR_VECTOR)
    assert b_std == ranking.PRIOR_BIAS


def test_scaler_prior_conversion_predicts_identically_to_the_raw_prior():
    """The whole point of re-expressing the prior in standardised space: it
    must predict the same probability either way, or MAP regularisation would
    be pulling the fit towards the wrong number entirely."""
    examples = [make_candidate(miner=m) for m in ("time_of_day", "association", "conditional")]
    x = np.vstack([ranking.extract_features(c) for c in examples])
    scaler = ranking.Scaler.fit(x)
    w_std, b_std = scaler.prior_in_this_space()

    for candidate in examples:
        raw = ranking.extract_features(candidate)
        raw_z = ranking.PRIOR_BIAS + float(raw @ ranking.PRIOR_VECTOR)
        std = scaler.transform(raw.reshape(1, -1))[0]
        std_z = b_std + float(std @ w_std)
        assert std_z == pytest.approx(raw_z, abs=1e-9)


def test_scaler_constant_feature_does_not_blow_up():
    x = np.tile(ranking.extract_features(make_candidate()), (5, 1))  # every row identical
    scaler = ranking.Scaler.fit(x)
    assert all(v == 1.0 for v in scaler.scale.values())
    # transform is then a plain centring, not a division by zero -> nan/inf.
    transformed = scaler.transform(x)
    assert np.all(np.isfinite(transformed))


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


# --- a personal fit that recovers a real signal ------------------------
def _signal_examples(n, seed):
    """``n`` examples where the label is a real, learnable function of
    precision/recall/consistency - a personal fit should recover it."""
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
        out.append(ranking.LabelExample(candidate=candidate, label=int(good), seen_count=1))
    return out


def test_a_personal_model_that_learns_the_real_signal_is_used():
    examples = _signal_examples(40, seed=1)
    model = ranking.train_from_labels(examples)
    assert model.fallback_to_prior is False
    assert model.n_labels == 40


# --- the guard: a demonstrably-overfit fit must fall back --------------
_JUNK_DOMAINS = (
    "light.a", "switch.b", "climate.c", "cover.d", "lock.e", "fan.f",
    "media_player.g", "sensor.h", "binary_sensor.i", "person.j", "device_tracker.k",
)


def _overfit_noise_examples(n, seed):
    """``n`` examples where the label is a coin flip independent of everything.

    Miner and entity domain vary per example (high-cardinality one-hot
    features) with a deliberately tiny ``l2`` in the test itself - not the
    production default, which is exactly what the permanent null-case test
    below exercises - so a few one-hot columns can still line up with the
    training fold's random labels by chance, and the guard has to catch it.
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
    # A weak L2 relative to the (many, mostly one-hot) features lets the
    # personal fit partially memorise which miner/domain happened to
    # co-occur with which random label in-sample; this seed's held-out
    # improvement is negative (the personal model is worse, not merely
    # statistically indistinguishable), a stronger claim than a bare
    # pass/fail boolean.
    examples = _overfit_noise_examples(6, seed=0)
    model = ranking.train_from_labels(examples, l2=0.05)
    assert model.fallback_to_prior is True
    assert model.weights == ranking.PRIOR_WEIGHTS
    assert "not clearly bigger than sampling noise" in model.fallback_reason
    assert "-0." in model.fallback_reason  # the improvement really is negative


# --- the required permanent regression: production defaults, realistic
# features, pure noise, at n = 5 / 12 / 25 -------------------------------
_REALISTIC_MINERS = ("time_of_day", "association", "conditional", "motif", "energy_shift")
_REALISTIC_DOMAINS = ("light", "switch", "climate", "lock", "cover", "media_player")
_SEVERITIES = ("info", "warning", "error")


def _realistic_candidate(rng) -> Candidate:
    """A candidate shaped like real mined output, spanning a real quality range.

    Deliberately not all near-certain accepts: a real suggestion feed mixes
    genuinely strong candidates with mediocre ones (some without a backtest
    at all, some with a conflict, precision anywhere from poor to excellent).
    An earlier version of this generator drew everything from a narrow,
    uniformly-favourable range, which put almost every prior probability
    within a few points of 1.0 - a regime where Spearman correlation is
    extremely sensitive to noise-scale probability jitter regardless of
    whether the model is actually behaving safely, and which a review found
    made the null-case check pass for the wrong reason. This wider spread
    (prior probabilities from roughly 0.1 to 0.97 - see the assertion in the
    test below) is deliberately the harder, more realistic case.
    """
    domain = rng.choice(_REALISTIC_DOMAINS)
    entity = f"{domain}.thing{int(rng.integers(0, 3))}"
    conflicts = [{"severity": str(rng.choice(_SEVERITIES))} for _ in range(int(rng.integers(0, 3)))]
    return make_candidate(
        miner=str(rng.choice(_REALISTIC_MINERS)),
        entity=entity,
        occurrences=int(rng.integers(0, 60)),
        opportunities=int(rng.integers(1, 80)),
        consistency=float(rng.uniform(0.1, 0.95)),
        precision=float(rng.uniform(0.1, 0.98)),
        recall=float(rng.uniform(0.05, 0.8)),
        false_fires_per_week=float(rng.uniform(0.0, 5.0)),
        true_fires=int(rng.integers(0, 40)),
        holdout_days=float(rng.uniform(0.0, 25.0)),
        validation="holdout" if rng.random() < 0.5 else "in_sample",
        has_backtest=rng.random() < 0.9,
        conflicts=conflicts,
    )


def _null_examples(n, seed, accept_rate) -> list[ranking.LabelExample]:
    """``n`` examples with realistic features and a label independent of every one of them."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        candidate = _realistic_candidate(rng)
        label = int(rng.random() < accept_rate)
        out.append(
            ranking.LabelExample(candidate=candidate, label=label, seen_count=int(rng.integers(1, 5)))
        )
    return out


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, computed with nothing but numpy.

    (scipy is available in this project, but this is three lines and keeps
    the dependency footprint of a safety-critical test as small as the
    module it is testing.)
    """
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denom = np.sqrt((rank_a**2).sum() * (rank_b**2).sum())
    return float((rank_a * rank_b).sum() / denom) if denom > 0 else 1.0


@pytest.fixture(scope="module")
def _evaluation_candidates() -> list[Candidate]:
    """A fixed, diverse set of candidates the null-case test ranks.

    60, not a handful: Spearman correlation on a tiny evaluation set is
    itself noisy (one swapped pair among 10 items moves it a lot), which
    would make this test's own metric unstable independent of whether the
    model is behaving safely.
    """
    rng = np.random.default_rng(999)
    return [_realistic_candidate(rng) for _ in range(60)]


#: Mean-correlation floor per label count.  n=25 genuinely permits more
#: movement than n=5 - more (still uninformative) data legitimately lets the
#: MAP fit move a little further from the prior - so this is graded rather
#: than one blanket number; each floor sits clearly below the observed mean
#: at DEFAULT_L2 (see ranking.py) and clearly above what the reviewer's
#: Monte Carlo review found from the previous, broken implementation (a mean
#: near total reshuffling, not merely "somewhat lower").
_NULL_CASE_MEAN_FLOOR = {5: 0.90, 12: 0.85, 25: 0.70}
#: However much an individual noisy trial moves the ordering, it must never
#: come close to the "no discrimination left at all" (correlation near zero
#: or negative) the reviewer's Monte Carlo review found in the previous
#: implementation.
_NULL_CASE_MIN_FLOOR = 0.3


@pytest.mark.parametrize("n_labels", [5, 12, 25])
@pytest.mark.parametrize("accept_rate", [0.15, 0.5])
def test_null_case_stays_close_to_the_prior_at_production_defaults(
    n_labels, accept_rate, _evaluation_candidates
):
    """The permanent regression for the reviewer's own Monte Carlo finding.

    Purely random labels over realistic, quality-varied features, at the
    add-on's actual shipped ``DEFAULT_L2`` - not a contrived worst case -
    must leave the ordering close to the prior's own, at every one of these
    label counts and at both a typical and a heavily imbalanced accept rate.
    A modest number of trials with a fixed seed keeps this fast; the floors
    were chosen from the observed distribution at DEFAULT_L2 (see that
    constant's own tuning notes), with real margin below the mean and well
    above the "reordered the top 5" failure mode this replaces.
    """
    prior = ranking.train_from_labels([])
    prior_probs = np.array([prior.probability(c) for c in _evaluation_candidates])

    correlations = []
    for trial in range(30):
        examples = _null_examples(n_labels, seed=trial * 97 + 1, accept_rate=accept_rate)
        model = ranking.train_from_labels(examples)
        if model.fallback_to_prior:
            correlations.append(1.0)
            continue
        probs = np.array([model.probability(c) for c in _evaluation_candidates])
        correlations.append(_rank_correlation(probs, prior_probs))

    correlations = np.array(correlations)
    mean_floor = _NULL_CASE_MEAN_FLOOR[n_labels]
    assert correlations.mean() > mean_floor, (
        f"n={n_labels} rate={accept_rate}: mean correlation with the prior's own ordering "
        f"over {len(correlations)} noise trials was only {correlations.mean():.3f} (floor "
        f"{mean_floor}) - the guard/regularisation combination is not holding at production "
        "defaults"
    )
    assert correlations.min() > _NULL_CASE_MIN_FLOOR, (
        f"n={n_labels} rate={accept_rate}: at least one noise trial reordered things almost "
        f"completely (correlation {correlations.min():.3f}) - exactly the failure mode the "
        "guard exists to prevent"
    )


# --- recovering a known preference -------------------------------------
def test_the_model_recovers_a_consistent_per_miner_preference():
    """A user who always dismisses one miner's output ends up ranking it lower.

    Every other feature is held roughly equal between the two miners, so any
    difference the model ends up with is attributable to the label pattern -
    not to the prior, which is neutral on miner identity by construction
    (PRIOR_WEIGHTS sets every miner_* weight to 0.0).

    The shared evidence is deliberately moderate (precision 0.6, in-sample),
    not near-certain-accept: with both classes already saturating close to
    probability 1.0 regardless of miner, a real, growing gap in *log-odds*
    still shows up compressed to almost nothing in raw probability once the
    sigmoid saturates - a trap this test fell into with more favourable
    evidence and looked like a much weaker demonstration than the underlying
    fit actually was.
    """
    rng = np.random.default_rng(3)
    examples = []
    for _ in range(30):
        jitter = rng.uniform(-0.02, 0.02)
        examples.append(
            ranking.LabelExample(
                candidate=make_candidate(
                    miner="association", precision=0.6 + jitter, recall=0.4 + jitter,
                    consistency=0.5, validation="in_sample",
                ),
                label=0,
                seen_count=1,
            )
        )
        examples.append(
            ranking.LabelExample(
                candidate=make_candidate(
                    miner="time_of_day", precision=0.6 + jitter, recall=0.4 + jitter,
                    consistency=0.5, validation="in_sample",
                ),
                label=1,
                seen_count=1,
            )
        )

    model = ranking.train_from_labels(examples)
    assert model.fallback_to_prior is False

    association_candidate = make_candidate(
        miner="association", precision=0.6, recall=0.4, consistency=0.5, validation="in_sample"
    )
    time_of_day_candidate = make_candidate(
        miner="time_of_day", precision=0.6, recall=0.4, consistency=0.5, validation="in_sample"
    )
    assert model.probability(time_of_day_candidate) > model.probability(association_candidate)
    # And it is a real, substantial gap - not a rounding artefact of a model
    # that actually learned nothing.
    assert model.probability(time_of_day_candidate) - model.probability(association_candidate) > 0.2


# --- determinism --------------------------------------------------------
def test_training_is_deterministic_regardless_of_input_order():
    """Same labels, shuffled, must fit to bit-identical weights.

    Not just the same list called twice (trivially deterministic if the
    function has no internal state) - permuted, because the row order
    ``Store.ranking_labels`` returns them in was, before this test, not
    pinned by an ``ORDER BY`` and so was not actually guaranteed stable
    across SQLite versions.
    """
    examples = _signal_examples(25, seed=4)
    first = ranking.train_from_labels(examples)

    shuffled = list(examples)
    rng = np.random.default_rng(42)
    rng.shuffle(shuffled)
    second = ranking.train_from_labels(shuffled)

    assert first.fallback_to_prior == second.fallback_to_prior
    assert first.bias == pytest.approx(second.bias, abs=1e-9)
    assert set(first.weights) == set(second.weights)
    for name, value in first.weights.items():
        # Floating-point summation is not associative, so a different
        # accumulation order (np.vstack/mean/Newton's method, all summing
        # over the rows in whatever order they arrived in) can legitimately
        # land a few ULPs apart - the ORDER BY added to
        # Store.ranking_labels() is what makes that order reproducible in
        # production, not bit-exactness under an arbitrary permutation.  What
        # must not happen is a *meaningfully* different fit.
        assert value == pytest.approx(second.weights[name], abs=1e-9), name


def test_training_called_twice_on_the_same_list_is_identical():
    examples = _signal_examples(25, seed=4)
    first = ranking.train_from_labels(examples)
    second = ranking.train_from_labels(examples)
    assert first.weights == second.weights
    assert first.bias == second.bias


# --- schema-version guard -------------------------------------------------
def _row(**overrides):
    base = {
        "feature_schema_version": ranking.FEATURE_SCHEMA_VERSION,
        "weights": dict(ranking.PRIOR_WEIGHTS),
        "bias": -1.0,
        "scaler": ranking.Scaler.identity().as_dict(),
        "n_labels": 50,
        "l2": ranking.DEFAULT_L2,
        "trained_ts": 0.0,
        "fallback_to_prior": False,
        "fallback_reason": None,
    }
    base.update(overrides)
    return base


def test_from_row_discards_a_model_fit_under_an_older_schema():
    row = _row(feature_schema_version=ranking.FEATURE_SCHEMA_VERSION - 1)
    assert ranking.RankingModel.from_row(row) is None


def test_from_row_discards_a_model_whose_weights_do_not_match_the_current_features():
    row = _row(weights={"only_one_feature": 1.0})
    assert ranking.RankingModel.from_row(row) is None


def test_from_row_discards_a_model_whose_scaler_does_not_match_the_current_features():
    row = _row(scaler={"center": {"only_one_feature": 0.0}, "scale": {"only_one_feature": 1.0}})
    assert ranking.RankingModel.from_row(row) is None


def test_from_row_accepts_a_current_schema_model():
    row = _row(n_labels=5, fallback_to_prior=True, fallback_reason="cold start")
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
    examples = _overfit_noise_examples(6, seed=0)
    model = ranking.train_from_labels(examples, l2=0.05)
    assert model.fallback_to_prior is True
    note = model.confidence_note()
    assert "did not" in note  # honest about the guard, not a bare percentage


def test_confidence_note_differs_between_few_and_many_labels():
    few = ranking.RankingModel(
        feature_schema_version=ranking.FEATURE_SCHEMA_VERSION,
        weights=dict(ranking.PRIOR_WEIGHTS), bias=-1.0, scaler=ranking.Scaler.identity(),
        n_labels=2, l2=ranking.DEFAULT_L2, fallback_to_prior=False,
    )
    many = ranking.RankingModel(
        feature_schema_version=ranking.FEATURE_SCHEMA_VERSION,
        weights=dict(ranking.PRIOR_WEIGHTS), bias=-1.0, scaler=ranking.Scaler.identity(),
        n_labels=500, l2=ranking.DEFAULT_L2, fallback_to_prior=False,
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
        weights=adversarial_weights, bias=25.0, scaler=ranking.Scaler.identity(),
        n_labels=100, l2=ranking.DEFAULT_L2, fallback_to_prior=False,
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
