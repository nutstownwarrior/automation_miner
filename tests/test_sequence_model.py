"""The learned sequence model (amminer.learn.sequence).

What this file exists to prove, each picked because a subtly broken version
of this feature would still pass a superficial test:

* the hand-derived GRU backward pass is actually correct (finite-difference
  gradient check), not merely "the loss goes down",
* it recovers a real, three-way planted interaction that the pairwise
  association miner cannot express, and does not manufacture one from pure
  noise,
* it never learns from data outside the training window it is given,
* it is bit-reproducible given the same input, in a *fresh* process,
* a confident-looking, well-supported candidate it proposes still gets
  rejected by the ordinary backtest gate when it does not hold up out of
  sample - the gate is never bypassed,
* the capability gate declines honestly, with a specific reason, and never
  silently trains a smaller model instead,
* every decline reason is a real string, never blank, never overconfident.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from amminer import backtest as backtest_module
from amminer.config import Options
from amminer.enrich.signals import build_signal_store
from amminer.learn import sequence as sm
from amminer.miners import association
from amminer.recorderdb.models import Cause, StateChange

os.environ.setdefault("TZ", "UTC")

RUN_SLOW_MODEL_TESTS = os.environ.get("AMMINER_RUN_SLOW_MODEL_TESTS")
_slow_model = pytest.mark.skipif(
    not RUN_SLOW_MODEL_TESTS,
    reason="set AMMINER_RUN_SLOW_MODEL_TESTS=1 to run the sequence-model noise sweep "
    "(see the dedicated test-model-sweeps CI job)",
)


@pytest.fixture
def options() -> Options:
    return Options(sequence_model_enabled=True)


# ---------------------------------------------------------------------------
# Capability gate
# ---------------------------------------------------------------------------


def test_capability_ok_on_an_ordinary_host():
    cap = sm.capability(cpu_count=4, available_memory_bytes=2 * 1024 * 1024 * 1024)
    assert cap.ok is True
    assert cap.reason is None


def test_capability_declines_on_too_few_cores():
    cap = sm.capability(cpu_count=1, available_memory_bytes=2 * 1024 * 1024 * 1024)
    assert cap.ok is False
    assert cap.reason
    assert "CPU" in cap.reason


def test_capability_declines_on_too_little_memory():
    cap = sm.capability(cpu_count=4, available_memory_bytes=64 * 1024 * 1024)
    assert cap.ok is False
    assert cap.reason
    assert "memory" in cap.reason


def test_capability_unmeasurable_memory_does_not_block(monkeypatch):
    """A host this cannot be measured on is noted as unmeasured, never
    silently treated as failing - only a floor it actually cleared or
    failed to clear blocks it."""
    monkeypatch.setattr(sm, "_available_memory_bytes", lambda: None)
    cap = sm.capability(cpu_count=4)
    assert cap.ok is True
    assert cap.available_memory_bytes is None


def test_capability_as_dict_reports_mb_not_bytes():
    cap = sm.capability(cpu_count=4, available_memory_bytes=1024 * 1024 * 1024)
    payload = cap.as_dict()
    assert payload["available_memory_mb"] == 1024.0
    assert payload["cpu_count"] == 4
    assert payload["ok"] is True


def test_cgroup_v2_limit_is_read(tmp_path: Path):
    (tmp_path / "memory.max").write_text("1073741824\n")  # 1 GiB
    (tmp_path / "memory.current").write_text("268435456\n")  # 256 MiB used
    assert sm._cgroup_available_memory_bytes(str(tmp_path)) == 1073741824 - 268435456


def test_cgroup_v2_unlimited_is_ignored(tmp_path: Path):
    (tmp_path / "memory.max").write_text("max\n")
    assert sm._cgroup_available_memory_bytes(str(tmp_path)) is None


def test_cgroup_v1_limit_is_read(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("536870912\n")  # 512 MiB
    (tmp_path / "memory" / "memory.usage_in_bytes").write_text("100000000\n")
    assert sm._cgroup_available_memory_bytes(str(tmp_path)) == 536870912 - 100000000


def test_cgroup_v1_unlimited_sentinel_is_ignored(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (tmp_path / "memory" / "memory.usage_in_bytes").write_text("100000000\n")
    assert sm._cgroup_available_memory_bytes(str(tmp_path)) is None


def test_cgroup_absent_is_none(tmp_path: Path):
    assert sm._cgroup_available_memory_bytes(str(tmp_path / "does-not-exist")) is None


def test_available_memory_is_the_tighter_of_host_and_cgroup(monkeypatch):
    """A roomy host with a tightly-capped container must not clear the gate
    on the host's figure alone - this is the exact OOM-kill scenario a
    cgroup-blind check would miss (see _cgroup_available_memory_bytes)."""
    monkeypatch.setattr(sm, "_host_available_memory_bytes", lambda: 16 * 1024 * 1024 * 1024)
    monkeypatch.setattr(sm, "_cgroup_available_memory_bytes", lambda: 200 * 1024 * 1024)
    assert sm._available_memory_bytes() == 200 * 1024 * 1024

    monkeypatch.setattr(sm, "_cgroup_available_memory_bytes", lambda: None)
    assert sm._available_memory_bytes() == 16 * 1024 * 1024 * 1024


def test_capability_declines_when_only_the_container_is_too_tight(monkeypatch):
    """The scenario issue #3 names directly: a roomy host, a container
    capped below the floor by its own cgroup - the gate must still decline."""
    monkeypatch.setattr(sm, "_host_available_memory_bytes", lambda: 16 * 1024 * 1024 * 1024)
    monkeypatch.setattr(sm, "_cgroup_available_memory_bytes", lambda: 200 * 1024 * 1024)
    cap = sm.capability(cpu_count=4)
    assert cap.ok is False
    assert "memory" in cap.reason


def test_fit_declines_when_host_cannot_afford_it(monkeypatch, options):
    monkeypatch.setattr(sm, "capability", lambda: sm.Capability(False, "only 1 CPU core(s) detected", 1, None))
    model, examples = sm.fit([], options, (0.0, 40 * 86400.0))
    assert model.fitted is False
    assert examples == []
    assert model.fallback_reason
    assert "cannot afford" in model.fallback_reason


# ---------------------------------------------------------------------------
# Gradient correctness
# ---------------------------------------------------------------------------


def test_gradients_match_finite_differences(monkeypatch):
    """The actual evidence the hand-derived backward pass is correct - not
    the derivation comments in ``amminer/learn/sequence.py``."""
    monkeypatch.setattr(sm, "WINDOW", 3)
    monkeypatch.setattr(sm, "EMBED_ITEM_DIM", 3)
    monkeypatch.setattr(sm, "EMBED_DT_DIM", 2)
    monkeypatch.setattr(sm, "HIDDEN_SIZE", 4)

    rng = np.random.default_rng(0)
    vocab_size = 5
    n = 4
    item_idx = rng.integers(0, vocab_size + 2, size=(n, sm.WINDOW))
    dt_idx = rng.integers(0, sm.DT_VOCAB_SIZE, size=(n, sm.WINDOW))
    labels = rng.integers(0, vocab_size, size=(n,))

    params = sm._init_params(np.random.default_rng(1), vocab_size)
    cache = sm._forward(params, item_idx, dt_idx)
    grads = sm._backward(params, cache, labels)

    eps = 1e-5
    checked = 0
    for name, arr in params.items():
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            if name == "item_emb" and idx[0] == sm.PAD_INDEX:
                continue  # pinned at zero on purpose - see _init_params
            orig = arr[idx]
            arr[idx] = orig + eps
            loss_plus, _ = sm._loss(sm._forward(params, item_idx, dt_idx)["logits"], labels, params)
            arr[idx] = orig - eps
            loss_minus, _ = sm._loss(sm._forward(params, item_idx, dt_idx)["logits"], labels, params)
            arr[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = grads[name][idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            assert abs(numeric - analytic) / denom < 1e-4, (name, idx, numeric, analytic)
            checked += 1
    assert checked > 20  # the loop above actually ran, not a vacuous pass


def test_pad_row_never_moves():
    """The PAD embedding row is pinned at zero, not merely initialised
    there - a fit that let it drift would let padding leak a fake signal
    into every short context."""
    rng = np.random.default_rng(2)
    params = sm._init_params(rng, vocab_size=6)
    assert np.all(params["item_emb"][sm.PAD_INDEX] == 0.0)
    item_idx = rng.integers(0, 8, size=(10, sm.WINDOW))
    dt_idx = rng.integers(0, sm.DT_VOCAB_SIZE, size=(10, sm.WINDOW))
    labels = rng.integers(0, 6, size=(10,))
    grads = sm._backward(params, sm._forward(params, item_idx, dt_idx), labels)
    assert np.all(grads["item_emb"][sm.PAD_INDEX] == 0.0)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _arrival_events(base: float, rng: random.Random, with_light: bool) -> list[StateChange]:
    """One "someone came home" episode: door, then motion a few seconds
    later - and, when ``with_light`` is true, the light a few seconds after
    that. Reused for both the true-arrival pattern (see
    ``_three_way_fixture``) and the holdout period in
    ``test_a_pattern_that_reverses_in_holdout_is_rejected_by_backtest``,
    where ``with_light`` is false because the pattern no longer holds."""
    t = base + rng.uniform(0, 86000.0)
    events = [_ch("binary_sensor.door_front", "on", t, Cause.DEVICE)]
    mt = t + rng.uniform(2, 8)
    events.append(_ch("binary_sensor.motion_hall", "on", mt, Cause.DEVICE))
    if with_light:
        events.append(_ch("light.hall", "on", mt + rng.uniform(2, 6), Cause.HUMAN))
    return events


def _three_way_fixture(days: int = 60, seed: int = 11):
    """A pattern no single-antecedent (pairwise) rule can express - each of
    ``binary_sensor.door_front`` and ``binary_sensor.motion_hall`` alone is a
    *weak, misleading* predictor of the light; only the conjunction, with the
    right ordering, is strong.

    The naive version of this fixture (an earlier draft of this test) made
    door and motion fire often *without* any other human-caused action
    nearby when they were not followed by light. ``amminer.miners.association``
    only ever builds a transaction ("basket") around a *human*-caused anchor
    event (see that module's own ``build_transactions`` docstring) - so those
    "door/motion without an arrival" events, isolated from every other human
    action, mostly never entered any basket at all, and confidence(door ->
    light) came out misleadingly high (~95%) purely because nearly every
    basket that happened to contain the door also happened to be anchored on
    the light itself. That was an accident of the fixture, not evidence about
    the mechanism - see the review this file's git history records.

    Here, every "false alarm" (door or motion with no arrival) is paired with
    its own real human action (``switch.decoy``) a few seconds later, so it
    *does* form its own basket - one that contains the door or motion but not
    the light. That is what actually drives confidence(door -> light) and
    confidence(motion -> light) down to a genuinely weak, sub-threshold level
    (see ``test_recovers_a_three_way_interaction_association_cannot_express``,
    which measures ~45% for both, below ``min_confidence``), while the
    ordered conjunction - door, then motion a few seconds later - still
    predicts the light almost perfectly.
    """
    rng = random.Random(seed)
    start = 1_700_000_000.0
    day_seconds = 86400.0
    changes: list[StateChange] = []
    for day in range(days):
        base = start + day * day_seconds
        # The real, reliable pattern: door, then motion, then light.
        for _ in range(4):
            changes.extend(_arrival_events(base, rng, with_light=True))
        # False alarm: door opens, nobody arrives - a real human action
        # (not light) follows nearby instead, so this becomes its own
        # basket containing the door but not the light.
        for _ in range(5):
            t = base + rng.uniform(0, day_seconds - 60)
            changes.append(_ch("binary_sensor.door_front", "on", t, Cause.DEVICE))
            changes.append(_ch("switch.decoy", "on", t + rng.uniform(2, 8), Cause.HUMAN))
        # False alarm: motion elsewhere, unrelated to the door, likewise
        # anchored by a real human action rather than left isolated.
        for _ in range(5):
            t = base + rng.uniform(0, day_seconds - 60)
            changes.append(_ch("binary_sensor.motion_hall", "on", t, Cause.DEVICE))
            changes.append(_ch("switch.decoy", "off", t + rng.uniform(2, 8), Cause.HUMAN))
        # Unrelated household noise across many other entities/domains.
        for _ in range(5):
            t = base + rng.uniform(0, day_seconds)
            entity = f"switch.noise{rng.randint(0, 5)}"
            changes.append(_ch(entity, rng.choice(["on", "off"]), t, Cause.HUMAN))
    changes.sort(key=lambda c: c.ts)
    end = changes[-1].ts + 3600.0
    return changes, (start, end)


def _ch(entity_id: str, state: str, ts: float, cause: Cause) -> StateChange:
    old = "off" if state == "on" else "on"
    return StateChange(entity_id=entity_id, state=state, ts=ts, old_state=old, cause=cause)


def _noise_only_fixture(days: int, seed: int):
    """Every entity's state is independent random noise - nothing to learn."""
    rng = random.Random(seed)
    start = 1_700_000_000.0
    day_seconds = 86400.0
    changes: list[StateChange] = []
    entities = [f"light.room{i}" for i in range(6)] + [f"binary_sensor.s{i}" for i in range(6)]
    for day in range(days):
        base = start + day * day_seconds
        for _ in range(rng.randint(20, 35)):
            t = base + rng.uniform(0, day_seconds)
            entity = rng.choice(entities)
            cause = Cause.HUMAN if entity.startswith("light") else Cause.DEVICE
            changes.append(_ch(entity, rng.choice(["on", "off"]), t, cause))
    changes.sort(key=lambda c: c.ts)
    end = changes[-1].ts + 3600.0
    return changes, (start, end)


# ---------------------------------------------------------------------------
# Decline reasons - honest, specific, never blank
# ---------------------------------------------------------------------------


def test_fit_declines_with_too_little_history(options):
    changes, _window = _three_way_fixture(days=5)
    model, examples = sm.fit(changes, options, (changes[0].ts, changes[0].ts + 5 * 86400.0))
    assert model.fitted is False
    assert examples == []
    assert "days" in model.fallback_reason
    assert str(options.sequence_model_min_train_days) in model.fallback_reason


def test_fit_declines_with_too_few_examples(options):
    start = 1_700_000_000.0
    window = (start, start + 40 * 86400.0)
    changes = [
        _ch("light.kitchen", "on", start + i * 86400.0, Cause.HUMAN) for i in range(3)
    ]
    model, examples = sm.fit(changes, options, window)
    assert model.fitted is False
    assert examples == []
    assert "examples" in model.fallback_reason


def test_disabled_by_default():
    assert Options().sequence_model_enabled is False


# ---------------------------------------------------------------------------
# Holdout discipline
# ---------------------------------------------------------------------------


def test_never_learns_from_rows_outside_the_training_window(options):
    changes, window = _three_way_fixture(days=60)
    start, end = window
    train_window = (start, start + 40 * 86400.0)
    train_changes = [c for c in changes if c.ts < train_window[1]]

    model_bounded, examples_bounded = sm.fit(train_changes, options, train_window)
    # A caller that (by mistake) also hands over the holdout rows, alongside
    # the correct train_window bounds, must not change anything - the same
    # guarantee amminer.learn.home_mode makes for build_feature_matrix.
    model_leaky, examples_leaky = sm.fit(changes, options, train_window)

    assert model_bounded.fitted and model_leaky.fitted
    assert model_bounded.n_examples == model_leaky.n_examples
    assert model_bounded.vocab == model_leaky.vocab
    assert [e.ts for e in examples_bounded] == [e.ts for e in examples_leaky]


# ---------------------------------------------------------------------------
# Recovering a real interaction pairwise mining cannot express
# ---------------------------------------------------------------------------


def _sequence_model_matches(candidates, *, target_entity: str, required_entities: set[str]):
    """Candidates targeting ``target_entity`` whose trigger+conditions cover
    at least ``required_entities`` - a superset match, not an exact one,
    because ``CANDIDATE_MAX_CONDITIONS`` fills both condition slots whenever
    a context has that many distinct entities available, so the true
    (door, motion) pairing is often reported alongside one more incidental,
    equally-real-but-not-load-bearing entity (see the module's own extraction
    code) rather than alone."""
    out = []
    for c in candidates:
        if not c.actions or c.actions[0].entity_id != target_entity:
            continue
        entities = {t.entity_id for t in c.triggers} | {cond.entity_id for cond in c.conditions}
        if required_entities <= entities:
            out.append(c)
    return out


def test_recovers_a_three_way_interaction_association_cannot_express(options):
    """The whole argument for this feature existing: it finds a real
    interaction that ``amminer.miners.association`` cannot, by mechanism -
    not by an accident of this fixture (see ``_three_way_fixture``'s own
    docstring for the earlier, misleading version of this test)."""
    changes, window = _three_way_fixture(days=60)
    model, examples = sm.fit(changes, options, window)
    assert model.fitted, model.fallback_reason

    candidates = sm.extract_candidates(model, examples, options, window, resolver=None)
    matches = _sequence_model_matches(
        candidates,
        target_entity="light.hall",
        required_entities={"binary_sensor.door_front", "binary_sensor.motion_hall"},
    )
    assert matches, "expected at least one (door, motion) -> light candidate"
    # The strongest one, by the counted ratio the backtest gate would also
    # see - not cherry-picked by model probability, which is not comparable
    # to it (see amminer/miners/base.py's own warning on this).
    candidate = max(matches, key=lambda c: c.evidence.confidence)
    assert candidate.miner == "sequence_model"
    assert candidate.evidence.confidence >= options.sequence_model_min_confidence
    assert candidate.evidence.occurrences >= options.sequence_model_min_occurrences
    assert candidate.evidence.extra["model_probability"] >= options.sequence_model_min_model_probability
    # Honesty: the model's own probability is reported, never substituted
    # for the counted ratio.
    assert candidate.evidence.confidence != candidate.evidence.extra["model_probability"]
    assert any("Model-derived" in note for note in candidate.evidence.notes)

    # The default, gated association miner: genuinely finds nothing for
    # light.hall on this fixture (not an accident - see below).
    assoc_candidates = association.mine(changes, options, window)
    assert not any(c.actions and c.actions[0].entity_id == "light.hall" for c in assoc_candidates)

    # The *mechanism*, isolated from every quality gate that might otherwise
    # be blamed for the empty result above: rebuild the same rules with
    # confidence and lift fully relaxed (min_support left alone, so this
    # still excludes pure sampling noise - see the docstring above). Even
    # with every quality bar removed, a rule can only ever pair one
    # antecedent item with one consequent item - fpgrowth is called with
    # max_len=2 in amminer.miners.association, so {door, motion} can never
    # become one antecedent. This is what actually bounds association's
    # best possible answer for this target, independent of any threshold.
    permissive = Options(min_confidence=0.0, min_lift=0.0, min_support=options.min_support)
    transactions, _counts, _order = association.build_transactions(
        changes, permissive.association_window_seconds, permissive
    )
    # Directly verifies the mechanism, not just its downstream effect: even
    # the raw frequent-itemset search (before confidence/lift/direction are
    # applied at all) never considers a three-item set - max_len=2 in
    # amminer.miners.association's own fpgrowth call rules out {door,
    # motion, light} as a single itemset from the start.
    import pandas as pd
    from mlxtend.frequent_patterns import fpgrowth

    frame = association.one_hot_encode(transactions, pd)
    frequent = fpgrowth(frame, min_support=permissive.min_support, use_colnames=True, max_len=2)
    assert not frequent.empty
    assert frequent["itemsets"].apply(len).max() <= 2

    raw_rules = association._rules_from_transactions(transactions, permissive)
    assert raw_rules, "expected fpgrowth to find *some* pairwise rules on this fixture"
    light_rules = [r for r in raw_rules if r["consequent"] == "light.hall=on"]
    assert light_rules, "expected door/motion single-item rules for light.hall to survive relaxed thresholds"
    best_pairwise_confidence = max(r["confidence"] for r in light_rules)

    # Unconditional: the best the pairwise miner can do for this target, with
    # every quality gate but support removed, is still markedly weaker than
    # what the sequence model found - because it is structurally blind to
    # the two-item conjunction that actually drives the outcome.
    assert best_pairwise_confidence < 0.6
    assert candidate.evidence.confidence > best_pairwise_confidence + 0.3


def test_finds_nothing_in_pure_noise():
    """The cheap, always-on half of the null-result check - see the
    slow_model sweep below for the multi-seed version."""
    options = Options(sequence_model_enabled=True)
    changes, window = _noise_only_fixture(days=60, seed=3)
    model, examples = sm.fit(changes, options, window)
    if not model.fitted:
        return  # too little to fit at all is also an acceptable honest outcome
    candidates = sm.extract_candidates(model, examples, options, window, resolver=None)
    assert candidates == []


@_slow_model
@pytest.mark.slow_model
def test_finds_nothing_in_pure_noise_across_seeds():
    """A confident-looking candidate must never be manufactured from data
    with nothing real in it, across many independent seeds - the multi-seed
    counterpart to amminer.learn.home_mode's own noise-sweep tests."""
    options = Options(sequence_model_enabled=True)
    for seed in range(10):
        changes, window = _noise_only_fixture(days=60, seed=100 + seed)
        model, examples = sm.fit(changes, options, window)
        if not model.fitted:
            continue
        candidates = sm.extract_candidates(model, examples, options, window, resolver=None)
        assert candidates == [], f"seed {seed} manufactured a candidate from pure noise"


# ---------------------------------------------------------------------------
# Never gates: a confident-looking candidate still fails a real backtest
# ---------------------------------------------------------------------------


def test_a_pattern_that_reverses_in_holdout_is_rejected_by_backtest():
    """The candidate this module proposes is confident and well-supported
    *in training* - and still rejected once the ordinary backtest gate
    replays it against history where the pattern does not hold, exactly as
    it would reject any other miner's candidate. This module never
    special-cases its own output past that gate."""
    options = Options(sequence_model_enabled=True, sequence_model_min_train_days=10)
    train_changes, train_window = _three_way_fixture(days=40, seed=21)
    start, train_end = train_window

    # The holdout: door and motion still fire the same way, but the light
    # never follows any more - the learned pattern has reversed.
    rng = random.Random(99)
    holdout_days = 15
    holdout_changes: list[StateChange] = []
    for day in range(holdout_days):
        base = train_end + day * 86400.0
        for _ in range(4):
            holdout_changes.extend(_arrival_events(base, rng, with_light=False))
    holdout_changes.sort(key=lambda c: c.ts)
    full_changes = sorted(train_changes + holdout_changes, key=lambda c: c.ts)
    full_window = (start, holdout_changes[-1].ts + 3600.0)

    model, examples = sm.fit(train_changes, options, train_window)
    assert model.fitted, model.fallback_reason
    candidates = sm.extract_candidates(model, examples, options, train_window, resolver=None)
    matches = _sequence_model_matches(
        candidates,
        target_entity="light.hall",
        required_entities={"binary_sensor.door_front", "binary_sensor.motion_hall"},
    )
    assert matches, "fixture must still produce a (door, motion) -> light candidate"
    candidate = max(matches, key=lambda c: c.evidence.confidence)

    all_entities = {c.entity_id for c in full_changes}
    store = build_signal_store(full_changes, all_entities, None, full_window)
    passed, rejected = backtest_module.backtest_all(
        [candidate], full_changes, store, options, full_window, validate_holdout=False
    )
    assert passed == []
    assert len(rejected) == 1
    assert rejected[0].backtest["passed"] is False


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_fit_is_deterministic_within_a_process(options):
    changes, window = _three_way_fixture(days=60, seed=5)
    model_a, _ = sm.fit(changes, options, window)
    model_b, _ = sm.fit(changes, options, window)
    assert model_a.fitted and model_b.fitted
    for name in model_a.params:
        assert np.array_equal(model_a.params[name], model_b.params[name])
    assert model_a.train_loss == model_b.train_loss


def test_fit_is_deterministic_across_processes(tmp_path: Path):
    """The same input must give the same weights in a *fresh* process - not
    merely the same process twice, which would miss anything seeded from
    wall-clock time or PYTHONHASHSEED-sensitive ordering."""
    script = tmp_path / "fit_once.py"
    amminer_root = str(Path(__file__).resolve().parents[1] / "automation_miner")
    script.write_text(
        "import json, os, random, sys\n"
        "os.environ['TZ'] = 'UTC'\n"
        f"sys.path.insert(0, {amminer_root!r})\n"
        "from amminer.config import Options\n"
        "from amminer.learn import sequence as sm\n"
        "from amminer.recorderdb.models import Cause, StateChange\n"
        "\n"
        "def ch(entity_id, state, ts, cause):\n"
        "    old = 'off' if state == 'on' else 'on'\n"
        "    return StateChange(entity_id=entity_id, state=state, ts=ts, old_state=old, cause=cause)\n"
        "\n"
        "rng = random.Random(42)\n"
        "start = 1_700_000_000.0\n"
        "changes = []\n"
        "for day in range(50):\n"
        "    base = start + day * 86400.0\n"
        "    for _ in range(4):\n"
        "        t = base + rng.uniform(0, 86000.0)\n"
        "        changes.append(ch('binary_sensor.door_front', 'on', t, Cause.DEVICE))\n"
        "        changes.append(ch('binary_sensor.motion_hall', 'on', t + 4.0, Cause.DEVICE))\n"
        "        changes.append(ch('light.hall', 'on', t + 8.0, Cause.HUMAN))\n"
        "    for _ in range(5):\n"
        "        t = base + rng.uniform(0, 86000.0)\n"
        "        changes.append(ch(f'switch.noise{rng.randint(0,4)}', rng.choice(['on','off']), t, Cause.HUMAN))\n"
        "changes.sort(key=lambda c: c.ts)\n"
        "window = (start, changes[-1].ts + 3600.0)\n"
        "options = Options(sequence_model_enabled=True)\n"
        "model, _ = sm.fit(changes, options, window)\n"
        "print(json.dumps({\n"
        "    'fitted': model.fitted,\n"
        "    'train_loss': model.train_loss,\n"
        "    'train_accuracy': model.train_accuracy,\n"
        "    'wo': model.params['Wo'].tolist() if model.fitted else None,\n"
        "}))\n"
    )
    results = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, check=True, timeout=120
        )
        results.append(json.loads(proc.stdout))

    assert results[0]["fitted"] and results[1]["fitted"]
    assert results[0]["train_loss"] == results[1]["train_loss"]
    assert results[0]["train_accuracy"] == results[1]["train_accuracy"]
    assert np.allclose(results[0]["wo"], results[1]["wo"])


# ---------------------------------------------------------------------------
# Redundant-condition pruning
# ---------------------------------------------------------------------------


def test_drops_a_superset_rule_that_adds_no_confidence():
    simple = sm._Rule(
        signature=("binary_sensor.motion_hall", "on", ()),
        target_key=("light.hall", "on"),
        rows=(0, 1, 2, 3, 4),
        occurrences=5,
        opportunities=5,
        confidence=1.0,
        model_confidence=0.9,
    )
    superset = sm._Rule(
        signature=("binary_sensor.motion_hall", "on", (("switch.noise0", "on"),)),
        target_key=("light.hall", "on"),
        rows=(0, 1, 2),
        occurrences=3,
        opportunities=3,
        confidence=1.0,  # no better than `simple`, just narrower
        model_confidence=0.9,
    )
    kept = sm._drop_redundant_supersets([simple, superset])
    assert kept == [simple]


def test_keeps_a_superset_rule_that_is_meaningfully_more_confident():
    weak = sm._Rule(
        signature=("binary_sensor.motion_hall", "on", ()),
        target_key=("light.hall", "on"),
        rows=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        occurrences=5,
        opportunities=10,
        confidence=0.5,
        model_confidence=0.5,
    )
    strong_superset = sm._Rule(
        signature=("binary_sensor.motion_hall", "on", (("binary_sensor.door_front", "on"),)),
        target_key=("light.hall", "on"),
        rows=(0, 1, 2, 3, 4),
        occurrences=5,
        opportunities=5,
        confidence=1.0,
        model_confidence=0.95,
    )
    kept = sm._drop_redundant_supersets([weak, strong_superset])
    assert set(kept) == {weak, strong_superset}


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


def test_options_clamp_sequence_model_fields():
    options = Options(
        sequence_model_min_train_days=-5,
        sequence_model_min_confidence=5.0,
        sequence_model_min_model_probability=-1.0,
        sequence_model_min_occurrences=1,
    )
    assert options.sequence_model_min_train_days == 1
    assert options.sequence_model_min_confidence == 1.0
    assert options.sequence_model_min_model_probability == 0.0
    assert options.sequence_model_min_occurrences == 2


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def test_disabled_by_default_produces_no_report_entry_and_no_candidates(
    ha_config_dir, store, fake_client
):
    from amminer.discovery.ha_config import HAConfig
    from amminer.pipeline import run_analysis

    options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir))
    report, candidates = run_analysis(options, store, fake_client, HAConfig(ha_config_dir))

    assert report.sequence_model == {
        "fitted": False,
        "fallback_reason": "disabled (sequence_model_enabled=false)",
    }
    assert not any(c.miner == "sequence_model" for c in candidates)


def test_pipeline_fits_sequence_model_on_the_training_window_only(
    ha_config_dir, store, fake_client, monkeypatch
):
    """The actual wiring, not just the module's own robustness (see
    test_never_learns_from_rows_outside_the_training_window above):
    amminer.pipeline must call sequence.fit with the train window, never the
    full one - the same guarantee test_home_mode.py pins down for that
    module."""
    from amminer import pipeline as pipeline_module
    from amminer.discovery.ha_config import HAConfig

    calls: list[tuple[float, float]] = []
    real_fit = pipeline_module.sequence_model_module.fit

    def spy(changes, options, train_window):
        calls.append(train_window)
        return real_fit(changes, options, train_window)

    monkeypatch.setattr(pipeline_module.sequence_model_module, "fit", spy)

    options = Options(
        ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir), sequence_model_enabled=True
    )
    report, _candidates = pipeline_module.run_analysis(
        options, store, fake_client, HAConfig(ha_config_dir)
    )

    assert calls, "sequence.fit was never called"
    called_window = calls[0]
    expected = backtest_module.split_window(report.window, options).train
    assert called_window == expected
    assert called_window[1] <= report.window[1]


def test_enabling_it_end_to_end_reports_honestly_either_way(ha_config_dir, store, fake_client):
    """On ordinary CI/dev hardware the capability gate clears, so this
    either fits and reports real stats, or declines with a specific,
    non-blank reason - never silent, never blank, never a bare 'False'."""
    from amminer.discovery.ha_config import HAConfig
    from amminer.pipeline import run_analysis

    options = Options(
        ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir), sequence_model_enabled=True
    )
    report, candidates = run_analysis(options, store, fake_client, HAConfig(ha_config_dir))

    assert report.status == "ok"
    assert "fitted" in report.sequence_model
    if report.sequence_model["fitted"]:
        assert report.sequence_model["training_seconds"] >= 0.0
        assert isinstance(report.sequence_model["n_examples"], int)
    else:
        assert report.sequence_model["fallback_reason"]
    # Whatever it proposed, if anything, went through the ordinary gate:
    # every sequence_model candidate that reached the suggestion store
    # carries a backtest result, exactly like every other miner's.
    for candidate in candidates:
        if candidate.miner == "sequence_model":
            assert candidate.backtest is not None
