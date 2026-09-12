"""The inferred household-mode model (amminer.learn.home_mode).

Five things this file exists to prove, each picked because a subtly broken
version of this feature would still pass a superficial test:

* it recovers a real, planted latent structure from synthetic activity,
* it does not manufacture states a quiet, simple home does not have,
* it is bit-reproducible given the same input, in a *fresh* process,
* it never learns anything from data outside the window it is given, even
  when a caller hands it extra rows by mistake,
* a miner actually produces a better-conditioned candidate once the signal
  is available, where clock time alone was not enough.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
from amminer import backtest as backtest_module
from amminer.config import Options
from amminer.enrich.detect import SignalSet
from amminer.enrich.signals import SignalStore
from amminer.learn import home_mode as hm
from amminer.miners import conditional, time_of_day
from amminer.miners.time_of_day import human_action_events
from amminer.recorderdb.models import Cause, StateChange
from amminer.store import Store
from amminer.testing.synthetic import build_two_regime_activity, build_winddown_habit_activity

os.environ.setdefault("TZ", "UTC")


@pytest.fixture
def options() -> Options:
    return Options()


def _best_agreement(labels: np.ndarray, truth: list[int], n_states: int) -> float:
    """Best label-permutation match - states are unlabelled, so index 0 in
    the fit need not be index 0 in the ground truth."""
    truth_arr = np.asarray(truth[: len(labels)])
    best = 0.0
    for perm in permutations(range(n_states)):
        mapped = np.array([perm[label] for label in labels])
        best = max(best, float((mapped == truth_arr).mean()))
    return best


# --- recovering a planted structure -------------------------------------
def _clean_two_regime(days: int, seed: int = 1, busy_start_hour: int = 8, busy_end_hour: int = 22):
    """A regime whose ground truth is directly what the activity shows.

    build_two_regime_activity's own bursts are short and randomly timed
    within an hour that is merely *eligible* for activity - which is
    realistic, but means "the hour is busy" and "there is visible activity"
    are not the same claim, and a model reading only activity has no way to
    recover an "eligibility" label it was never shown evidence of (see the
    diagnosis in this test's own history). This fixture makes the two claims
    the same one on purpose - each busy hour is active for (almost) the
    whole hour, aligned to bin boundaries - specifically so recovery against
    ground truth is a meaningful thing to assert on.
    """
    rng = random.Random(seed)
    entity_id = "light.living"
    changes: list[StateChange] = []
    n_bins = int(days * 86400.0 // 900.0)
    regime_by_bin = [0] * n_bins
    for bin_idx in range(n_bins):
        hour = int((bin_idx * 900.0 % 86400.0) // 3600)
        regime_by_bin[bin_idx] = 1 if busy_start_hour <= hour < busy_end_hour else 0
    for day in range(days):
        for hour in range(24):
            hour_start = day * 86400.0 + hour * 3600.0
            busy = busy_start_hour <= hour < busy_end_hour
            p = 0.95 if busy else 0.03
            if rng.random() >= p:
                continue
            duration = 3600.0 if busy else rng.uniform(300.0, 600.0)
            on = StateChange(entity_id, "on", hour_start, old_state="off")
            on.cause = Cause.HUMAN
            changes.append(on)
            off = StateChange(entity_id, "off", hour_start + duration, old_state="on")
            off.cause = Cause.HUMAN
            changes.append(off)
    return changes, (0.0, days * 86400.0), regime_by_bin


def test_recovers_known_modes_from_synthetic_data(options):
    """A clean two-regime household: the decoded labels should track it."""
    changes, window, regime_by_bin = _clean_two_regime(days=45, seed=1)
    model = hm.fit(changes, SignalSet(), options, window)
    assert model.fitted

    counts, first_idx = hm.build_feature_matrix(changes, SignalSet(), options, *window)
    x = hm._to_features(counts)
    labels = hm.decode_labels(x, model.initial, model.transition, model.means, model.variances,
                               model.var_floor)
    assert first_idx == 0
    agreement = _best_agreement(labels, regime_by_bin, model.n_states)
    # Not 100%: causal filtering will occasionally mislabel a bin at a
    # transition edge, or when a "busy" hour happened to be quiet. A test
    # that demanded perfection would break on the first unlucky bin; one
    # that accepts anything would not be testing recovery at all.
    assert agreement > 0.85, f"decoded labels only agreed with ground truth {agreement:.0%}"


def test_recovered_states_are_honestly_described(options):
    """State summaries must describe what was observed, not invent a name."""
    fixture = build_two_regime_activity(days=45, seed=1)
    model = hm.fit(fixture.changes, SignalSet(), options, fixture.window)
    assert model.fitted
    assert len(model.state_summaries) == model.n_states
    busy = [s for s in model.state_summaries if s["top_domains"]]
    assert busy, "no state picked up the daytime light activity at all"
    for summary in model.state_summaries:
        # Never a name asserting a fact the model cannot know.
        assert summary["llm_label"] is None
        assert summary["llm_label_is_advisory"] is True
        assert 0.0 <= summary["occupancy_share"] <= 1.0
        assert summary["label"] == f"mode_{summary['state']}"


# --- state-count selection ------------------------------------------------
def test_state_count_stays_small_for_a_quiet_home(options):
    """A simple two-regime home must not be reported as having five modes.

    This is the test a Gaussian-mixture EM implementation with a naive
    in-sample criterion (plain BIC, or a bare argmax over held-out scores)
    fails: see amminer/learn/home_mode.py's select_model docstring for the
    two concrete failure modes this project's own development hit before
    landing on a significance-gated held-out-likelihood walk.
    """
    fixture = build_two_regime_activity(days=45, seed=1)
    model = hm.fit(fixture.changes, SignalSet(), options, fixture.window)
    assert model.fitted
    assert model.n_states == 2, (
        f"expected a simple two-regime home to keep to 2 states, got {model.n_states} "
        f"(scores: {model.score_by_states})"
    )
    assert not model.weakly_separated


def test_a_third_real_regime_is_still_found(options):
    """The flip side: real, separated structure must not be flattened away."""
    fixture = build_winddown_habit_activity(days=60, seed=3)
    model = hm.fit(fixture.changes, SignalSet(), options, fixture.window)
    assert model.fitted
    assert model.n_states >= 3, (
        "a genuinely distinct third regime (wind-down) should not collapse into two"
    )


def test_em_does_not_silently_collapse_to_one_effective_state():
    """A model that reports k>=2 but only ever occupies one of them is exactly
    as dishonest as reporting k=1 outright.

    A *single* EM run can land in a degenerate local optimum by chance - that
    is exactly why fitting is never done with just one (see RESTARTS and
    _drop_unoccupied_states's own docstring on this failure mode). What must
    never happen is the *actual* fitting path - best-of-restarts, then
    pruned - reporting two states while only ever deciding one of them, on
    data genuinely built to have two.
    """
    changes, window, _regime = _clean_two_regime(days=45, seed=1)
    x, _ = hm.build_feature_matrix(changes, SignalSet(), Options(), *window)
    x = hm._to_features(x)
    var_floor = hm._variance_floor(x)
    fits = hm._fit_candidates(x, var_floor, max_k_exclusive=len(x))
    best = hm._drop_unoccupied_states(fits[2], x, var_floor)
    labels = hm.decode_labels(x, best.initial, best.transition, best.means, best.variances,
                               var_floor)
    occupied = {int(label) for label in labels}
    assert len(occupied) == len(best.initial) == 2, (
        f"only state(s) {occupied} were ever decoded, on data built to have two"
    )


# --- determinism -----------------------------------------------------------
def test_fit_is_deterministic_within_a_process(options):
    fixture = build_two_regime_activity(days=45, seed=1)
    first = hm.fit(fixture.changes, SignalSet(), options, fixture.window)
    second = hm.fit(fixture.changes, SignalSet(), options, fixture.window)
    assert first.n_states == second.n_states
    assert np.array_equal(first.means, second.means)
    assert np.array_equal(first.transition, second.transition)
    assert np.array_equal(first.initial, second.initial)
    assert first.log_likelihood == second.log_likelihood


def test_fit_is_deterministic_across_processes(tmp_path: Path):
    """The same input must give the same model in a *fresh* Python process -
    not merely the same process twice, which would miss anything seeded from
    wall-clock time, PYTHONHASHSEED-sensitive dict/set ordering, or per-process
    global state."""
    script = tmp_path / "fit_once.py"
    amminer_root = str(Path(__file__).resolve().parents[1] / "automation_miner")
    script.write_text(
        "import json, sys, os\n"
        "os.environ['TZ'] = 'UTC'\n"
        f"sys.path.insert(0, {amminer_root!r})\n"
        "from amminer.config import Options\n"
        "from amminer.enrich.detect import SignalSet\n"
        "from amminer.learn import home_mode as hm\n"
        "from amminer.testing.synthetic import build_two_regime_activity\n"
        "fixture = build_two_regime_activity(days=45, seed=1)\n"
        "model = hm.fit(fixture.changes, SignalSet(), Options(), fixture.window)\n"
        "print(json.dumps({\n"
        "    'n_states': model.n_states,\n"
        "    'means': model.means.tolist(),\n"
        "    'transition': model.transition.tolist(),\n"
        "    'log_likelihood': model.log_likelihood,\n"
        "}))\n"
    )
    results = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, check=True, timeout=120
        )
        results.append(json.loads(proc.stdout))

    assert results[0]["n_states"] == results[1]["n_states"]
    assert results[0]["log_likelihood"] == results[1]["log_likelihood"]
    assert np.allclose(results[0]["means"], results[1]["means"])
    assert np.allclose(results[0]["transition"], results[1]["transition"])


# --- the holdout discipline -------------------------------------------------
def test_fit_ignores_activity_outside_the_given_window(options):
    """Rows in the input that fall outside [start, end) must never move the fit -
    the guarantee is in build_feature_matrix's own filtering, not in callers
    being careful about what they pass."""
    fixture = build_two_regime_activity(days=45, seed=1)
    split_ts = fixture.window[0] + 30 * 86400.0
    train_window = (fixture.window[0], split_ts)
    train_only = [c for c in fixture.changes if c.ts < split_ts]

    clean = hm.fit(train_only, SignalSet(), options, train_window)

    # A markedly different "holdout" - a home that never turns anything off,
    # so if it leaked in at all it would show up unmistakably.
    polluted_tail = []
    for change in fixture.changes:
        if change.ts >= split_ts:
            noisy = StateChange(
                change.entity_id, "on", change.ts, old_state="off"
            )
            noisy.cause = Cause.HUMAN
            polluted_tail.append(noisy)
    polluted = hm.fit(train_only + polluted_tail, SignalSet(), options, train_window)

    assert clean.fitted and polluted.fitted
    assert clean.n_states == polluted.n_states
    assert np.array_equal(clean.means, polluted.means)
    assert np.array_equal(clean.transition, polluted.transition)
    assert clean.train_bins == polluted.train_bins
    assert clean.log_likelihood == polluted.log_likelihood


def test_pipeline_fits_the_mode_model_on_the_training_window_only(
    ha_config_dir, store: Store, fake_client, monkeypatch
):
    """The actual wiring, not just the module's own robustness: amminer.pipeline
    must call home_mode.fit with the train window, never the full one."""
    from amminer import pipeline as pipeline_module

    calls: list[tuple[float, float]] = []
    real_fit = pipeline_module.home_mode_module.fit

    def spy(changes, signals, options, train_window, resolver=None):
        calls.append(train_window)
        return real_fit(changes, signals, options, train_window, resolver)

    monkeypatch.setattr(pipeline_module.home_mode_module, "fit", spy)

    options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir))
    from amminer.discovery.ha_config import HAConfig

    report, _candidates = pipeline_module.run_analysis(
        options, store, fake_client, HAConfig(ha_config_dir)
    )

    assert calls, "home_mode.fit was never called"
    called_window = calls[0]
    expected = backtest_module.split_window(report.window, options).train
    assert called_window == expected
    # And it must be strictly shorter than the full analysis window whenever
    # there is a real holdout - the whole point of the split.
    assert called_window[1] <= report.window[1]
    assert called_window[1] < report.window[1] or report.window[1] == report.window[0]


# --- "must never gate" ------------------------------------------------------
def test_disabling_home_mode_only_removes_a_signal_never_a_gate(ha_config_dir, fake_client):
    """Turning home_mode off must change nothing about what passes or fails -
    only the signal offered to the conditional miner, exactly like a signal
    amminer.enrich.detect never found."""
    from amminer.discovery.ha_config import HAConfig
    from amminer.pipeline import run_analysis

    on_options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir),
                          home_mode_enabled=True)
    off_options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir),
                           home_mode_enabled=False)

    with Store(":memory:") as store_on:
        report_on, candidates_on = run_analysis(on_options, store_on, fake_client,
                                                  HAConfig(ha_config_dir))
    with Store(":memory:") as store_off:
        report_off, candidates_off = run_analysis(off_options, store_off, fake_client,
                                                    HAConfig(ha_config_dir))

    assert report_off.home_mode == {}
    # home_mode may (or may not, on this particular fixture) let the
    # conditional miner surface one extra candidate that uses it - that is
    # the feature working, not it gating anything. What "never gates" means
    # here is narrower and exact: every candidate that did *not* depend on
    # the mode signal must be identical, id for id, whether or not the
    # feature ran at all - a coincidence of two same-sized but different sets
    # would not satisfy this.
    ids_on_without_mode = {
        c.id for c in candidates_on if hm.HOME_MODE_ENTITY_ID not in c.entities
    }
    ids_off = {c.id for c in candidates_off}
    assert ids_on_without_mode == ids_off


# --- a miner actually benefiting from the signal ---------------------------
def test_conditional_miner_recovers_a_habit_clock_time_cannot_explain(options):
    """The motivating example from the module's own docstring: a habit spread
    across a wide time window that only the inferred mode can explain."""
    fixture = build_winddown_habit_activity(days=60, seed=3)

    # Plain clock time: no candidate for the habit clears the bar.
    tod_candidates = time_of_day.mine(fixture.changes, options, fixture.window)
    habit_tod = [
        c for c in tod_candidates
        if c.actions and c.actions[0].entity_id == fixture.habit_entity_id
    ]
    assert not habit_tod, "time-of-day alone should not have recovered this habit"

    # The action really did happen often enough to be a real habit - the
    # miner failing above is about consistency, not about there being
    # nothing there.
    grouped = human_action_events(fixture.changes, options)
    hits = grouped.get((fixture.habit_entity_id, fixture.habit_state), [])
    assert len(hits) >= options.min_occurrences

    model = hm.fit(fixture.changes, SignalSet(), options, fixture.window)
    assert model.fitted
    signals = SignalSet(home_mode=[hm.HOME_MODE_ENTITY_ID])
    series = hm.decode_series(model, fixture.changes, signals, options, fixture.window)
    assert series is not None
    store = SignalStore()
    store.add(series)

    conditional_candidates = conditional.mine(
        fixture.changes, options, signals, store, fixture.window, resolver=None
    )
    habit_cond = [
        c for c in conditional_candidates
        if c.actions and c.actions[0].entity_id == fixture.habit_entity_id
    ]
    assert habit_cond, "conditioning on the inferred mode should have recovered this habit"
    found = habit_cond[0]
    assert hm.HOME_MODE_ENTITY_ID in found.entities
    assert found.evidence.consistency >= conditional.MIN_PURITY
    assert found.evidence.lift >= conditional.MIN_LIFT
    # The improvement this whole feature exists to deliver: conditioned on
    # the mode, this habit is comfortably more consistent than any bare
    # clock-time reading of the same occurrences could have been.
    assert found.evidence.consistency > options.min_consistency


# --- migration on a populated database --------------------------------------
def test_home_mode_table_appears_on_a_populated_v4_database(tmp_path: Path):
    """CREATE TABLE IF NOT EXISTS is a no-op on a table that already exists,
    so the new table has to be exercised against a database that predates it
    - not a fresh one, which would pass even if the migration only worked by
    accident of being the first thing ever run against that file."""
    from amminer.store import db as db_module

    path = tmp_path / "state.db"
    start = db_module._SCHEMA.index(
        "-- A single row (id fixed at 1) holding whatever household-mode"
    )
    marker = "CREATE TABLE IF NOT EXISTS home_mode_models"
    table_start = db_module._SCHEMA.index(marker, start)
    end = db_module._SCHEMA.index(");", table_start) + len(");")
    v4_schema = db_module._SCHEMA[:start] + db_module._SCHEMA[end:]
    assert "home_mode_models" not in v4_schema

    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.executescript(v4_schema)
    conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', '4')")
    conn.execute(
        "INSERT INTO suggestions(id, miner, title, summary, score, status, payload,"
        " first_seen_ts, last_seen_ts, seen_count, run_id) VALUES"
        " ('sugg1', 'time_of_day', 'A suggestion', 'summary', 0.5, 'new', '{}', 1.0, 1.0, 1, 1)"
    )
    conn.execute(
        "INSERT INTO ranking_models(id, feature_schema_version, n_labels, trained_ts,"
        " fallback_to_prior, fallback_reason, l2, bias, weights, scaler) VALUES"
        " (1, 1, 0, 1.0, 1, 'cold start', 30.0, -1.0, '{}', '{}')"
    )
    conn.commit()
    conn.close()

    store = Store(path)
    try:
        assert store.get_meta("schema_version") == str(db_module.SCHEMA_VERSION)
        # Pre-existing data survived the migration untouched.
        suggestion = store.get_suggestion("sugg1")
        assert suggestion is not None
        assert suggestion["title"] == "A suggestion"
        assert store.get_ranking_model() is not None

        # The new table exists and is immediately usable.
        assert store.get_home_mode_model() is None
        fixture = build_two_regime_activity(days=45, seed=1)
        model = hm.fit(fixture.changes, SignalSet(), Options(), fixture.window)
        store.save_home_mode_model(model.as_dict())
        row = store.get_home_mode_model()
        assert row is not None
        assert row["fitted"] is True
        assert row["n_states"] == model.n_states

        rebuilt = hm.HomeModeModel.from_row(row)
        assert rebuilt is not None
        assert rebuilt.n_states == model.n_states
        assert np.allclose(rebuilt.means, model.means)
    finally:
        store.close()


def test_home_mode_counted_in_store_counts(store: Store, options):
    fixture = build_two_regime_activity(days=45, seed=1)
    model = hm.fit(fixture.changes, SignalSet(), options, fixture.window)
    store.save_home_mode_model(model.as_dict())
    assert store.counts()["home_mode_models"] == 1
