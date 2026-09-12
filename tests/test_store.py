"""The add-on's own SQLite store: persistence, dismissals, feedback."""

from __future__ import annotations

from amminer.store.db import (
    STATUS_ACCEPTED,
    STATUS_DISMISSED,
    STATUS_NEW,
    STATUS_SHADOW,
    STATUS_SUPPRESSED,
    Store,
)


def add(store, suggestion_id="s1", miner="time_of_day", score=0.8, run_id=1):
    return store.upsert_suggestion(
        suggestion_id, miner, "Title", "Summary", score, {"actions": [{"service": "light.turn_on"}]}, run_id
    )


def test_schema_is_created_and_counts_start_empty(store):
    assert store.get_meta("schema_version") == "5"
    assert all(count == 0 for count in store.counts().values())


def test_upsert_then_read_back(store):
    assert add(store) == STATUS_NEW
    stored = store.get_suggestion("s1")
    assert stored["title"] == "Title"
    assert stored["payload"]["actions"][0]["service"] == "light.turn_on"
    assert stored["seen_count"] == 1


def test_re_seeing_increments_without_resetting_status(store):
    add(store)
    store.set_status("s1", STATUS_ACCEPTED)
    add(store, score=0.9)
    stored = store.get_suggestion("s1")
    assert stored["seen_count"] == 2
    assert stored["score"] == 0.9
    assert stored["status"] == STATUS_ACCEPTED


def test_dismissal_is_sticky_across_runs(store):
    add(store)
    store.dismiss("s1", "not useful", signature="s1")
    assert store.is_dismissed("s1") is True
    # A later run rediscovers the same rule; it must stay dismissed.
    assert add(store, run_id=2) == STATUS_DISMISSED
    assert store.get_suggestion("s1")["status"] == STATUS_DISMISSED
    assert store.list_suggestions(status=STATUS_NEW) == []


def test_dismissed_signature_blocks_a_renamed_duplicate(store):
    add(store)
    store.dismiss("s1", signature="sig-abc")
    assert store.is_dismissed("other", signature="sig-abc") is True
    assert "sig-abc" in store.dismissed_signatures()


def test_dismissal_records_feedback(store):
    add(store)
    store.dismiss("s1", "too noisy")
    kinds = [f["kind"] for f in store.feedback_for("s1")]
    assert "dismissed" in kinds
    assert store.feedback_for("s1")[0]["payload"]["reason"] == "too noisy"


def test_prune_removes_stale_new_suggestions_only(store):
    add(store, "old", run_id=1)
    add(store, "kept", run_id=2)
    add(store, "accepted", run_id=1)
    store.set_status("accepted", STATUS_ACCEPTED)
    removed = store.prune_suggestions(run_id=2)
    assert removed == 1
    assert store.get_suggestion("old") is None
    assert store.get_suggestion("kept") is not None
    assert store.get_suggestion("accepted") is not None


def test_backtest_round_trip(store):
    add(store)
    store.save_backtest("s1", {"precision": 0.9, "recall": 0.8, "true_fires": 9,
                               "false_fires": 1, "missed": 2, "false_fires_per_week": 0.5,
                               "passed": True, "summary": "good"})
    stored = store.get_backtest("s1")
    assert stored["precision_score"] == 0.9
    assert stored["passed"] == 1
    assert stored["payload"]["summary"] == "good"
    # No validation was recorded: a plain backtest, not a holdout-checked one.
    assert stored["validation"] == "in_sample"
    assert stored["train_days"] is None
    # It is also attached when the suggestion is read.
    assert store.get_suggestion("s1")["backtest"]["precision_score"] == 0.9


def test_backtest_round_trip_carries_holdout_validation(store):
    add(store)
    store.save_backtest("s1", {"precision": 0.95, "recall": 0.8, "true_fires": 7,
                               "false_fires": 0, "missed": 1, "false_fires_per_week": 0.0,
                               "passed": True, "summary": "good", "validation": "holdout",
                               "train_days": 33.8, "holdout_days": 11.2})
    stored = store.get_backtest("s1")
    assert stored["validation"] == "holdout"
    assert stored["train_days"] == 33.8
    assert stored["holdout_days"] == 11.2


def test_an_old_database_migrates_the_backtest_validation_columns(tmp_path):
    """A database from before this feature existed has no holdout columns at
    all; opening it must add them rather than fail."""
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key, value) VALUES ('schema_version', '1');
        CREATE TABLE backtests (
            suggestion_id        TEXT PRIMARY KEY,
            ts                   REAL NOT NULL,
            precision_score      REAL,
            recall_score         REAL,
            true_fires           INTEGER,
            false_fires          INTEGER,
            missed               INTEGER,
            false_fires_per_week REAL,
            passed               INTEGER NOT NULL DEFAULT 0,
            payload              TEXT
        );
        INSERT INTO backtests(suggestion_id, ts, precision_score, passed, payload)
        VALUES ('old-one', 0.0, 0.6, 1, '{}');
        """
    )
    conn.commit()
    conn.close()

    with Store(path) as store:
        assert store.get_meta("schema_version") == "5"
        old = store.get_backtest("old-one")
        assert old["precision_score"] == 0.6
        assert old["validation"] == "in_sample"  # the new column's default
        assert old["train_days"] is None
        # And the store is fully usable afterwards, not just readable.
        store.save_backtest(
            "old-one", {"precision": 0.9, "passed": True, "validation": "holdout"}
        )
        assert store.get_backtest("old-one")["validation"] == "holdout"


def test_an_old_ranking_models_table_migrates_and_ranking_still_works(tmp_path):
    """This project's own branch briefly shipped ``ranking_models`` with a
    ``prior_k`` column and no ``scaler`` - a blend fraction, before the fit
    became a MAP estimate towards the prior.  ``CREATE TABLE IF NOT EXISTS``
    does nothing to a database that already has the table in that shape, so
    opening one must add what is missing rather than fail the first time
    anything tries to write or read a model - and the row that shape left
    behind must not be handed back as if it still meant something, since its
    weights are in a since-abandoned space this version would misinterpret.
    """
    import sqlite3

    from amminer.learn import ranking

    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key, value) VALUES ('schema_version', '3');
        CREATE TABLE ranking_models (
            id                     INTEGER PRIMARY KEY CHECK (id = 1),
            feature_schema_version INTEGER NOT NULL,
            n_labels               INTEGER NOT NULL,
            trained_ts             REAL NOT NULL,
            fallback_to_prior      INTEGER NOT NULL DEFAULT 1,
            fallback_reason        TEXT,
            prior_k                INTEGER NOT NULL,
            bias                   REAL NOT NULL,
            weights                TEXT NOT NULL
        );
        INSERT INTO ranking_models(id, feature_schema_version, n_labels, trained_ts,
            fallback_to_prior, fallback_reason, prior_k, bias, weights)
        VALUES (1, 1, 5, 0.0, 1, 'cold start', 20, -1.0, '{}');
        """
    )
    conn.commit()
    conn.close()

    with Store(path) as store:
        # The old row is discarded outright, not silently reinterpreted.
        assert store.get_ranking_model() is None

        # Not merely "the ALTER succeeded" - the whole path a real run takes
        # (train, persist, read back, score with it) works on this database,
        # which is exactly what a stray OperationalError on the missing `l2`
        # column, before this fix, took down permanently and silently.
        model = ranking.train_from_labels([])
        store.save_ranking_model(model.as_dict())
        row = store.get_ranking_model()
        assert row is not None
        restored = ranking.RankingModel.from_row(row)
        assert restored is not None
        assert restored.fallback_to_prior is True
        assert 0.0 <= restored.probability(_bare_candidate()) <= 1.0


def _bare_candidate():
    from amminer.miners.base import Action, Candidate, Evidence

    return Candidate(
        miner="time_of_day",
        title="t",
        actions=[Action(service="light.turn_on", entity_id="light.x")],
        evidence=Evidence(),
    )


def test_overrides_are_deduplicated(store):
    from amminer.recorderdb.models import OverrideEvent

    events = [
        OverrideEvent("light.a", 100.0, "automation.x", "off", "on", 20.0),
        OverrideEvent("light.a", 100.0, "automation.x", "off", "on", 20.0),  # duplicate
        OverrideEvent("light.b", 200.0, "automation.x", "on", "off", 30.0),
    ]
    store.record_overrides(events)
    store.record_overrides(events)  # a second run over the same window
    assert store.counts()["overrides"] == 2
    assert store.override_counts() == {"automation.x": 2}
    assert len(store.overrides_for_automation("automation.x")) == 2


def test_shadow_report(store):
    add(store)
    store.log_shadow_fire("s1", 100.0, matched=True)
    store.log_shadow_fire("s1", 200.0, matched=True)
    store.log_shadow_fire("s1", 300.0, matched=False)
    report = store.shadow_report("s1")
    assert report == {"fires": 3, "matched": 2, "unmatched": 1, "precision": 2 / 3}


def test_runs_are_tracked(store):
    run_id = store.start_run()
    store.finish_run(run_id, "ok", {"surfaced": 4})
    last = store.last_run()
    assert last["status"] == "ok"
    assert last["stats"]["surfaced"] == 4
    assert len(store.recent_runs()) == 1


def test_gap_suggestions_round_trip(store):
    store.upsert_gap("g1", "hardware", "Add mmWave", {"gap": "no presence"})
    assert len(store.list_gaps("new")) == 1
    store.set_gap_status("g1", "dismissed")
    assert store.list_gaps("new") == []
    assert store.list_gaps()[0]["payload"]["gap"] == "no presence"


def test_listing_filters_and_orders(store):
    add(store, "a", miner="time_of_day", score=0.2)
    add(store, "b", miner="association", score=0.9)
    ordered = [s["id"] for s in store.list_suggestions()]
    assert ordered == ["b", "a"]
    assert [s["id"] for s in store.list_suggestions(miner="association")] == ["b"]


# --- applied automations & their health (amminer.health) ------------------
def test_applied_automation_round_trips_with_no_health_yet(store):
    store.record_applied_automation(
        "amminer_1", "sugg1", "Kitchen light",
        {"triggers": [], "actions": [{"service": "light.turn_on"}]},
        {"id": "amminer_1", "trigger": [], "action": []},
    )
    applied = store.get_applied_automation("amminer_1")
    assert applied["suggestion_id"] == "sugg1"
    assert applied["title"] == "Kitchen light"
    assert applied["candidate_payload"]["actions"][0]["service"] == "light.turn_on"
    # No run has judged it yet - never a fabricated verdict standing in.
    assert applied["health"] is None
    assert [a["automation_id"] for a in store.list_applied_automations()] == ["amminer_1"]


def test_re_applying_the_same_automation_refreshes_the_snapshot(store):
    store.record_applied_automation("amminer_1", "sugg1", "v1", {"a": 1}, {"id": "amminer_1"})
    store.record_applied_automation("amminer_1", "sugg1", "v2", {"a": 2}, {"id": "amminer_1"})
    rows = store.list_applied_automations()
    assert len(rows) == 1
    assert rows[0]["title"] == "v2"
    assert rows[0]["candidate_payload"]["a"] == 2


def test_automation_health_round_trip(store):
    store.record_applied_automation("amminer_1", "sugg1", "Kitchen light", {}, {})
    store.save_automation_health(
        "amminer_1", "active", "healthy", {"verdict": "healthy", "actual_fires": 9}
    )
    applied = store.get_applied_automation("amminer_1")
    assert applied["health"]["status"] == "active"
    assert applied["health"]["verdict"] == "healthy"
    assert applied["health"]["payload"]["actual_fires"] == 9

    # A later run overwrites it wholesale, the same way ranking_models does.
    store.save_automation_health(
        "amminer_1", "active", "dormant", {"verdict": "dormant", "actual_fires": 0}
    )
    applied = store.get_applied_automation("amminer_1")
    assert applied["health"]["verdict"] == "dormant"
    assert store.counts()["applied_automations"] == 1
    assert store.counts()["automation_health"] == 1


def test_an_old_database_gains_the_applied_automations_tables(tmp_path):
    """A v3 database (this branch's earlier schema) has never heard of
    applied_automations or automation_health.  Opening it must add both
    without disturbing the data already in it - CREATE TABLE IF NOT EXISTS is
    a genuine no-op here only because these are brand-new tables; the
    existing ones must come through untouched."""
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key, value) VALUES ('schema_version', '3');
        CREATE TABLE suggestions (
            id TEXT PRIMARY KEY, miner TEXT NOT NULL, title TEXT NOT NULL,
            summary TEXT, score REAL NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'new',
            payload TEXT NOT NULL, first_seen_ts REAL NOT NULL, last_seen_ts REAL NOT NULL,
            seen_count INTEGER NOT NULL DEFAULT 1, run_id INTEGER
        );
        INSERT INTO suggestions(id, miner, title, summary, score, status, payload,
            first_seen_ts, last_seen_ts)
        VALUES ('old-sugg', 'time_of_day', 'Old suggestion', 'summary', 0.5, 'new', '{}', 0, 0);
        """
    )
    conn.commit()
    conn.close()

    with Store(path) as store:
        assert store.get_meta("schema_version") == "5"
        # Pre-existing data survived the migration untouched.
        assert store.get_suggestion("old-sugg")["title"] == "Old suggestion"
        # And the new tables are not just present but fully usable end to end,
        # on this same, previously-v3, database file - not only on a fresh one.
        assert store.list_applied_automations() == []
        store.record_applied_automation(
            "amminer_1", "old-sugg", "Old suggestion", {"a": 1}, {"id": "amminer_1"}
        )
        store.save_automation_health("amminer_1", "active", "healthy", {"verdict": "healthy"})
        applied = store.get_applied_automation("amminer_1")
        assert applied["health"]["verdict"] == "healthy"


def test_store_survives_reopen(tmp_path):
    path = tmp_path / "state.db"
    with Store(path) as store:
        add(store)
        store.dismiss("s1")
    with Store(path) as store:
        assert store.is_dismissed("s1") is True
        assert store.get_suggestion("s1")["status"] == STATUS_DISMISSED


def test_restore_undoes_the_dismissal_the_next_run_reads(store):
    """Flipping the status back is not a restore if mining still filters it out."""
    store.upsert_suggestion("x1", "time_of_day", "T", "s", 0.9, {"actions": [1]}, 1)
    store.dismiss("x1", "not useful", signature="sig-1")
    assert store.is_dismissed("x1") is True

    assert store.restore("x1") is True
    assert store.get_suggestion("x1")["status"] == "new"
    # The next analysis run consults these, not the status column.
    assert store.is_dismissed("x1") is False
    assert "x1" not in store.dismissed_ids()
    assert "sig-1" not in store.dismissed_signatures()


def test_restoring_an_unknown_suggestion_reports_failure(store):
    assert store.restore("never-existed") is False


# --- ranking labels (amminer.learn.ranking) --------------------------
def test_ranking_labels_excludes_suppressed_and_shadow_suggestions(store):
    for suggestion_id, status in (
        ("accepted1", STATUS_ACCEPTED),
        ("dismissed1", STATUS_DISMISSED),
        ("suppressed1", STATUS_SUPPRESSED),
        ("shadow1", STATUS_SHADOW),
    ):
        add(store, suggestion_id)
        store.set_status(suggestion_id, status)

    # Only accept()/dismiss() record the decision-time snapshot ranking needs;
    # a bare set_status (as used above for suppressed/shadow, which have no
    # "decision" of their own) leaves nothing for ranking_labels to find, so
    # accepted1/dismissed1 need their own real calls to produce one.
    store.upsert_suggestion("accepted2", "time_of_day", "T", "s", 0.9, {"actions": [1]}, 1)
    store.accept("accepted2")
    store.upsert_suggestion("dismissed2", "association", "T", "s", 0.9, {"actions": [1]}, 1)
    store.dismiss("dismissed2", "not useful")

    labels = {row["id"]: row["status"] for row in store.ranking_labels()}
    assert labels == {"accepted2": STATUS_ACCEPTED, "dismissed2": STATUS_DISMISSED}


def test_ranking_labels_use_the_payload_as_of_the_decision_not_the_live_row(store):
    store.upsert_suggestion(
        "s1", "time_of_day", "T", "s", 0.5, {"score": 0.5, "evidence": {"occurrences": 1}}, 1
    )
    store.dismiss("s1", "meh")
    # A later run re-mines the same rule with very different numbers - as
    # would happen to an *accepted* rule kept being re-mined every night.
    store.upsert_suggestion(
        "s1", "time_of_day", "T", "s", 0.99, {"score": 0.99, "evidence": {"occurrences": 999}}, 2
    )

    labels = store.ranking_labels()
    assert len(labels) == 1
    assert labels[0]["payload"]["evidence"]["occurrences"] == 1


def test_ranking_labels_records_seen_count_as_of_the_decision(store):
    store.upsert_suggestion("s1", "time_of_day", "T", "s", 0.5, {"score": 0.5}, 1)
    store.upsert_suggestion("s1", "time_of_day", "T", "s", 0.5, {"score": 0.5}, 2)  # seen_count -> 2
    store.dismiss("s1", "meh")
    store.upsert_suggestion("s1", "time_of_day", "T", "s", 0.5, {"score": 0.5}, 3)  # after the decision

    labels = store.ranking_labels()
    assert labels[0]["seen_count_at_decision"] == 2


def test_seen_counts_reads_the_live_row(store):
    add(store, "a")
    add(store, "a")  # seen_count -> 2
    add(store, "b")
    assert store.seen_counts(["a", "b", "missing"]) == {"a": 2, "b": 1}
    assert store.seen_counts([]) == {}


def test_ranking_model_round_trips(store):
    assert store.get_ranking_model() is None
    model = {
        "feature_schema_version": 1,
        "weights": {"consistency": 1.2},
        "bias": -1.0,
        "scaler": {"center": {"consistency": 0.5}, "scale": {"consistency": 0.2}},
        "n_labels": 10,
        "l2": 15.0,
        "trained_ts": 12345.0,
        "fallback_to_prior": False,
        "fallback_reason": None,
    }
    store.save_ranking_model(model)
    stored = store.get_ranking_model()
    assert stored["n_labels"] == 10
    assert stored["fallback_to_prior"] is False
    assert stored["weights"] == {"consistency": 1.2}
    assert stored["scaler"] == {"center": {"consistency": 0.5}, "scale": {"consistency": 0.2}}

    # Overwritten wholesale on the next run, not appended.
    model["n_labels"] = 20
    model["fallback_to_prior"] = True
    model["fallback_reason"] = "cold start"
    store.save_ranking_model(model)
    stored = store.get_ranking_model()
    assert stored["n_labels"] == 20
    assert stored["fallback_to_prior"] is True


def test_accept_probability_orders_suggestions_ahead_of_score(store):
    store.upsert_suggestion("low", "time_of_day", "T", "s", 0.9, {}, 1, accept_probability=0.1)
    store.upsert_suggestion("high", "time_of_day", "T", "s", 0.1, {}, 1, accept_probability=0.9)
    ordered = [s["id"] for s in store.list_suggestions(status=STATUS_NEW)]
    assert ordered == ["high", "low"]


def test_without_accept_probability_ordering_falls_back_to_score(store):
    store.upsert_suggestion("low_score", "time_of_day", "T", "s", 0.1, {}, 1)
    store.upsert_suggestion("high_score", "time_of_day", "T", "s", 0.9, {}, 1)
    ordered = [s["id"] for s in store.list_suggestions(status=STATUS_NEW)]
    assert ordered == ["high_score", "low_score"]
