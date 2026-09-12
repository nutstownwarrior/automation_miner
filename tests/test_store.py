"""The add-on's own SQLite store: persistence, dismissals, feedback."""

from __future__ import annotations

from amminer.store.db import STATUS_ACCEPTED, STATUS_DISMISSED, STATUS_NEW, Store


def add(store, suggestion_id="s1", miner="time_of_day", score=0.8, run_id=1):
    return store.upsert_suggestion(
        suggestion_id, miner, "Title", "Summary", score, {"actions": [{"service": "light.turn_on"}]}, run_id
    )


def test_schema_is_created_and_counts_start_empty(store):
    assert store.get_meta("schema_version") == "2"
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
        assert store.get_meta("schema_version") == "2"
        old = store.get_backtest("old-one")
        assert old["precision_score"] == 0.6
        assert old["validation"] == "in_sample"  # the new column's default
        assert old["train_days"] is None
        # And the store is fully usable afterwards, not just readable.
        store.save_backtest(
            "old-one", {"precision": 0.9, "passed": True, "validation": "holdout"}
        )
        assert store.get_backtest("old-one")["validation"] == "holdout"


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
