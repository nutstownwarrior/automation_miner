"""End-to-end pipeline behaviour, including every graceful-degradation path."""

from __future__ import annotations

import datetime as dt

import pytest
from amminer.config import Options
from amminer.discovery.ha_config import HAConfig
from amminer.pipeline import MIN_DAYS_FOR_SEQUENCE_MINING, resolve_window, run_analysis
from amminer.testing.synthetic import build_default_fixture


def run(ha_dir, store, client=None, **option_overrides):
    options = Options(ha_config_dir=str(ha_dir), state_dir=str(ha_dir), **option_overrides)
    return run_analysis(options, store, client, HAConfig(ha_dir))


def test_zero_config_run_produces_suggestions(ha_config_dir, store, fake_client):
    """The documented plug-and-play path: defaults only, and it must work."""
    report, candidates = run(ha_config_dir, store, fake_client)

    assert report.status == "ok"
    assert report.state_rows > 0
    assert report.window_days > 40
    assert report.surfaced > 0
    assert report.overrides > 0
    assert report.causality["human"] > 0

    suggestions = store.list_suggestions(status="new")
    assert suggestions
    # Every surfaced suggestion carries its evidence and its backtest.
    for suggestion in suggestions:
        payload = suggestion["payload"]
        assert payload["evidence"]
        if payload.get("actions"):
            assert payload["backtest"]["passed"] is True


def test_the_injected_habits_survive_the_whole_pipeline(ha_config_dir, store, fake_client):
    run(ha_config_dir, store, fake_client)
    titles = " | ".join(s["title"] for s in store.list_suggestions(status="new"))
    payloads = [s["payload"] for s in store.list_suggestions(status="new")]
    targets = {
        action["entity_id"]
        for payload in payloads
        for action in payload.get("actions", [])
        if action.get("entity_id")
    }
    assert "light.kitchen" in targets, f"kitchen habit lost; got: {titles}"


def test_run_is_recorded_and_repeatable(ha_config_dir, store, fake_client):
    run(ha_config_dir, store, fake_client)
    first = store.last_run()
    run(ha_config_dir, store, fake_client)
    second = store.last_run()
    assert first["id"] != second["id"]
    assert second["status"] == "ok"
    # Re-running must not duplicate suggestions - ids are stable.
    ids = [s["id"] for s in store.list_suggestions(status="new")]
    assert len(ids) == len(set(ids))
    assert store.get_suggestion(ids[0])["seen_count"] == 2


def test_dismissed_suggestions_never_come_back(ha_config_dir, store, fake_client):
    run(ha_config_dir, store, fake_client)
    victim = store.list_suggestions(status="new")[0]["id"]
    store.dismiss(victim, "no thanks", signature=victim)

    run(ha_config_dir, store, fake_client)
    assert victim not in {s["id"] for s in store.list_suggestions(status="new")}
    assert store.get_suggestion(victim)["status"] == "dismissed"


def test_overrides_are_persisted(ha_config_dir, store, fake_client):
    run(ha_config_dir, store, fake_client)
    assert store.counts()["overrides"] > 0
    assert "automation.bedtime_dim" in store.override_counts()


def test_gap_suggestions_are_produced(ha_config_dir, store, fake_client):
    run(ha_config_dir, store, fake_client)
    gaps = store.list_gaps()
    assert gaps
    kinds = {g["payload"]["kind"] for g in gaps}
    assert kinds & {"integration", "hardware", "infrastructure"}


# --- graceful degradation ----------------------------------------------
def test_missing_recorder_degrades_without_error(tmp_path, store):
    (tmp_path / "configuration.yaml").write_text("homeassistant: {}\n")
    report, candidates = run(tmp_path, store)
    assert report.status == "degraded"
    assert candidates == []
    assert any("Recorder unavailable" in d for d in report.degradations)
    assert store.last_run()["status"] == "degraded"


def test_short_history_disables_sequence_mining(tmp_path, store, fake_client):
    """The default HA OS + SQLite + 10-day-retention case."""
    (tmp_path / ".storage").mkdir(parents=True)
    (tmp_path / "configuration.yaml").write_text("recorder:\n  purge_keep_days: 10\n")
    build_default_fixture(tmp_path / "home-assistant_v2.db", days=5, seed=7)

    report, _candidates = run(tmp_path, store, fake_client)
    assert report.status == "ok"
    assert report.window_days < MIN_DAYS_FOR_SEQUENCE_MINING
    assert report.miner_counts["association"] == 0
    assert report.miner_counts["sequence"] == 0
    assert any("association and sequence mining are disabled" in d for d in report.degradations)
    assert any("MariaDB" in d for d in report.degradations)


def test_no_registry_and_no_api_still_mines(ha_config_dir, store):
    """Worst case: only the recorder is readable."""
    report, _candidates = run(ha_config_dir, store, client=None)
    assert report.status == "ok"
    assert report.state_rows > 0
    assert any("No entities could be resolved" in d for d in report.degradations)
    # The recorder seeds the entity set so mining still has names to work with.
    assert report.entities["total"] > 0


def test_missing_user_context_is_flagged(tmp_path, store, fake_client, monkeypatch):
    (tmp_path / ".storage").mkdir(parents=True)
    (tmp_path / "configuration.yaml").write_text("recorder:\n  purge_keep_days: 60\n")
    build_default_fixture(tmp_path / "home-assistant_v2.db", days=20, seed=11)

    import amminer.recorderdb.causality as causality_module

    original = causality_module.build_index

    def stripped(changes, events=()):
        for change in changes:
            change.context_user_id = None
        index = original(changes, [])
        index.has_user_context = False
        return index

    monkeypatch.setattr(causality_module, "build_index", stripped)
    report, _candidates = run(tmp_path, store, fake_client)
    assert any("context_user_id" in d for d in report.degradations)


def test_no_external_signals_is_reported(ha_config_dir, store):
    report, _candidates = run(ha_config_dir, store, client=None)
    # The recorder-seeded entity set has person/climate but no weather or price.
    assert "energy_price" not in report.signals
    assert report.miner_counts["energy_shift"] == 0


def test_one_failing_miner_does_not_kill_the_run(ha_config_dir, store, fake_client, monkeypatch):
    """A miner raising must cost only its own findings.

    This is the shape of a real failure: a dependency changed signature under a
    loose pin and one miner started raising. Every other miner still has useful
    output, so the run must complete and say what was lost.
    """
    import amminer.miners.association as association_module

    def explode(*args, **kwargs):
        raise TypeError("association_rules() missing 1 required positional argument")

    monkeypatch.setattr(association_module, "mine", explode)
    report, candidates = run(ha_config_dir, store, fake_client)

    assert report.status == "partial"
    assert report.error is None
    assert "association" in report.miner_errors
    assert "TypeError" in report.miner_errors["association"]
    assert any("association miner failed" in d for d in report.degradations)
    # The rest of the pipeline still ran and produced results.
    assert report.miner_counts["time_of_day"] > 0
    assert report.surfaced > 0
    assert store.list_suggestions(status="new")


def test_every_miner_failing_still_completes(ha_config_dir, store, fake_client, monkeypatch):
    import amminer.pipeline as pipeline_module

    for name in ("time_of_day", "conditional", "motif", "energy", "association",
                 "sequence", "stale"):
        module = getattr(pipeline_module, name)
        monkeypatch.setattr(module, "mine", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))

    report, _candidates = run(ha_config_dir, store, fake_client)
    assert report.status == "partial"
    assert report.error is None
    assert len(report.miner_errors) >= 5
    assert report.surfaced == 0


def test_pipeline_error_is_captured_not_raised(ha_config_dir, store, monkeypatch):
    import amminer.pipeline as pipeline_module

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_module, "build_resolver", explode)
    report, candidates = run(ha_config_dir, store)
    assert report.status == "error"
    assert "boom" in report.error
    assert candidates == []
    assert store.last_run()["status"] == "error"


# --- window resolution --------------------------------------------------
class _Recorder:
    def __init__(self, oldest, newest):
        self.info = type(
            "Info", (), {"oldest_state_ts": oldest, "newest_state_ts": newest}
        )()


def test_auto_window_is_capped_at_60_days():
    now = dt.datetime(2024, 6, 1).timestamp()
    recorder = _Recorder(now - 200 * 86400, now)
    start, end, notes = resolve_window(Options(), recorder)
    assert (end - start) / 86400 == pytest.approx(60, abs=0.1)
    assert any("auto-selected" in n for n in notes)


def test_auto_window_uses_all_history_when_short():
    now = dt.datetime(2024, 6, 1).timestamp()
    recorder = _Recorder(now - 9 * 86400, now)
    start, end, _notes = resolve_window(Options(), recorder)
    assert (end - start) / 86400 == pytest.approx(9, abs=0.1)


def test_explicit_window_is_clamped_to_available_history():
    now = dt.datetime(2024, 6, 1).timestamp()
    recorder = _Recorder(now - 9 * 86400, now)
    start, end, notes = resolve_window(Options(analysis_window_days=90), recorder)
    assert (end - start) / 86400 == pytest.approx(9, abs=0.1)
    assert any("only" in n for n in notes)
