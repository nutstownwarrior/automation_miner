"""End-to-end pipeline behaviour, including every graceful-degradation path."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from amminer.config import Options
from amminer.discovery.ha_config import HAConfig
from amminer.pipeline import MIN_DAYS_FOR_SEQUENCE_MINING, resolve_window, run_analysis
from amminer.store import Store
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


# --- optional AI features -----------------------------------------------
class _Stub:
    """A provider stub that answers whichever AI feature is asking."""

    name = "stub"
    enabled = True

    def __init__(self, raises: bool = False):
        self.raises = raises
        self.calls = 0
        self.prompts: list[str] = []

    def status(self):
        from amminer.llm.provider import LLMStatus

        return LLMStatus("stub", True, "http://stub", "stub-model")

    def complete_json(self, system, user):
        from amminer.llm.provider import LLMError

        self.calls += 1
        self.prompts.append(user)
        if self.raises:
            raise LLMError("stub is down")
        if "label Home Assistant entities" in system:
            return {"assignments": [
                {"entity_id": "sensor.outdoor_temperature",
                 "roles": ["outdoor_temperature"]},
            ]}
        if "explain WHEN" in system:
            return {"hypotheses": [
                {"reason": "it tracks the outdoor temperature",
                 "conditions": [{"kind": "numeric_state",
                                 "entity_id": "sensor.outdoor_temperature",
                                 "below": 8}]},
            ]}
        if "make SENSE" in system:
            payload = json.loads(user)
            return {"reviews": [
                {"id": rule["id"], "verdict": "plausible", "reason": "fine"}
                for rule in payload["rules"]
            ]}
        if "standing preferences" in system and "generalise" in system:
            payload = json.loads(user)
            ids = [d["id"] for d in payload["dismissals"]][:2]
            return {"preferences": [
                {"rule": "Never automate the hallway.", "from": ids},
            ]}
        if "apply a person's standing preferences" in system:
            payload = json.loads(user)
            preference = payload["preferences"][0]["id"]
            return {"matches": [
                {"id": payload["suggestions"][0]["id"], "preference": preference,
                 "reason": "This is the hallway."},
            ]}
        if "one plain\nsentence" in system:
            payload = json.loads(user)
            return {"explanations": [
                {"id": s["id"], "text": "You have done this most days."}
                for s in payload["suggestions"]
            ]}
        if "really ONE routine" in system:
            return {"scenes": []}
        if "assign home-automation entities" in system:
            payload = json.loads(user)
            if not payload["areas"] or not payload["entities"]:
                return {"placements": []}
            return {"placements": [
                {"entity_id": payload["entities"][0]["entity_id"],
                 "area": payload["areas"][0], "reason": "the id says so"},
            ]}
        return {}


def _use_stub(monkeypatch, stub):
    import amminer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "build_provider", lambda options: stub)
    return stub


def test_ai_features_are_off_by_default(ha_config_dir, store, fake_client):
    report, _candidates = run(ha_config_dir, store, fake_client)
    assert report.ai == {}
    assert all(v is False for v in Options().ai_features_requested.values())


def test_disabled_ai_changes_nothing(ha_config_dir, store, fake_client, tmp_path, monkeypatch):
    """The assisted run with every feature off must equal the plain run."""
    stub = _use_stub(monkeypatch, _Stub())
    baseline, _ = run(ha_config_dir, store, fake_client)

    other = Store(tmp_path / "second.db")
    options = Options(
        ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir),
        llm_provider="ollama",  # a provider IS configured, the features are not
    )
    second, _ = run_analysis(options, other, fake_client, HAConfig(ha_config_dir))

    assert stub.calls == 0, "no AI call may happen with the features switched off"
    assert second.surfaced == baseline.surfaced
    assert second.rejected == baseline.rejected
    assert second.ai == {}
    other.close()


def test_enabled_without_a_provider_degrades_clearly(ha_config_dir, store, fake_client):
    report, _candidates = run(
        ha_config_dir, store, fake_client,
        llm_entity_classification=True, llm_hypotheses=True, llm_triage=True,
    )
    assert report.status == "ok"
    assert any("llm_provider is 'none'" in d for d in report.degradations)
    assert all(info["ran"] is False for info in report.ai.values())
    assert set(report.ai) == {"entity_classification", "hypotheses", "triage"}


def test_all_three_features_run_and_are_reported(ha_config_dir, store, fake_client, monkeypatch):
    stub = _use_stub(monkeypatch, _Stub())
    report, _candidates = run(
        ha_config_dir, store, fake_client, llm_provider="ollama",
        llm_entity_classification=True, llm_hypotheses=True, llm_triage=True,
    )
    assert report.status == "ok"
    assert stub.calls > 0
    for feature in ("entity_classification", "hypotheses", "triage"):
        assert report.ai[feature]["ran"] is True, feature
    assert report.ai["triage"]["reviewed"] > 0
    assert report.ai["hypotheses"]["considered"] > 0


def test_a_verified_hypothesis_is_surfaced_and_supersedes_its_origin(
    ha_config_dir, store, fake_client, monkeypatch
):
    """Wiring test: whether a rescue is possible depends on the data, so the
    rescue itself is stubbed here and exercised for real in test_llm_assist.py.
    What must hold regardless is that an accepted hypothesis is persisted with
    its provenance and that its origin stops being counted as a rejection."""
    import amminer.pipeline as pipeline_module
    from amminer.llm.hypothesis import HypothesisResult
    from amminer.miners.base import Action, Candidate, Trigger

    _use_stub(monkeypatch, _Stub())
    captured: dict[str, object] = {}

    def fake_propose(rejected, changes, signal_store, options, window, provider,
                     resolver=None, overrides=()):
        assert rejected, "the pipeline must hand over the rejected candidates"
        origin = rejected[0]
        captured["origin_id"] = origin.id
        rescued = Candidate(
            miner=f"{origin.miner}+hypothesis",
            title=origin.title,
            triggers=list(origin.triggers) or [Trigger(kind="time", at="06:30:00")],
            actions=list(origin.actions)
            or [Action(service="light.turn_on", entity_id="light.kitchen")],
        )
        rescued.backtest = {"passed": True, "precision": 0.95, "summary": "verified"}
        rescued.extra["hypothesis"] = {
            "reason": "it only happens on cold days",
            "origin_candidate": origin.id,
        }
        result = HypothesisResult(considered=1, proposed=1)
        result.accepted = [rescued]
        return result

    monkeypatch.setattr(pipeline_module.llm_hypothesis, "propose_and_verify", fake_propose)
    report, _candidates = run(
        ha_config_dir, store, fake_client, llm_provider="ollama", llm_hypotheses=True,
    )

    assisted = [
        s for s in store.list_suggestions(status="new")
        if (s["payload"].get("extra") or {}).get("hypothesis")
    ]
    assert assisted, "an accepted hypothesis must be persisted"
    payload = assisted[0]["payload"]
    assert payload["miner"].endswith("+hypothesis")
    assert payload["backtest"]["passed"] is True
    assert payload["extra"]["hypothesis"]["reason"] == "it only happens on cold days"
    assert report.ai["hypotheses"]["accepted"] == 1
    # The rescued rule replaces its origin rather than sitting alongside it.
    assert captured["origin_id"] not in {s["id"] for s in store.list_suggestions()}


def test_a_failing_ai_feature_does_not_break_the_run(
    ha_config_dir, store, fake_client, monkeypatch
):
    _use_stub(monkeypatch, _Stub(raises=True))
    report, _candidates = run(
        ha_config_dir, store, fake_client, llm_provider="ollama",
        llm_entity_classification=True, llm_hypotheses=True, llm_triage=True,
    )
    # A provider that is simply down is handled inside each feature and shows
    # up as a degradation, not as a broken run.
    assert report.status == "ok"
    assert report.error is None
    assert report.ai_errors == {}
    assert report.surfaced > 0, "the deterministic miners must still deliver"
    assert store.list_suggestions(status="new")


def test_an_ai_feature_that_crashes_makes_the_run_partial(
    ha_config_dir, store, fake_client, monkeypatch
):
    """A feature raising is not the same as a provider being down."""
    import amminer.llm.triage as llm_triage

    def explode(*args, **kwargs):
        raise RuntimeError("a bug in the triage feature itself")

    monkeypatch.setattr(llm_triage, "triage", explode)
    _use_stub(monkeypatch, _Stub())
    report, _candidates = run(
        ha_config_dir, store, fake_client, llm_provider="ollama", llm_triage=True,
    )
    assert report.status == "partial"
    assert "triage" in report.ai_errors
    # "partial" must still mean the rest of the run delivered.
    assert report.surfaced > 0
    assert store.list_suggestions(status="new")


def test_triage_cannot_push_a_rule_past_the_backtest_gate(
    ha_config_dir, store, fake_client, monkeypatch
):
    """Even a model that loves everything cannot surface a rejected rule."""

    class Enthusiast(_Stub):
        def complete_json(self, system, user):
            if "make SENSE" in system:
                payload = json.loads(user)
                return {"reviews": [
                    {"id": rule["id"], "verdict": "plausible", "reason": "superb"}
                    for rule in payload["rules"]
                ]}
            return {}

    _use_stub(monkeypatch, Enthusiast())
    baseline, _ = run(ha_config_dir, store, fake_client)
    baseline_surfaced = baseline.surfaced

    other_store = Store(ha_config_dir / "triage.db")
    options = Options(
        ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir),
        llm_provider="ollama", llm_triage=True,
    )
    report, _ = run_analysis(options, other_store, fake_client, HAConfig(ha_config_dir))
    assert report.surfaced == baseline_surfaced
    other_store.close()


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


def test_a_failing_stage_does_not_discard_what_the_miners_produced(
    ha_config_dir, store, fake_client, monkeypatch
):
    """Only the miners were isolated; everything after them was all-or-nothing."""
    import amminer.pipeline as pipeline_module

    def explode(*args, **kwargs):
        raise MemoryError("simulated OOM inside gap analysis")

    monkeypatch.setattr(pipeline_module.gap_analysis, "suggest", explode)
    report, _candidates = run(ha_config_dir, store, fake_client)

    assert report.status == "partial"
    assert report.error is None
    assert "gap analysis" in report.miner_errors
    # The candidates that were already mined and backtested are still here.
    assert report.surfaced > 0
    assert store.list_suggestions(status="new")


def test_a_failing_conflict_check_says_so_rather_than_implying_clean(
    ha_config_dir, store, fake_client, monkeypatch
):
    import amminer.pipeline as pipeline_module

    def explode(*args, **kwargs):
        raise RuntimeError("conflict checking blew up")

    monkeypatch.setattr(pipeline_module.conflict_checks, "annotate_candidates", explode)
    report, _candidates = run(ha_config_dir, store, fake_client)

    assert report.status == "partial"
    assert report.surfaced > 0
    assert any("not been compared" in d for d in report.degradations)
    assert store.list_suggestions(status="new")


def test_an_unfinished_run_is_closed_on_the_next_start(store):
    """A Supervisor restart mid-analysis left the row saying 'running' forever."""
    run_id = store.start_run()
    assert store.last_run()["status"] == "running"
    assert store.close_interrupted_runs() == 1
    closed = store.last_run()
    assert closed["id"] == run_id
    assert closed["status"] == "interrupted"
    assert closed["finished_ts"] is not None
    # Nothing left to close the second time.
    assert store.close_interrupted_runs() == 0


# --- the four later AI features, end to end -----------------------------
def _dismiss_housekeeping(store):
    """Leave the preference learner something to read.

    These are dismissals of suggestions from an earlier era of the database,
    which is what makes them usable here: dismissing one of *this* run's
    suggestions would also remove it from the next run, and the fixture has few
    enough candidates that doing it three times leaves nothing to group.
    """
    for index in range(3):
        store.dismiss(f"gone-{index}", "nothing in the hallway, please")
def test_all_four_later_features_run_and_are_reported(
    ha_config_dir, store, fake_client, monkeypatch
):
    _use_stub(monkeypatch, _Stub())
    # Dismissal reasons are what preferences are learned from, so there have
    # to be some before the feature has anything to read.
    run(ha_config_dir, store, fake_client)
    _dismiss_housekeeping(store)

    report, _candidates = run(
        ha_config_dir, store, fake_client, llm_provider="ollama",
        llm_preferences=True, llm_explain=True, llm_scenes=True, llm_areas=True,
    )
    assert report.status == "ok"
    assert set(report.ai) >= {"preferences", "explanations", "scenes", "area_inference"}
    assert all(info["ran"] for info in report.ai.values())


def test_a_suppressed_suggestion_is_stored_hidden_and_not_counted(
    ha_config_dir, store, fake_client, monkeypatch
):
    from amminer.store import STATUS_SUPPRESSED

    _use_stub(monkeypatch, _Stub())
    run(ha_config_dir, store, fake_client)
    _dismiss_housekeeping(store)

    report, _candidates = run(
        ha_config_dir, store, fake_client, llm_provider="ollama", llm_preferences=True,
    )
    hidden = store.list_suggestions(status=STATUS_SUPPRESSED)
    assert report.suppressed == 1
    assert len(hidden) == 1
    # Hidden, but accountable: the rule that hid it travels with it.
    by = hidden[0]["payload"]["extra"]["suppressed_by"]
    assert by["rule"] == "Never automate the hallway."
    # And "surfaced" counts what the user can actually see.
    assert report.surfaced == len(store.list_suggestions(status="new")) - len(
        [s for s in store.list_suggestions(status="new")
         if s["miner"] in ("stale_automation", "unused_entity")]
    )


def test_a_preference_never_overrules_a_decision_the_user_made(
    ha_config_dir, store, fake_client, monkeypatch
):
    """Accepted, dismissed and shadow-tested are the user's calls, not a model's."""
    from amminer.store import STATUS_SHADOW

    _use_stub(monkeypatch, _Stub())
    run(ha_config_dir, store, fake_client)
    _dismiss_housekeeping(store)
    # Whatever the stub picks first, the user has already asked to shadow-test.
    for suggestion in store.list_suggestions(status="new"):
        store.set_status(suggestion["id"], STATUS_SHADOW)

    run(ha_config_dir, store, fake_client, llm_provider="ollama", llm_preferences=True)
    assert store.list_suggestions(status="suppressed") == []


def test_explanations_reach_the_card(ha_config_dir, store, fake_client, monkeypatch):
    _use_stub(monkeypatch, _Stub())
    run(ha_config_dir, store, fake_client, llm_provider="ollama", llm_explain=True)
    explained = [
        s for s in store.list_suggestions(status="new")
        if s["payload"].get("extra", {}).get("explanation")
    ]
    assert explained, "the sentence has to land where the UI reads it"


def test_a_crash_in_any_of_the_four_only_makes_the_run_partial(
    ha_config_dir, store, fake_client, monkeypatch
):
    import amminer.llm.scenes as llm_scenes

    def explode(*args, **kwargs):
        raise RuntimeError("a bug in the scene feature itself")

    monkeypatch.setattr(llm_scenes, "propose_and_verify", explode)
    _use_stub(monkeypatch, _Stub())
    report, _candidates = run(
        ha_config_dir, store, fake_client, llm_provider="ollama", llm_scenes=True,
    )
    assert report.status == "partial"
    assert "scenes" in report.ai_errors
    assert report.surfaced > 0


def test_a_hidden_suggestion_is_not_announced(ha_config_dir, store, fake_client, monkeypatch):
    """Hiding it on the page and still pushing it to a phone would be worse than not hiding it.

    One run, with the preference already in place, so the suppression lands on a
    suggestion the store is seeing for the very first time.  Suppressing on a
    *second* run proves nothing: `suggestions_first_seen_in` filters on
    `seen_count = 1` as well, and that clause alone would exclude it however the
    status filter behaved.
    """
    _use_stub(monkeypatch, _Stub())
    store.add_preference("Never automate the hallway.")

    report, _candidates = run(
        ha_config_dir, store, fake_client, llm_provider="ollama", llm_preferences=True,
        notify_on_new_suggestions=True,
    )
    hidden = store.list_suggestions(status="suppressed")
    assert hidden, "this test needs something to have been hidden"
    assert all(s["seen_count"] == 1 for s in hidden), "must be a first sighting"

    last_run = store.last_run()
    announced = {s["id"] for s in store.suggestions_first_seen_in(last_run["id"])}
    assert not ({s["id"] for s in hidden} & announced)
    assert report.notified["new"] == len(announced)


def test_the_rule_the_user_amended_is_the_one_applied(
    ha_config_dir, store, fake_client, monkeypatch
):
    """Editing a preference has to change what gets hidden, not just what is displayed."""
    stub = _use_stub(monkeypatch, _Stub())
    run(ha_config_dir, store, fake_client)
    _dismiss_housekeeping(store)
    run(ha_config_dir, store, fake_client, llm_provider="ollama", llm_preferences=True)

    learned = store.list_preferences()[0]
    assert learned["rule"] == "Never automate the hallway."
    store.update_preference(learned["id"], "Only the hallway lamp, not the whole hallway.")

    stub.prompts.clear()
    run(ha_config_dir, store, fake_client, llm_provider="ollama", llm_preferences=True)

    matching = [p for p in stub.prompts if '"suggestions"' in p and '"preferences"' in p]
    assert matching, "the matcher must have been asked"
    assert "Only the hallway lamp, not the whole hallway." in matching[-1]
    assert "Never automate the hallway." not in matching[-1]
    # And relearning did not quietly put the model's wording back.
    assert store.list_preferences()[0]["rule"] == (
        "Only the hallway lamp, not the whole hallway."
    )


def test_a_preference_written_by_hand_is_applied_without_any_dismissals(
    ha_config_dir, store, fake_client, monkeypatch
):
    stub = _use_stub(monkeypatch, _Stub())
    store.add_preference("Nothing in the bathroom, ever.")

    run(ha_config_dir, store, fake_client, llm_provider="ollama", llm_preferences=True)
    matching = [p for p in stub.prompts if '"suggestions"' in p and '"preferences"' in p]
    assert matching, "a hand-written preference must be applied on its own"
    assert "Nothing in the bathroom, ever." in matching[-1]


def _accepting_scene(monkeypatch):
    """Make the scene step actually produce something to apply.

    Without this the stub proposes no grouping, `grouped.accepted` is empty and
    the apply step never runs at all - so a test that breaks it proves nothing.
    """
    import amminer.llm.scenes as llm_scenes
    from amminer.miners.base import Action, Candidate, Trigger

    scene = Candidate(
        miner="scene", title="Bedtime",
        triggers=[Trigger(kind="time", at="22:00:00")],
        actions=[Action(service="light.turn_off", entity_id="light.kitchen")],
        score=0.8,
    )
    scene.backtest = {"passed": True, "precision": 1.0}

    def fake_propose(*args, **kwargs):
        result = llm_scenes.SceneResult(proposed=1)
        result.scenes.append(
            llm_scenes.Scene(name="Bedtime", reason="r", members=[],
                             candidate=scene, accepted=True)
        )
        return result

    monkeypatch.setattr(llm_scenes, "propose_and_verify", fake_propose)
    return scene


@pytest.mark.parametrize(
    ("module_name", "attribute"),
    [
        ("amminer.llm.areas", "apply_inferences"),
        ("amminer.llm.scenes", "apply_scenes"),
        ("amminer.llm.explain", "apply_explanations"),
    ],
)
def test_a_crash_while_applying_an_ai_result_does_not_end_the_run(
    ha_config_dir, store, fake_client, monkeypatch, module_name, attribute
):
    """`run_ai` wraps the model call; the step that applies its result needs it too."""
    import importlib

    _accepting_scene(monkeypatch)
    module = importlib.import_module(module_name)

    def explode(*args, **kwargs):
        raise RuntimeError(f"a bug in {attribute}")

    monkeypatch.setattr(module, attribute, explode)
    _use_stub(monkeypatch, _Stub())
    report, _candidates = run(
        ha_config_dir, store, fake_client, llm_provider="ollama",
        llm_areas=True, llm_scenes=True, llm_explain=True,
    )
    assert report.status != "error"
    assert report.surfaced > 0, "the deterministic miners must still deliver"
    assert store.list_suggestions(status="new")


def test_a_model_that_is_down_does_not_wipe_the_preferences_you_have(
    ha_config_dir, store, fake_client, monkeypatch
):
    """`learn` reports a timeout as an empty list, and empty means 'withdraw them all'."""
    _use_stub(monkeypatch, _Stub())
    run(ha_config_dir, store, fake_client)
    _dismiss_housekeeping(store)
    run(ha_config_dir, store, fake_client, llm_provider="ollama", llm_preferences=True)
    assert store.list_preferences(), "this test needs a preference to have been learned"

    _use_stub(monkeypatch, _Stub(raises=True))
    report, _candidates = run(
        ha_config_dir, store, fake_client, llm_provider="ollama", llm_preferences=True,
    )
    assert store.list_preferences(), "a bad night must not delete what was learned"
    assert any("could not be refreshed" in d for d in report.degradations)


def test_a_dismissed_scene_is_not_surfaced_again(
    ha_config_dir, store, fake_client, monkeypatch
):
    """A scene's id is a hash of its parts, so it comes back identical every run.

    Mined candidates are filtered against the dismissal list before backtesting;
    scenes are built afterwards, so without their own filter a dismissed scene
    kept being counted as surfaced for ever.
    """
    scene = _accepting_scene(monkeypatch)
    _use_stub(monkeypatch, _Stub())

    first, _ = run(ha_config_dir, store, fake_client, llm_provider="ollama", llm_scenes=True)
    assert store.get_suggestion(scene.id) is not None, "the scene must reach the store"
    store.dismiss(scene.id, "not a routine I have")

    second, _ = run(ha_config_dir, store, fake_client, llm_provider="ollama", llm_scenes=True)
    assert scene.id not in {s["id"] for s in store.list_suggestions(status="new")}
    assert second.surfaced == first.surfaced - 1


# --- temporal holdout validation ------------------------------------------
def test_a_real_habit_is_validated_on_a_holdout_it_was_not_mined_from(
    ha_config_dir, store, fake_client
):
    """The default fixture has 45 days of history - enough for a real holdout."""
    report, _candidates = run(ha_config_dir, store, fake_client)
    assert report.status == "ok"

    validated = [
        s for s in store.list_suggestions(status="new")
        if (s["payload"].get("backtest") or {}).get("validation") == "holdout"
    ]
    assert validated, "at least the strong 06:30 habit should clear a real holdout"
    for suggestion in validated:
        backtest = suggestion["payload"]["backtest"]
        assert backtest["holdout_days"] >= Options().backtest_min_holdout_days
        assert "Validated on" in backtest["validation_note"]
    assert not any("held out to validate" in d for d in report.degradations), (
        "45 days is enough history; this run must not claim otherwise"
    )


def test_short_history_falls_back_to_in_sample_validation_and_says_so(
    tmp_path, store, fake_client
):
    """Too little history for a trustworthy holdout: fall back, and say so."""
    (tmp_path / ".storage").mkdir(parents=True)
    (tmp_path / "configuration.yaml").write_text("recorder:\n  purge_keep_days: 60\n")
    build_default_fixture(tmp_path / "home-assistant_v2.db", days=10, seed=3)

    report, _candidates = run(tmp_path, store, fake_client)
    assert report.status == "ok"
    assert any("held out to validate" in d for d in report.degradations)

    for suggestion in store.list_suggestions(status="new"):
        backtest = suggestion["payload"].get("backtest") or {}
        if backtest.get("holdout_evaluated"):
            assert backtest["validation"] == "in_sample"
            assert "not enough held-out history" in backtest["validation_note"]
